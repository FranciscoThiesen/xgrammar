/*!
 * \file bench_mask_gen.cc
 * \brief Benchmark for grammar compilation and per-token mask generation.
 *
 * This is the C++ counterpart of examples/benchmark/bench_grammar_compile_mask_gen.py: it
 * measures the two costs that structured generation adds to a decode loop, without the
 * Python, torch and tokenizer layers in between, so that changes to the C++ core can be
 * attributed.
 *
 * Usage:
 *   bench_mask_gen <vocab.txt> <schema.json> <tokens.txt> [iterations]
 *
 *   vocab.txt   one encoded token per line, in token-id order
 *   schema.json a JSON schema
 *   tokens.txt  one token id per line: a decode sequence that the schema accepts
 */

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ToUs(Clock::duration d) {
  return std::chrono::duration<double, std::micro>(d).count();
}

std::vector<std::string> ReadLines(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "cannot open " << path << std::endl;
    std::exit(1);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(line);
  }
  return lines;
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "cannot open " << path << std::endl;
    std::exit(1);
  }
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// Median plus the extremes: mask generation is latency-sensitive, so the tail matters as
// much as the average.
struct Stats {
  double median, min, max, mean;
};

Stats Summarize(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
  return Stats{
      xs[xs.size() / 2], xs.front(), xs.back(), sum / static_cast<double>(xs.size())
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: bench_mask_gen <vocab.txt> <schema.json> <tokens.txt> [iterations]"
              << std::endl;
    return 1;
  }
  const std::string vocab_path = argv[1];
  const std::string schema_path = argv[2];
  const std::string tokens_path = argv[3];
  const int iterations = argc > 4 ? std::atoi(argv[4]) : 20;

  const std::vector<std::string> encoded_vocab = ReadLines(vocab_path);
  const std::string schema = ReadFile(schema_path);
  std::vector<int32_t> token_ids;
  for (const std::string& line : ReadLines(tokens_path)) {
    if (!line.empty()) token_ids.push_back(static_cast<int32_t>(std::stol(line)));
  }

  const int vocab_size = static_cast<int>(encoded_vocab.size());
  xgrammar::TokenizerInfo tokenizer_info(encoded_vocab, xgrammar::VocabType::BYTE_LEVEL, vocab_size);

  std::cout << "vocab size: " << vocab_size << ", decode sequence: " << token_ids.size()
            << " tokens, iterations: " << iterations << std::endl;

  // ---- grammar compilation -------------------------------------------------
  // A fresh compiler per iteration, so the schema is never served from the compiler's
  // grammar cache: this is the cost an agentic workload pays when every request carries a
  // new schema. The cache flag stays on, as it is by default, because it also enables the
  // compiler's internal rule-level cache. Threads are pinned to 1 so the number is the work
  // done rather than the size of the pool.
  std::vector<double> compile_us;
  for (int i = 0; i < iterations; ++i) {
    xgrammar::GrammarCompiler compiler(tokenizer_info, /*max_threads=*/1, /*cache_enabled=*/true);
    const auto start = Clock::now();
    volatile auto compiled = compiler.CompileJSONSchema(schema);
    compile_us.push_back(ToUs(Clock::now() - start));
    (void)compiled;
  }
  const Stats compile = Summarize(compile_us);


  // ---- per-token mask generation -------------------------------------------
  xgrammar::GrammarCompiler compiler(tokenizer_info, /*max_threads=*/1, /*cache_enabled=*/true);
  const xgrammar::CompiledGrammar compiled_grammar = compiler.CompileJSONSchema(schema);

  const int32_t bitmask_size = xgrammar::GetBitmaskSize(vocab_size);
  std::vector<int32_t> bitmask(static_cast<size_t>(bitmask_size));
  int64_t shape[1] = {bitmask_size};
  DLTensor bitmask_tensor{};
  bitmask_tensor.data = bitmask.data();
  bitmask_tensor.device = DLDevice{kDLCPU, 0};
  bitmask_tensor.ndim = 1;
  bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
  bitmask_tensor.shape = shape;
  bitmask_tensor.strides = nullptr;
  bitmask_tensor.byte_offset = 0;

  uint64_t checksum = 14695981039346656037ULL;
  // Warm up: the first matcher touches lazily built structures in the compiled grammar.
  {
    xgrammar::GrammarMatcher warmup(compiled_grammar);
    for (int32_t id : token_ids) {
      warmup.FillNextTokenBitmask(&bitmask_tensor);
      // FNV-1a over every mask produced, outside the timed region: printed below so that a
      // change meant to be behaviour-preserving can be checked against the previous build.
      for (int32_t word : bitmask) {
        checksum =
            (checksum ^ static_cast<uint64_t>(static_cast<uint32_t>(word))) * 1099511628211ULL;
      }
      if (!warmup.AcceptToken(id)) {
        std::cerr << "warmup: the grammar rejected token " << id
                  << "; the decode sequence must be accepted by the schema" << std::endl;
        return 1;
      }
    }
  }

  std::vector<double> per_token_us;
  std::vector<double> per_pass_us;
  per_token_us.reserve(token_ids.size() * static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    xgrammar::GrammarMatcher matcher(compiled_grammar);
    const auto pass_start = Clock::now();
    for (int32_t id : token_ids) {
      const auto start = Clock::now();
      matcher.FillNextTokenBitmask(&bitmask_tensor);
      per_token_us.push_back(ToUs(Clock::now() - start));
      matcher.AcceptToken(id);
    }
    per_pass_us.push_back(ToUs(Clock::now() - pass_start));
  }
  const Stats mask = Summarize(per_token_us);
  const Stats pass = Summarize(per_pass_us);

  std::printf("\n%-34s %10s %10s %10s %10s\n", "metric", "median", "mean", "min", "max");
  std::printf(
      "%-34s %10.1f %10.1f %10.1f %10.1f\n",
      "grammar compile (us)",
      compile.median,
      compile.mean,
      compile.min,
      compile.max
  );
  std::printf(
      "%-34s %10.2f %10.2f %10.2f %10.2f\n",
      "mask gen per token (us)",
      mask.median,
      mask.mean,
      mask.min,
      mask.max
  );
  std::printf(
      "%-34s %10.1f %10.1f %10.1f %10.1f\n",
      "mask gen per sequence (us)",
      pass.median,
      pass.mean,
      pass.min,
      pass.max
  );
  std::printf("\nmask checksum: %016llx\n", static_cast<unsigned long long>(checksum));
  return 0;
}
