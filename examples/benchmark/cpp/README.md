# C++ benchmarks

`bench_mask_gen` measures the two costs structured generation adds to a decode loop —
compiling a grammar, and generating the token mask at each step — directly against the
C++ core, without the Python, torch and tokenizer layers in between. It is meant for
attributing a change in `cpp/` to a number.

Build it with the `XGRAMMAR_BUILD_CXX_BENCHMARKS` option:

```bash
mkdir -p build && cd build
echo "set(XGRAMMAR_BUILD_CXX_BENCHMARKS ON)" > config.cmake
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --target bench_mask_gen
```

It takes the inputs as files so that any tokenizer and schema can be measured:

```
bench_mask_gen <vocab.txt> <schema.json> <tokens.txt> [iterations]

  vocab.txt    one encoded token per line, in token-id order
  schema.json  a JSON schema
  tokens.txt   one token id per line: a decode sequence the schema accepts
```

To produce them for a given model:

```python
import json
from transformers import AutoTokenizer

model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

vocab = tokenizer.get_vocab()
ordered = [""] * (max(vocab.values()) + 1)
for token, token_id in vocab.items():
    ordered[token_id] = token
with open("vocab.txt", "w") as f:
    f.write("\n".join(ordered))

instance = json.dumps({"name": "Ada Lovelace", "id": 1815}, separators=(",", ":"))
with open("tokens.txt", "w") as f:
    f.write("\n".join(str(i) for i in tokenizer(instance, add_special_tokens=False)["input_ids"]))
```

The benchmark compiles with the cache disabled and one thread, so the compile column is the
work itself rather than the pool size or a cache hit — the case an agentic workload pays
when every request carries a new schema. It also prints an FNV-1a checksum over every mask
it generates, so a change meant to be behaviour-preserving can be checked against the
previous build.
