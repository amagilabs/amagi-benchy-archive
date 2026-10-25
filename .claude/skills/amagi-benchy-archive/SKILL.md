---
name: amagi-benchy-archive
description: Archive llama-benchy results from one or many machines in a plain folder, join nvidia-smi power samples onto each test, and report a leaderboard, tokens per joule, regressions between runs, and a static HTML page. Use when asked to keep or compare llama-benchy runs over time, rank local LLM setups (model, quant, engine, GPU) on the same test, measure tokens per watt or per joule, check whether a driver, engine or quant change made a model slower, or publish a local-LLM leaderboard page.
---

# amagi-benchy-archive

A companion CLI for [llama-benchy](https://github.com/eugr/llama-benchy). llama-benchy measures
an OpenAI-compatible server and writes a JSON report and an `--emit-progress` JSONL stream. This
tool files those outputs in an archive folder, adds GPU board power from `nvidia-smi` sampled
during the run, and reads the archive back as tables and one HTML page. It does not start
servers or load models, and it does not change how llama-benchy measures.

## When to use this skill

- The user benchmarks local models with llama-benchy and wants results kept, not lost in a terminal.
- The user wants to rank setups (model, quant, engine, GPU, machine) on one test.
- The user asks for tokens per joule, watts during generation, or idle power.
- The user changed a driver, engine build, quant or flag and wants to know if anything got slower.
- The user wants a shareable leaderboard page or a markdown table for a post.

## How to use it

Install once: `pipx install amagi-benchy-archive` (or `pip install -e .` from this repo), and have
`llama-benchy` 0.4.0 or later on PATH (`pipx install llama-benchy`). 0.4.0 is the first release with `--emit-progress`.

```bash
# run llama-benchy with a power sampler and archive the result in one step
amagi-benchy-archive record --gpu "RTX 3090" --engine "llama.cpp b6000" --quant Q4_K_M -- \
    llama-benchy --base-url http://localhost:8080/v1 --model Qwen/Qwen2.5-7B-Instruct \
    --pp 512 2048 --tg 128 --depth 0 4096 --concurrency 1 2

# GPU on another box: sample its nvidia-smi over ssh
amagi-benchy-archive record --smi-cmd "ssh gpubox nvidia-smi" -- llama-benchy --base-url http://gpubox:8080/v1

# files you already have (type is detected); add a power log if one ran alongside
amagi-benchy-archive ingest report.json progress.jsonl --power power.csv --machine gpubox
amagi-benchy-archive power-log -o power.csv          # sampler on its own, Ctrl-C to stop

amagi-benchy-archive runs                            # what is archived
amagi-benchy-archive show 3f2a                       # every test in one run (id prefix is fine)
amagi-benchy-archive leaderboard --pp 2048 --tg 128  # rank setups on one test
amagi-benchy-archive leaderboard --sort tok_per_j --format md
amagi-benchy-archive regressions --fail              # exit 1 if the latest run got slower (CI)
amagi-benchy-archive html -o site/index.html --title "One 3090, every model"
amagi-benchy-archive merge /mnt/laptop/benchy-archive   # pull in another machine's runs
```

The archive folder is `$AMAGI_BENCHY_ARCHIVE`, else `./benchy-archive`; `-a PATH` overrides it.
Each run is `runs/<run_id>.json` plus the original files under `raw/<run_id>/`. The run id is a
hash of the content, so ingesting the same files twice or merging overlapping folders does not
create duplicates. Several machines can share one archive through any file sync.

`record` owns llama-benchy's `--emit-progress`, `--format` and `--save-result` flags. Passing any
of them after `--` is an error. Every other llama-benchy flag passes through untouched.

Exit codes: 0 ok, 1 `regressions --fail` found a regression, 2 usage or input error (bad file,
unknown run id, empty archive, llama-benchy wrote nothing).

## What the numbers mean

- Test labels follow llama-benchy: `pp2048/tg128 @ d4096 c2 [ctx]` is a 2048-token prompt, 128
  generated tokens, 4096 tokens of prior context, 2 concurrent requests, context-load phase
  (only with `--enable-prefix-caching`).
- **gen t/s, prompt t/s, peak t/s, TTFT**: llama-benchy's own figures from its JSON report.
  With only the JSONL, the tool rebuilds them with llama-benchy's formulas. On a real run against
  llama-benchy's mock server, 11 of 12 tests landed within 0.1% of the report and the worst was
  1.1%, because JSONL timestamps are taken when an event is written.
- **watts**: mean GPU board power over the decode windows (first token to end of each request),
  summed across the counted GPUs. **tok/J** is gen t/s divided by those watts. **idle** is the
  mean power sampled before the first request.
- Board power only. CPU, RAM, fans and PSU losses are not in it, so tok/J compares GPUs and
  settings, not whole machines.
- **regressions** compares the latest run of each setup (machine, model, engine, quant) with the
  run before it on every test both share. A change is listed when it is at least `--threshold`
  percent (default 5) and larger than the two runs' combined standard deviation.

## Working with the results

1. Before quoting a number, check `errors` in `show`. A test with failed requests has fewer
   samples behind its mean.
2. Rank only on one test shape. The leaderboard already does this. Do not compare a pp512 result
   against a pp2048 one.
3. If `power` says `-` the run had no samples in its windows: no sampler ran, the clocks of the
   two machines disagree, or a report was ingested without its JSONL (a report alone has no
   per-test timestamps, so no power join is possible).
4. Quote numbers as the tool prints them, with the machine and GPU. Never present one machine's
   result as typical for a GPU model.

## Privacy

Run files store the machine name (default: hostname), GPU label, engine, quant, model, tag and
note, plus `llama-benchy` as the command name without its arguments. They do not store the server
URL or API key. The copies under `raw/` are llama-benchy's files as written, and its JSONL
includes the `base_url`. Pass `--no-raw` to `ingest`, or leave `raw/` out, when sharing an
archive. The HTML page shows machine, GPU, model, engine, quant, tag and dates.

## Limits

- Power needs NVIDIA GPUs and `nvidia-smi`. Without it everything except watts and tok/J works.
- The power join assumes the sampler and llama-benchy share a clock. With `--smi-cmd "ssh ..."`
  the timestamps are taken locally when each sample arrives, which keeps them on one clock.
- Raw `nvidia-smi --format=csv` logs carry local time without a zone. Pass
  `--power-utc-offset` when the log came from a machine in another time zone.
