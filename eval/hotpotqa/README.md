# HotpotQA Eval Harness

Paired with-PG vs without-PG evaluation on HotpotQA multi-hop questions.

## Install

```bash
pip install methodos[eval]
```

This installs the `datasets` package needed to download HotpotQA on first
run. Subsequent runs reuse the cached JSONL at `~/.methodos/eval_data/`.

## Run

```bash
# Both as Typer CLI and as a Python module:
methodos eval hotpotqa --graph-id my-graph --n 200 --model gpt-4o-mini
python -m eval.hotpotqa.run --graph-id my-graph --n 200 --model gpt-4o-mini
```

The harness:
1. Downloads HotpotQA distractor split (first run only) to
   `~/.methodos/eval_data/hotpotqa_dev_distractor.jsonl`.
2. Loads the procedural graph (if `graph_id` provided) via the configured
   Repository backend.
3. For each of N random tasks (seed=0 default), runs the question with and
   without PG wrapping. Same LLM, same solver, same seed → paired.
4. Prints EM/F1 for both arms and the delta.

## Plug in a real search backend

The default `noop_search` returns `(no results)` — replace it with a real
retrieval client before evaluating on production:

```python
from eval.hotpotqa.run import run_eval

async def real_search(query: str) -> str:
    # call your ColBERT / Elasticsearch / etc.
    return "..."

# Run with the custom search by calling _run_one directly or by
# monkey-patching noop_search in your own entrypoint.
```

## Method

For each task:
- A `HotpotQASolver` is constructed with the configured LLM and search
  callback. The solver implements the `Solver` Protocol via a single
  `async def step(state)` method.
- With-PG run: a `PGAdapter` wraps the solver and injects procedural
  guidance into `state.context` before each call.
- Without-PG run: the solver is invoked directly with an empty context.
- Both arms see identical LLM, search, and RNG seed → the delta is
  attributable to procedural guidance.

## Files

- `tasks.py` — `HotpotQATask`, `load_tasks`, `download_if_missing`
- `solver.py` — `HotpotQASolver` (ReAct-style), `noop_search`
- `run.py` — `run_eval`, CLI entry, paired-aggregation helpers
