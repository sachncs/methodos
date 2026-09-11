# Evaluation

`methodos` ships with one in-repo benchmark harness (`eval/hotpotqa/`)
gated by the `[eval]` extra. This document explains the methodology and
how to wire `methodos` into other paper benchmarks.

## HotpotQA (in-repo)

```bash
pip install methodos[eval]
methodos eval hotpotqa --graph-id my-graph --n 200 --model gpt-4o-mini
# or as a module:
python -m eval.hotpotqa.run --graph-id my-graph --n 200 --model gpt-4o-mini
```

### Methodology

For each task:
1. Construct the same `HotpotQASolver` (ReAct-style LLM + injected
   search callback) for both arms.
2. **With-PG arm**: wrap the solver in a `PGAdapter`. The adapter injects
   `Ψ` guidance into `state.context` before each solver call.
3. **Without-PG arm**: invoke the solver directly with `state.context = ""`.
4. Both arms see identical LLM, identical search callback, identical RNG
   seed (task order is shuffled with `random.Random(seed).shuffle(tasks)`).
5. Score with exact match (case-insensitive, whitespace-normalized) and
   token F1.

### Output

```
HotpotQA eval (n=200)
  with PG:    EM=0.4500  F1=0.5800
  without PG: EM=0.4200  F1=0.5400
  delta:      EM=+0.0300  F1=+0.0400
```

The delta is the contribution of procedural guidance, controlling for
LLM noise.

### Reproducibility

- `temperature=0.0` on all LLM calls (greedy decoding).
- `--seed` controls task order; both arms use identical seed.
- HotpotQA distractor split JSONL is cached at
  `~/.methodos/eval_data/hotpotqa_dev_distractor.jsonl` after first
  download.

### Plug in a real search backend

The default `noop_search` returns `(no results)`. Before evaluating in
production, replace it with a real retriever:

```python
from eval.hotpotqa.solver import noop_search
from eval.hotpotqa import solver as solver_mod


async def real_search(query: str) -> str:
    return await my_retriever.search(query)


solver_mod.noop_search = real_search
```

## Wiring other paper benchmarks

The paper evaluates on six benchmarks besides HotpotQA:
MultiChallenge, GDPval, ALFWorld, τ-bench, BFCL, EnterpriseArena.

`methodos` does not bundle these because each requires a heavy
domain-specific environment (synthetic user simulators, embodied
action spaces, tool APIs). The general pattern for wiring any of them:

1. **Implement the benchmark's task iterator and scorer** in
   `eval/<name>/tasks.py`. Return a list of `Task` and a scoring
   function.
2. **Implement a benchmark-specific `Solver`** in
   `eval/<name>/solver.py`. The solver must satisfy `Solver` Protocol
   (async `step(state) -> str`).
3. **Compose with PGAdapter** — same pattern as HotpotQA: the adapter
   injects guidance into `state.context`.
4. **Wire a CLI command** in `methodos/cli.py` (one `cli.command()` per
   benchmark) that delegates to `eval.<name>.run.run_eval`.

The paired-comparison methodology (same LLM, same RNG seed, with vs
without PG) is identical across benchmarks.

## Property-based checks (no benchmark required)

`tests/test_graph.py` uses `hypothesis` to assert graph invariants
across randomly-generated acyclic graphs:

- `acyclic_graph_has_no_cycle`
- `first_node_reaches_terminal`
- `validate_clean_on_acyclic`
- `apply_empty_edits_idempotent`
- `neighborhood_root_in_subgraph`

These run in <1 second and provide a structural sanity check independent
of any benchmark.

## CI integration

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

- `uv run ruff check .`
- `uv run mypy methodos/ tests/ eval/`
- `uv run pytest --cov=methodos --cov-fail-under=95`

The benchmark harness itself is NOT run in CI (it requires network and
LLM credentials). Run it manually via the commands above.
