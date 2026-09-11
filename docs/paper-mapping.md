# Paper Mapping

Each paper section is implemented at the following location.

## Paper → Code

| Paper section | Subject | Implementation |
|---|---|---|
| §3.1 | Formal representation (Relation, Attribute, Node, Edge, ProceduralGraph, Edit) | `methodos/schema.py` |
| §3.2 | `Match(a, V)` | `methodos/graph.py::match_node` |
| §3.2 | Neighborhood extraction | `methodos/graph.py::neighborhood` |
| §3.2 | `generate_guidance` (paper Eq. 2) | `methodos/guidance.py::generate_guidance` |
| §3.2 | `w=3` trajectory window | `methodos/guidance.py::format_trajectory_window` |
| §3.2 | `h=2` neighborhood radius | `methodos/graph.py::neighborhood` (default `h=2`) |
| §3.3 step 1 | Diagnostic rollout | `methodos/evolution.py::run_rollout` + `_collect_diagnostic_traces` |
| §3.3 step 2 | Refiner LLM | `methodos/evolution.py::propose_edits` + `REFINER_SYSTEM_PROMPT` |
| §3.3 step 3 | Validate candidate | `methodos/evolution.py::validate_candidate` |
| §3.3 step 4 | Rejection memory | `methodos/evolution.py::RejectionMemory` |
| §3.3 | `Tail_{L_max}` truncation | `methodos/evolution.py::tail_concat` (also in `repo.tail_tokens`) |
| §3.3 | Algorithm 1 orchestration | `methodos/evolution.py::EvolutionEngine.run` |
| §4 | Settings (`h=2`, `w=3`, 1-hop neighborhood) | `methodos/adapter.py::PGAdapter.__init__` defaults |
| §5.1 | Main results across benchmarks | `eval/hotpotqa/` (in-repo); other benchmarks via user-implemented harnesses |
| §5.2 | Long-horizon resilience (EnterpriseArena) | User-implemented harness; see `docs/evaluation.md` |
| §5.3 | Construction modes (5 strategies) | `methodos/evolution.py::EvolutionEngine.run` (algorithm 1) covers modes 1, 3, 5 |
| §5.4 | Self-evolution over rounds | `methodos/evolution.py::EvolutionEngine.run` |
| §5.5 | Localization vs full graph | Demonstrated by `guidance_hops` parameter in `PGAdapter` |
| App. B.6 | Algorithm 1 pseudocode | `methodos/evolution.py::EvolutionEngine.run` (verbatim translation) |
| App. B.6.1 | `PrepareCandidate` structural checks | `methodos/evolution.py::validate_candidate` + `methodos/graph.py::validate` |
| §B.4 | 5 relations (leads_to, requires, triggers, converges_to, replaces) | `methodos/schema.py::Relation` (StrEnum; `replaces` is operational, 4 paper relations are required) |
| §B.5 | Node kind ∈ {`ACTION`, `STATUS`} | `methodos/schema.py::Node.kind` (Literal) |
| §B.5 | Refiner system prompt — 7 rules (action matching, transitions, mandatory guidance/pitfalls, generality, node-id compat, structure) | `methodos/evolution.py::REFINER_SYSTEM_PROMPT` |

## Notation → Identifier

| Paper symbol | Python name | Location |
|---|---|---|
| `G` (procedural graph) | `ProceduralGraph` | `methodos/schema.py` |
| `N` (set of nodes) | `graph.nodes` (dict) | `methodos/schema.py` |
| `E` (set of edges) | `graph.edges` (list) | `methodos/schema.py` |
| `T` (set of terminals) | `graph.terminal_ids` (set) | `methodos/schema.py` |
| `a` (action name) | string passed to `match_node` | `methodos/graph.py` |
| `V` (nodes dict) | `graph.nodes` | — |
| `h` (neighborhood hops) | `guidance_hops` parameter | `methodos/adapter.py` |
| `w` (trajectory window) | `trajectory_window` parameter | `methodos/adapter.py` |
| `g_t` (guidance text at step t) | return value of `generate_guidance` | `methodos/guidance.py` |
| `s_t` (agent state at step t) | `AgentState` | `methodos/adapter.py` |
| `q_t` (context block) | `AgentState.context` | `methodos/adapter.py` |
| `ψ` (LLM guidance model) | `LLMClient.complete` (called by `generate_guidance`) | `methodos/llm.py` |
| `T_max` (max token budget) | `l_max_tokens` parameter | `methodos/evolution.py::EvolutionEngine.__init__` |
| `H_rejected` (rejection history) | `RejectionMemory` | `methodos/evolution.py` |
| `S_val` (validation score) | `score_validation` (mean of rollout scores on val tasks) | `methodos/evolution.py` |

## Paper claims → Evidence in code

| Paper claim | Where in code |
|---|---|
| Procedural graphs are the procedural counterpart of knowledge graphs | `methodos/schema.py::ProceduralGraph` (typed triplets, same shape as a KG minus literal values) |
| `Match(a, V)` does exact-string matching | `methodos/graph.py::match_node` |
| `generate_guidance` is paper Eq. 2 | `methodos/guidance.py::generate_guidance` (locate done by caller; extract+generate done here) |
| Self-evolution follows Algorithm 1 | `methodos/evolution.py::EvolutionEngine.run` |
| Rejection memory is bounded FIFO | `methodos/evolution.py::RejectionMemory` (uses `collections.deque(maxlen=...)`) |
| `Tail_{L_max}` truncates from beginning, preserves end | `methodos/evolution.py::tail_concat` |
| Subgraphs are 2-hop neighborhoods (default) | `methodos/adapter.py::PGAdapter.__init__` default `guidance_hops=2` |
| Trajectory window is 3 steps (default) | `methodos/adapter.py::PGAdapter.__init__` default `trajectory_window=3` |
| Refiner is invoked with rejected-candidate history | `methodos/evolution.py::REFINER_SYSTEM_PROMPT` includes a `## Rejected (do not repeat)` section populated from `RejectionMemory` |
| Ties accepted (paper §3.3 Eq. 5: `S_k == S_{k-1}`) | `methodos/evolution.py::EvolutionEngine.run` (`if score >= prev_score: keep`) |

## Deviations from paper

| Aspect | Paper | methodos | Rationale |
|---|---|---|---|
| Graph storage | unspecified | SQLite default, FS optional | Production-grade persistence is not the paper's focus; SQLite is the right default |
| Vector search | described in §5.5 as future work | `SqliteVecIndex` opt-in | We added fuzzy Match for production robustness (paper notes exact-match brittleness as a limitation) |
| Solver signature | unspecified | `async def step(state) -> str` Protocol | Hosts plug in their existing agent loop without modification |
| CLI / HTTP | not in paper | Typer CLI + FastAPI service | Production deployment needs an API surface; not in paper scope but obvious extension |
| Eval harness | paper evaluates 7 benchmarks | HotpotQA only in-repo | Heavy benchmarks need domain-specific simulators (τ-bench, EnterpriseArena) that don't belong in a library |

All other algorithm-level details are implemented verbatim per the
paper's pseudocode (App. B.6).
