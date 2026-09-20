# Reviewer guide

## Read in this order

1. README.md: phase status and evidence rules.
2. STAGE-00.md: inherited baseline, separate from new tasks.
3. STAGE-01.md: independent verifiers and review corrections.
4. STAGE-02.md / STAGE-03.md / STAGE-04.md: opt-in components and tests.
5. STAGE-05.md and stage-05-measurements.json: matched empirical outcomes and explicit scoring caveats.

## Commit boundaries

| Stage | Commits | Review focus |
|---|---|---|
| 00 | `29bf675` | Scope, baseline and protocol |
| 01 | `2e6bada`, `3b8a202` | Verifier false positives, ordered evidence, accounting |
| 02 | `09deae9` | Model view vs raw evidence; fit and recent-window invariants |
| 03 | `97563bf` | Plan bounds, isolation, injection and completion semantics |
| 04 | `42c7228` | Separate interventions and second review fixes |
| 05 | Final `docs(study): report matched harness pilot and adoption decisions` commit | Paired evidence, negative results and adoption decision |

Runtime comparison revision: `42c7228`, inherited from `d7f3887`.
All production defaults stay off. The original `memory.py` remains untracked.

## Replay

Use the existing transformer-practice environment. Choose a fresh output directory:

```sh
python -m unittest discover -s agent_from_scratch/tests -v
python -m agent_from_scratch.evals.harness_study --profile baseline --output outputs/my-study/baseline
python -m agent_from_scratch.evals.harness_study --profile elision --output outputs/my-study/elision
python -m agent_from_scratch.evals.harness_study --profile planning --output outputs/my-study/planning
python -m agent_from_scratch.evals.harness_study --profile elision_planning --output outputs/my-study/elision_planning
python -m agent_from_scratch.evals.harness_study --profile search --tasks direct,nested,search,missing_path --output outputs/my-study/search
python -m agent_from_scratch.evals.harness_study --profile repeat --tasks search,missing_path --output outputs/my-study/repeat
python -m agent_from_scratch.evals.harness_study --profile diagnostics --tasks no_op,nested,json_repair --output outputs/my-study/diagnostics
python empirical_study/analyze.py outputs/my-study empirical_study/my-measurements.json
```

The analyzer rejects mismatched runtime/source/model/seed/fixture manifests and
incomplete panels, and compares targeted profiles only with the matching baseline
subset. It preserves unknown usage. Initialization/model loading is outside summed
turn latency; tool execution, input preparation and compaction are inside it.

## Interpreting the pilot

- Eight handcrafted development cases are not a broad capability benchmark.
- Two cases start with fixture-generated history: they probe history admission and
  retention, not a claim that the model itself produced the earlier trajectory.
- Exact-answer and strict no-write-attempt checks contribute to task success. The
  no-op wording ambiguity and endpoint sensitivity audit are disclosed in Stage 05. Read failed_checks
  before inferring a semantic reasoning failure from a strict failure.
- Random run IDs, file versions and temporary paths differ between runs; greedy
  decoding and a fixed seed do not make all inputs byte-identical. Confirm proposed
  improvements on additional frozen tasks and repetitions before adoption.
- A feature may prevent a context error without improving artifact correctness.
- No timing confidence interval, held-out result or false-completion rate is claimed.
  Final-answer claim review is distinct from automated success scoring.
