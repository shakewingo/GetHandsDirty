# Stage 00 — frozen baseline

Base: `d7f3887`. Python: existing transformer-practice environment.

- 206 unit/integration tests passed (4.083 s).
- Unchanged restricted-tool panel: **10/17**, 44 model calls, 93.66 s.
- Input/output tokens: 53,560 / 1,432; total 54,992.
- Six parse errors and six invalid-call counts (these categories can overlap).
- One task with unintended changes. False-completion claims are **unreviewed**.
- Raw evidence: `outputs/empirical-study/stage-00/legacy-baseline/`.

This is a fresh greedy development sample, not the historical 12/17 condition.
It uses legacy byte-offset file tools and fixed shell/web fixtures. Do not interpret
it as a score for the general tool interface or compare different conditions as
if only one component changed.

| Task | Pass | Stop |
|---|---|---|
| profile_output | False | final_response |
| second_chunk_code | True | final_response |
| header_only | False | final_response |
| full_document | False | final_response |
| flat_config_update | False | final_response |
| nested_config_update | True | final_response |
| already_correct | False | final_response |
| check_fix_clean | True | final_response |
| check_fix_fault | True | final_response |
| web_release_clean | False | no_progress |
| web_release_fault | False | no_progress |
| truncated_page | True | final_response |
| blocked_path | True | final_response |
| missing_profile | True | final_response |
| timeout_stop | True | final_response |
| calculator_regression | True | final_response |
| direct_ready | True | final_response |

Commands:

```sh
python -m unittest discover -s agent_from_scratch/tests -v
python -m agent_from_scratch.evals.run --output outputs/empirical-study/stage-00/legacy-baseline --seed 11
```

Decision: retain this regression panel and establish a separate fixed general-tool
pilot before interpreting intervention effects.
