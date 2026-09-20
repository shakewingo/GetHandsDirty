# Stage 02 — conservative elision before summary

Implemented opt-in `elision_enabled`, soft threshold 0.6 of prompt capacity after
output reserve/margin, and minimum body length 1,024 characters.

Only old successful complete read/list/search bodies are candidates. Errors,
partial results, cursors, versions, tool envelopes, two recent batches and unseen
observations stay intact. Raw evidence and summary inputs are unchanged. Candidate
views publish only after exact measurement confirms fewer prompt tokens.

Validation: 5 focused tests cover raw preservation, recent/unseen/error/partial
boundaries, no-op measurements, malformed history and actual actor input tracing.
Existing compactor tests remain green. Full suite: 218 tests pass.

Real-model evidence is recorded below and in Stage 05. Default remains off.
Telemetry adds optional `elided_call_ids` to schema 5 (backward-compatible additive
field); the original schema 5 budget rename and old trace readers remain unchanged.

Known scope: summary input still uses raw observations and its existing fit gate.
Elision cannot rescue an oversized, non-elidable summary input. No recall tool,
summary-input chunker, or semantic claim that omitted facts are unimportant is added.

## Matched model evidence (frozen runtime 42c7228)

- Baseline vs elision: strict success **1/8 vs 1/8**; elapsed **156.77 vs 173.73 s**;
  total tokens **72,441 vs 73,674**; model calls **28 vs 27**.
- Nine unique observations were elided. The two supplied-history cases progressed
  from pre-generation context blocks to model execution, but neither task passed.
- In history_retain the model interpreted the elided empty content as missing/empty
  source content and did not reread, despite the elision metadata/instruction.
- No actual summary generation occurred in either panel. Thus this sample does NOT
  establish reduced summarization cost; a medium-pressure case is needed for that.
- Decision: keep opt-in. Investigate explicit nonempty omission markers/limited
  previews and add retention-focused held-out cases before considering adoption.
