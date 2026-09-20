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

Real-model efficacy is pending the frozen Stage 05 panel. Default remains off.
Telemetry adds optional `elided_call_ids` to schema 5 (backward-compatible additive
field); the original schema 5 budget rename and old trace readers remain unchanged.

Known scope: summary input still uses raw observations and its existing fit gate.
Elision cannot rescue an oversized, non-elidable summary input. No recall tool,
summary-input chunker, or semantic claim that omitted facts are unimportant is added.
