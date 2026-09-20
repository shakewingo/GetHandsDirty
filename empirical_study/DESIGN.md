# Harness transfer study design

## User intent

Inherit the current branch, implement the previously discussed improvements in
stages, and preserve changes and experimental validation for human review.
The target is a tiny local agent's accuracy and efficiency. The user authorized
planning and execution together; phase reports provide reviewable checkpoints.

## Approach

Keep the existing ReAct loop, model, request fit check, raw evidence, and independent
verifiers. Add small opt-in interventions. A full rewrite would confound the study;
prompt-only changes cannot implement output elision or reliable diagnostics.

No external API spend, publishing, merges, or changes to model weights. Use the
existing `transformer-practice` Python environment and local Qwen2.5-7B GGUF.
Preserve the pre-existing untracked `agent_from_scratch/memory.py` untouched and
uncommitted. Work on the explicitly requested new branch in the existing checkout.

## Components

1. **Evaluation:** a fixed general-file-tool pilot with fresh disposable workspaces,
   scripted reference trajectories, independent outcome checks, full traces,
   manifests/hashes and reproducible summaries. Use task-appropriate structured
   tools without unrestricted shell or live web; do not call this a shell benchmark.
   Include no-op, nested target selection, search, invalid JSON, recovery, and
   long-history retention. Long-history cases explicitly label supplied history.
2. **Elision:** off by default. Above a soft token threshold, replace only old,
   successful, bulky, re-readable observation bodies in disposable actor input.
   Preserve envelopes, cursors, version and truncation metadata, recent batches,
   unsent observations, and raw transcript. Summaries still receive raw evidence.
   Elision must reduce measured actor tokens; summarization remains the fallback.
3. **Planning:** off by default. A bounded `update_plan` tool stores at most five
   short steps and a completion condition for the current user turn. Inject only
   the latest plan into model input; don't accumulate injected copies in history.
   Keep plans independent across turns/agents. Planning is guidance, not a success
   oracle. Skip simple questions by instruction rather than a separate classifier.
4. **Action support:** individually switchable bounded literal file search,
   repeated-success reminder, and Python/JSON syntax diagnostics after file edits.
   No automatic test-suite execution and no correctness claims from syntax alone.
   Retain existing sequential batches; concurrency is not justified before timing.

## Invariants

- Experimental flags default off, preserving baseline behavior.
- Summaries and plans never replace the independent verifier.
- Tool call/result pairing and raw transcript stay intact.
- Never hide errors, unseen results, or pagination/truncation metadata through elision.
- All actual model calls, including summaries, count toward existing limits.
- Study records actual inputs and intervention configuration.
- Do not modify source during a real-model panel; freeze each panel's source hashes.
- Run model panels serially to avoid local accelerator contention.
- Keep failure/error metrics distinct from missing measurements.

## Acceptance and limits

Each component needs focused behavioral tests and a green existing suite. Run
real-model paired pilots before adoption. Retain defaults off if evidence is mixed
or task coverage is too small. Publish negative findings and inactive mechanisms.
No statistical generalization from a small greedy development suite, no inference
that fewer turns mean improvement, and no claim of an optimal harness.
