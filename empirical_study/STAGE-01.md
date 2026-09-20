# Stage 01 — fixed development pilot

Implemented an 8-case general-file-tool pilot, explicit history fixtures, independent
artifact and source checks, immutable output directories and source/settings manifests.

Seven verifier tests pass, including wrong answers, guessed content behind empty reads,
write-before-source, recovery bypass, no-op rewrites, and history metric accounting.
The initial tests failed on the missing runner; review regressions failed before fixes.

Independent review found three important weaknesses in v1: path-only evidence,
recovery without a fault, and historical calls charged as new execution. All are fixed
in suite v2. The incomplete v1 model pilot was terminated and is **not a baseline**.
Its evidence remains under `outputs/empirical-study/stage-01/baseline`.

Ruling: run final real-model comparisons for all profiles from a common, frozen code
revision after component tests. This avoids repeated baselines across implementation
changes and gives cleaner attribution. Each component stage records implementation
validation first; Stage 05 links the matched model evidence back to each stage.

Scope: no unrestricted shell, live network, concurrency or statistical generalization.
The pilot tests transfer to general file operations, not a SWE-Bench reproduction.
