# Harness design lessons from "An Empirical Study of Harness Design for Coding Agents"

Paper: Fan, Zhang et al., [arXiv 2609.20804](https://arxiv.org/abs/2609.20804), September 17, 2026
(UMass Amherst, Emory, UNC Charlotte; work done at Zoom). Read September 20, 2026.
Implementation plan derived from this note: [EMPIRICAL_STUDY_PLAN.md](EMPIRICAL_STUDY_PLAN.md).

## What the paper does

A small from-scratch ReAct harness with a fixed loop. Three components are varied one at a
time, and everything else is held fixed as a common substrate: workspace guard,
read-before-write, post-edit diagnostics and stuck detection.

- **Context management**, five tiers built from three mechanisms: M1 elision (replace a stale
  tool output with a stub), M2 recall (store the elided original, expose `recall_event(id)`),
  M3 summarization (fold old events into a running summary with a separate tool-free call).
  T0 none, T1 = M1, T2 = M1+M2, T3 = M3, T4 = all three, staged: elide at a soft threshold B₁,
  summarize only above a hard threshold B₂.
- **Planning**: an `update_plan` todo tool whose plan is stored *outside* history and
  re-injected before every model call.
- **Action space**: eight predefined tools (read/write/edit/list/glob/grep/web_fetch/bash)
  versus bash only.

Four models (Nemotron-3 30B / 120B / 550B, Mistral-Medium-3.5-128B), SWE-Bench Verified (500)
and Terminal-Bench 2.1 (89), budgets 32k/64k/96k/128k, 176 settings. Success rate and cost per
task, paired exact McNemar tests with Benjamini–Hochberg FDR 0.05, plus trajectory-level
analysis with an LLM judge (94.2% raw agreement, weighted κ 0.929 against three humans over
15,610 labels).

## Where this agent sits on the paper's axes

| Axis | Paper range | This agent |
|---|---|---|
| Context window | 32k–128k | **8,000** (`config.N_CTX`), 5,952 usable after the 2,048 output reserve |
| Model capability | 30B–550B | **Qwen2.5-7B-Instruct Q4_K_M** |
| Context strategy | T0–T4 | **T3** — summarization only (`compact.py`), no elision |
| Planning | on / off | **absent** |
| Action space | predefined / bash-only | predefined tools; no content search or glob |

Both axes are **outside the studied range, on the harder side**. Every conclusion below is
labelled as direct evidence or as extrapolation to this corner.

## Finding 1 — context management matters most when the window is tight

Managed-minus-T0 success gap on SWE-Bench shrinks 35.7 → 15.9 → 5.5 → 2.7 points from 32k to
128k. The mechanism is overflow: T0 loses 78.7% of SWE-Bench tasks to window overflow at 32k
and 8.7% at 128k, while **every managed tier overflows on zero tasks at every budget**.
Trajectory analysis: management extends runs (median 20–30 → 50–180 turns at 32k) without
changing the phase mix at comparable turns.

**For this agent (extrapolation, strong):** at 8k, compaction decides whether a run reaches
its end at all. Stage 3's priority was right. The overflow rate — runs ending with
`context_limit` — should be reported as its own eval metric with a target of zero.

## Finding 2 — stage cheap elision before summarization (T4)

T4 matches T1–T3 on success and has the lowest cost in 7 of 8 model–benchmark panels, the lowest
peak-context/window ratio at all four budgets, and fewer M3 calls than T3 at every budget.
Early rule-based elision handles most pressure before a summary call is needed.

**For this agent (direct mechanism, extrapolated size):** a summary call here costs a slot of
the turn's model-call budget, not money: `max_compact_calls=4` is charged against
`max_iterations=20`. Elision costs zero model calls.

The existing trigger already sits at the paper's B₂. Compaction starts when
`remaining < context_margin_tokens + compact_headroom_tokens = 768`, i.e. prompt > 5,184 of
5,952 usable tokens ≈ **0.87**, against the paper's 0.85. The hard fit gate is at ≈ 0.96. What
is missing is **B₁ with its cheap action**: at 0.6 of usable (≈ 3,571 tokens), replace bulky
tool outputs outside the two most recent batches with stubs, re-measure, and only then let the
existing trigger decide whether to summarize. The paper keeps the preamble plus a recent
window of at least two turns verbatim and caps tool results at 24k characters. The
equivalent cap here would be 2–4k characters. `read_file` returns up to 16,000 characters, so one
read can take more than half the usable window. That cap is left unchanged and noted as a
follow-up knob.

## Finding 3 — do not build recall (M2)

T2 vs T1 differ only in M2: over 32 paired comparisons, 15 wins, 14 losses, 3 ties, mean
−0.36 points. 36 of 64 T2/T4 settings (56.3%) never call `recall_event`; the median rate is
zero; the mean falls from 0.540 calls/task at 32k to 0.007 at 128k. Use concentrates in the
**weakest model at the tightest window**, and even the heaviest-use cell (30B, Terminal-Bench,
32k, T2: 4.326 calls/task) scores 3.37 points below T1.

**For this agent (extrapolation, but the heavy-use corner still loses):** elision stubs are
irreversible and say "re-read or re-run to get it again". No external store and no retrieval
tool. This bears on mechanism 2, "Store and retrieve", in
[CONTEXT_STATE_DESIGN.md](CONTEXT_STATE_DESIGN.md#keeping-useful-tool-evidence-in-a-small-context).
The paper's evidence argues against building it for in-trajectory history. Cross-session
memory (Stage 4A–4B) is a different question the paper does not test.

## Finding 4 — planning is an accuracy scaffold for weak models

Nemotron-3 30B at T4/128k: planning adds 11.6 points on SWE-Bench and 4.5 on Terminal-Bench.
Without it the median SWE-Bench run collapses from 40 turns to 5, and 58.4% of runs stall in
localization versus 10.4%. For strong models, planning mainly cuts cost by trimming redundant
verification (550B: 108 → 74 median turns, −30% cost, −2.0 points).

**For this agent (extrapolation, strong: 7B is weaker than 30B):** planning is expected to be
an accuracy component here, not a cost saver. Two design details matter at 8k:

1. The plan lives **outside the transcript** and is re-injected fresh before each call, so it
   costs O(1) tokens, not O(turns), and survives compaction without being summarized.
2. The cost is real: for 30B, planning raised turns +293% and tool calls +474%. If the turn
   budget stays tight, an ablation measures planning's cost without its benefit.

## Finding 5 — predefined tools scaffold models that cannot drive a shell

Bash-only drops 30B from 25.2% to 10.2% on SWE-Bench and from 13.5% to 3.4% on Terminal-Bench.
66% of its bash-only Terminal-Bench runs end after the model emits calls to tools that are
not registered. The reverse holds for bash-capable models: 550B gains 3.6 points at −53%
cost with bash only. For the weakest model, the dominant failure stage is **locating the right
file**: 56.3% of unresolved runs, rising to 76.6% with bash only.

**For this agent:** keep the predefined tools. The concrete gap is search. The registry has
`list_files` but no content search and no glob, which are the two tools that serve the
weak-model failure stage directly.

## Fixed substrate worth copying

These were held constant in every condition, so the paper gives **no isolated effect size**
for them. Adopt them as cheap hygiene, not as measured wins.

- **Stuck detection with two kinds of streak.** The paper counts byte-identical calls (same
  name and arguments) whatever their status, and separately identical failing calls. It injects
  a reminder once per streak (at 5), and terminates only when the identical failing streak keeps
  growing (at 8). The current `repeated_failure` counts consecutive identical failures only. Any
  success resets it, and it stops the turn at 3 without warning first. Two changes: add the
  success-streak check ("you already have this result"), and warn the model before stopping.
  The thresholds scale down with this agent's 20-call budget (paper: 300 steps).
- **Post-edit diagnostics.** After a write or edit, run a fast read-only check and append the
  findings to the same tool result. The paper uses ruff/pyflakes for Python. Here the standard
  library already covers the useful cases: `compile()` for `.py` and `json.loads` for `.json`,
  and the second matters most for the text/JSON workspaces this agent targets.
- **Tool errors as observations**, never raised into the loop — already true here.

## Evaluation: trajectory analysis, kept domain-agnostic

The paper's most transferable method is reading **trajectories**, not just pass/fail. For each
configuration it plots the fraction of runs still active at every turn, split by phase. That view
is what shows *how* each component changes behaviour: management lengthens runs, planning
keeps weak runs alive and trims strong ones, and bash-only changes action granularity.

This agent is not committed to being a coding agent, so its trajectory metrics avoid coding
concepts (no "terminated without an edit", no localization/patch stages):

| Metric | Why |
|---|---|
| stop-reason distribution, especially `context_limit` rate | the paper's overflow mechanism; target 0 |
| actor requests per run (median) and % of runs still active at request *k* | trajectory length and survival |
| peak prompt tokens / usable window | how hard a run pressed the window |
| elided messages, summary calls, plan updates, stuck reminders per run | mechanism usage, as in the paper's Figure 5(b) |
| action mix by tool category (explore / modify / execute / plan) per request index | a rule-based, domain-agnostic stand-in for the paper's LLM-judged phases |
| model requests and tokens | cost for a local model |

The paper's phase labels come from a much stronger LLM judge. A 7B model cannot validate its
own trajectories that way, so this agent uses rule-based categories from tool names.

**Statistics at this scale.** The paper's paired exact McNemar test is the right design for
comparing two configurations on the same tasks, but Terminal-Bench's 89 tasks often failed to
reach significance, and the dev suite here has **17**. Report paired outcomes (both pass / only
A / only B / both fail) and the exact p-value, and read direction, not significance. The
suite's `allowed_tools` allowlist and recorded web replay already avoid the paper's search-
leakage concern. The frozen Stage 2B suite uses legacy file tools, has no `edit_file` and runs
only fixed shell commands, so it cannot measure Finding 5 or the post-edit diagnostics.

## Order of work

1. T4: elision at B₁ = 0.6 of usable before the existing ~0.87 summary trigger.
2. Planning: `update_plan`, plan held outside history and re-injected, reminder until a plan exists.
3. Eval trajectory metrics and paired comparison (domain-agnostic, above).
4. Stuck detection with success streaks and warn-before-stop; post-edit `.py`/`.json` diagnostics.
5. Search tools: `glob_files` and `grep_text`.

Every mechanism in steps 1–2 gets a switch in `AgentLimits`, so each one can be ablated and every
run's `settings` records which were on.

## Size note

HEAD is 2,804 physical core lines against the revised 3,000 alarm in
[context-memory.md](context-memory.md#design-boundary-review-2584-physical-lines), which
projected about 2,950 at the end of the required stages *including* Stage 4A's `memory.py`.
This work was not in that projection. The plan records the count after each step; crossing
3,000 is the review trigger defined there, not a reason to compress code.
