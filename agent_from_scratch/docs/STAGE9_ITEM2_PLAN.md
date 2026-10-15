# Stage 9 Item 2 — verified trajectories to SFT examples: implementation guide

Written October 12, 2026, against commit `dbe98ad` (342 tests passing).
Revised the same day at `5bda57e`: the teacher is DeepSeek-V4.1-Flash (API model ID
`deepseek-flash`) instead of Claude Opus 5.5, which cost too much for this budget. Section 3
(D3, D7, cost), section 5 (item 5), section 6, Steps 6–9 and Appendices B–C are rewritten for
it, and Steps 6–9 record their results. A second revision the same day adds `--flawed-turns`
to `export` (D4, Steps 3, 4 and 9), after the first teacher export covered 5 of 20
`points_save_recovery` tasks.
Spec: [STAGE.md, Stage 9](STAGE.md#stage-9--verified-trajectories-and-adapter-sft--draft), second checkbox.

> **How to use this guide.** Per the coding rules in STAGE.md, the training-data logic is yours
> to write. So this guide gives you the contract, the tests and the order of work for the
> export code, and leaves its body to you. It gives complete reference code only for plumbing:
> the teacher API adapter and the CLI wiring. Steps use `- [ ]` so you can tick them off.
> As revised, the appendices hold 45 new tests (31 for export, 14 for the teacher) and the full
> suite is green at 388.

**Goal:** turn verifier-passed rollouts on the 80 train tasks into two provenance-tagged SFT
datasets (teacher track and RFT track) whose every example is exactly the prompt the model saw
and the action it then took.

**Architecture:** nothing new in the runtime. `bench --split train` already produces schema-5
traces for any backend that looks like `VLLMClient`, so collection is two more ways to run it
(a teacher API backend, and N seeds of the base model on vLLM). One new command, `export`,
reads those run directories, filters, deduplicates and writes `(prompt, completion)` pairs.

**Tech stack:** Python stdlib, `jinja2` and `httpx` (both already used), `unittest`, `llama_cpp`
to read the student's vocabulary for the teacher's token budget, vLLM on a rented GPU for RFT
sampling. The teacher adds no dependency.

## 1. What Item 2 delivers

| STAGE.md requirement | What satisfies it |
|---|---|
| (a) teacher distillation, the stronger model running inside our harness | `TeacherClient` behind `bench --split train --backend teacher` |
| (b) RFT/STaR, N base rollouts per task | `bench --split train --backend vllm --seed s` for $s = 0 \dots N-1$ |
| each track recorded with provenance | every example names its track, task, seed, run directory, run ID and request index |
| keep only verifier-passed trajectories | `export` reads `passed` from `results.json`, written by the unchanged verifier |
| then deduplicate | per task: identical trajectories collapse, a cap on distinct ones, shared prefixes counted once |
| export from schema-5 `input_messages` | prompt is rendered from `ModelRequest.input_messages` and `.tools` with the inference renderer |

**Not in Item 2:** loss masking, tokenization, the overfit smoke test and QLoRA (Item 3); any
test run (see decision D6); public function-calling data; expert-iteration rounds (Stage 10).

```
train tasks ──► bench --backend teacher ──► run dir ─┐
                (API model, our tools, our verifier) │
                                                     ├─► export ─► examples.jsonl + metadata.json
train tasks ──► bench --backend vllm × N seeds ──► N run dirs ─┘   (one directory per track)
                (base checkpoint, frozen decoding)
```

## 2. What I verified in the code

These are the facts the design rests on. Each was checked, not assumed.

1. **The trace already holds the exact input.** [agent.py:268-273](../agent.py#L268-L273)
   stores a deep copy of the prepared view and the schemas on every `ModelRequest`
   (`input_messages`, `tools`), before generation.
2. **The assistant turn for a completed request is `messages[input_message_count]`.**
   `input_message_count` is `len(raw_messages)` when the request is built, and a successful
   response is appended next ([agent.py:326](../agent.py#L326)). After a parse error the
   message at that index is the runtime feedback instead, with role `user`.
3. **Rendering the turn back through the template extends the prompt exactly.** For all 690
   requests in the scripted traces under `outputs/stage9-item1-20261012/data/runs/*-correct`,
   `render(input_messages + [assistant])` without the generation prompt starts with
   `render(input_messages)` with it, and the remainder ends in `<|im_end|>\n`. That remainder is
   the training target.
4. **`render_chat` hard-codes `add_generation_prompt=True`** ([llm.py:327-332](../llm.py#L327-L332)).
   Export needs the `False` form from the same function, so there is one renderer.
5. **Every prompt is unique, even for the same task.** File tools put the workspace root in
   their description ([tools/files.py:30](../tools/files.py#L30)), and the bench workspace is a
   fresh `tiny-agent-bench-XXXXXXXX` temporary directory. `read_file` results also carry a
   `version` built from inode and mtime. So two rollouts of one task never share a prompt
   string, and deduplicating on prompt text would remove nothing.
6. **The runtime refuses a backend without an exact token count**
   ([context.py `context_blocker`](../context.py)). A teacher backend must implement
   `measure_context` with `count_method: "exact"`. DeepSeek has no token-count endpoint, so
   the count is made locally with the student's tokenizer (D3).
7. **`LLM.parse_response` already accepts native `tool_calls`** with dict or string arguments
   ([llm.py:208-246](../llm.py#L208-L246)). DeepSeek's chat reply already has that envelope,
   so it is parsed by the same code as the student with no reshaping.
8. **Adding any non-test `.py` file drifts both manifests.** `source_sha256` in
   `evals/bench/manifest.py` hashes every `.py`/`.jinja` outside `tests/`. Train runs are not
   `--final`, so nothing blocks; test runs reject drift with no override.
9. **`.gitignore` will hide some new files.** `agent_from_scratch/docs/` is ignored (this file
   needs `git add -f`), `*result*` matches any filename containing "result", and `**/dataset/`
   matches a directory of that name. The names below avoid all three.

## 3. Decisions, with my recommendation

Read these before coding. Each one changes what the code does; the "to change" column says how.

| # | Decision | Recommendation | Why | To change it |
|---|---|---|---|---|
| D1 | Unit of an example | One example per model request: `prompt` is the rendered input, `completion` is that one assistant turn | It is literally "what the model saw", it stays exact if a view is ever elided or compacted, and every assistant turn is trained exactly once | A whole-trajectory example with several masked spans saves tokens but needs every view to be a strict prefix of the next |
| D2 | Target text | The canonical rendering of the recorded assistant message, for both tracks; record whether the sampled text matched | It is the only option for the teacher, whose reply never existed in Qwen format. Using it for RFT too means the two tracks differ in data source only. It is also how the turn appears in the model's own later prompts | Use `raw_response.choices[0].text` for RFT; then teacher and RFT targets differ in format |
| D3 | Teacher's context budget | Count the request as the student would see it: render it with the project template and tokenize it with the student's own vocabulary, read from a GGUF of the student (`--student-gguf`, vocabulary only). Hold it against the student's window and reserve | The runtime needs an exact count and DeepSeek has no count endpoint. This is the student's exact token count, so elision and compaction trigger exactly where the student's would: on the Step 5 traces the vocabulary-only count equals the model-reported `prompt_tokens` on 17 of 17 requests. The teacher's own usage is a different number (6.8% higher on the full pass) and stays in `raw_response.usage` | Count with DeepSeek's offline tokenizer instead. It only matters if the teacher's 1M window could bind, which it cannot here |
| D4 | Filter beyond `passed` | A passed trajectory can hold two kinds of flawed turn: one with the model's own failed tool call (not the injected fault, not a call skipped after one), and a final answer that is not exactly the requested value. `--flawed-turns reject` (the default) refuses the whole trajectory; `--flawed-turns drop` keeps it and leaves only those turns out. A parse error or a stuck reminder refuses the trajectory in both modes. The teacher dataset uses `drop` | Training on the step that made the mistake teaches the mistake, but the other turns of a verified trajectory are sound. Strict rejection left the teacher with 5 of 20 `points_save_recovery` tasks (Step 9). With `drop`, a dropped failure is still in the prompt of later examples as history, never as a target, and `provenance.dropped_turns` marks those examples. Injected faults stay, because recovering from them is a Stage 9 target | `--flawed-turns reject` |
| D5 | Deduplication | Per task: collapse identical action sequences, keep at most $K = 2$ distinct trajectories (fewest own-error turns, then fewest dropped turns, then shortest, then lowest seed), count a shared prefix once | Without it, tasks the base model always passes contribute $N$ copies and dominate the RFT set. Fact 5 is why the key is the action history and not the prompt text | `--max-per-task K` |
| D6 | Test in Item 2 | Do not run it | Item 2 and Item 3 both add source files, so a manifest frozen now is stale before the adapter exists. Run base and adapter on test in one session, under one freeze, at the end of Stage 9 | Run the base on test at `dbe98ad` before changing anything; it will not share a commit with the adapter run |
| D7 | Teacher model | DeepSeek-V4.1-Flash (API ID `deepseek-flash`), thinking on at `reasoning_effort: "low"`, one pass, then retry passes over the tasks that gave no usable trajectory | Claude Opus 5.5 was the first choice and was dropped for cost. The tasks are short and well specified: the teacher passed 80 of 80 at low effort. `deepseek-flash` is an alias that moves between releases, so the release name from `/models` is recorded in `settings.model_name` | `--teacher-model deepseek-v4-pro`, `--teacher-effort {none,low,high,max}` |
| D8 | Where the teacher runs | On the Mac with `TMPDIR=/tmp`, and record it | Fact 5: tool descriptions carry the workspace path. On the Mac it is `/private/var/folders/...`; on the pod it is `/tmp/...`. `TMPDIR=/tmp` narrows that to `/private/tmp/...`. It is prompt context, not a loss target, but it is a real difference between the two tracks | Run teacher collection on a Linux host for an identical path shape |

**Cost shape.** RFT coverage of a task with per-sample pass probability $p$ after $N$ samples is

$$P(\text{at least one verified trajectory}) = 1 - (1 - p)^N$$

so $p = 0.2$ gives $1 - 0.8^8 \approx 0.83$ at $N = 8$, and $p = 0$ gives nothing at any $N$.
Tasks the base model never solves are exactly where the teacher track is the only data source;
report them by skeleton. Sampling time is cheap: Stage 8 measured about $0.63$ s per request on
an A40 (718 requests in 7.6 min), and the scripted solutions need 400 requests for the 80 tasks,
so $N = 8$ is roughly $8 \times 400 \times 0.63\ \text{s} \approx 34$ min if the model takes the
scripted number of steps.

For the teacher, with $T_{\text{miss}}$ and $T_{\text{hit}}$ the input tokens billed as cache
misses and hits and $T_{\text{out}}$ the output tokens (thinking included),

$$\text{cost} \approx \frac{0.15\,T_{\text{miss}} + 0.003\,T_{\text{hit}} + 0.60\,T_{\text{out}}}{10^{6}}\ \text{USD}$$

at DeepSeek-V4.1-Flash off-peak list prices, read from the pricing page on October 12, 2026. Peak
hours (01:00–04:00 and 06:00–10:00 UTC, Monday to Friday) bill at double. Measured on the full
80-task pass: 279 requests, $T_{\text{miss}} = 127{,}099$, $T_{\text{hit}} = 376{,}704$,
$T_{\text{out}} = 23{,}400$, which is \$0.034. All teacher runs in Item 2 together come to about
\$0.07 (Step 9). These are usage times list price, not an invoice.

## 4. Global constraints

- Only the `train` split may become training data. `export` aborts on any other split, and
  `bench --backend teacher` refuses any other split before a model is constructed.
- Scripted gate fixtures (`settings.backend == "scripted"`) are never training data, even with `--pilot`.
- No change to the verifier, the skeletons, `prompts/`, `AgentLimits` defaults or `BENCH_*`.
- One renderer: export uses `llm.render_chat` and `prompts/qwen_chat.jinja`, never its own Jinja call.
- Evidence directories are never overwritten; every command writes to a new directory outside the package.
- New code lives under `evals/sft/`, which the core-size audit (`evals/core_lines.py`) does not count.
- Tests use `unittest` and run from the repository root (`GetHandsDirty/`).
- Budget: Item 2 draws on the \$10–25 trajectory-data line and part of the GPU line of the
  \$100 total. Every rental has an hour cap and ends with the pod stopped; record spend.
- The teacher sees `Task.prompt` and the workspace only. Never send `task.json`, `Expect` or a
  scripted solution to any model.

## 5. Review focus

The spec does not mention these inputs, and each would silently corrupt the dataset. Each has a
test in Appendix A or C.

1. **A run directory from val or test is passed to `export`.** Expected: abort, write
   nothing. `CollectTests.test_only_the_train_split_can_become_training_data`.
2. **A passed trajectory contains a parse error.** `messages[input_message_count]` is then
   runtime feedback, not an assistant turn. Expected: never a target.
   `ExamplesFromTraceTests.test_a_parse_error_request_is_never_a_target`.
3. **The model-facing view is not the raw prefix** (planning reminder, elision, compaction).
   Expected: the trajectory is rejected as `edited_view`.
   `RejectionTests.test_a_view_that_is_not_the_raw_prefix_is_rejected`.
4. **Arguments contain `<` or non-ASCII text.** The template escapes them (`<`, `\uXXXX`).
   Expected: the target still parses back to the recorded arguments.
   `ExamplesFromTraceTests.test_escaped_arguments_and_narration_survive_the_round_trip`.
5. **A teacher request was truncated, filtered or failed server-side, or the model alias now
   points to another release.** Expected: a response error in the loop for `length`,
   `content_filter` and `insufficient_system_resource`; the release name in
   `settings.model_name` for the alias. `ConversionTests.test_finish_reasons_reach_the_shared_parser`,
   `ConversionTests.test_settings_identify_the_teacher`.

## 6. File map

| File | Change | Responsibility |
|---|---|---|
| `llm.py` | modify `render_chat` (one keyword) | the single renderer, now usable with and without the generation prompt |
| `evals/sft/__init__.py` | create, empty | package marker |
| `evals/sft/export.py` | create (you write) | trace → examples, eligibility, selection, `export` command |
| `evals/sft/teacher.py` | create (Appendix B) | `TeacherClient` for the DeepSeek API, same interface as `VLLMClient`; `student_token_counter` |
| `evals/bench/run.py` | modify `load_model` and `main` | `--backend teacher`, `--student-gguf`, train-only guard |
| `evals/__main__.py` | modify | register `export` |
| `tests/test_sft_export.py` | create (Appendix A) | export behaviour |
| `tests/test_teacher_backend.py` | create (Appendix C) | adapter conversion, harness round trip, bench guard |
| `evals/README.md`, `docs/STAGE.md` | modify | commands and evidence |

---

## 7. Steps

Run everything from the repository root. `python` is the project environment.

### Step 0 — baseline

- [x] Create a branch: `git switch -c feat/stage9-build-sft-examples`.
- [x] Confirm the starting point: `python -m unittest discover -s agent_from_scratch/tests -q`.
  Expected: `Ran 342 tests`, `OK`.

### Step 1 — one renderer for inference and export

**Files:** modify `agent_from_scratch/llm.py`; test in `tests/test_vllm_backend.py`.

- [x] Add this test to `RenderTests` in `tests/test_vllm_backend.py`:

```python
    def test_rendering_a_turn_without_the_generation_prompt_extends_the_prompt(self):
        compiled = compile_chat_template(CHAT_TEMPLATE_PATH.read_text())
        turn = {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "b.json"})}}]}
        prompt = render_chat(compiled, CONVERSATION, default_tool_schemas)
        full = render_chat(compiled, [*CONVERSATION, turn], default_tool_schemas,
                           add_generation_prompt=False)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(full[len(prompt):],
                         '<tool_call>\n{"name": "read_file", "arguments": {"path": "b.json"}}\n</tool_call><|im_end|>\n')
        self.assertTrue(full.startswith(prompt))
```

- [x] Run it: `python -m unittest agent_from_scratch.tests.test_vllm_backend.RenderTests -v`.
  Expected: `TypeError: render_chat() got an unexpected keyword argument 'add_generation_prompt'`.
- [x] Give `render_chat` a keyword-only `add_generation_prompt: bool = True` and pass it through
  to `compiled.render`. Nothing else in the function changes.
- [x] Run the class again. Expected: both tests pass, including the existing byte-for-byte
  comparison with llama-cpp's formatter, which proves the default is unchanged.

### Step 2 — `examples_from_trace`, the core of the item

**Files:** create `evals/sft/__init__.py` (empty), `evals/sft/export.py`, `tests/test_sft_export.py`.

- [X] Create `tests/test_sft_export.py` with the whole of Appendix A.
- [X] Create `evals/sft/export.py` with the module docstring, the imports you need, and the four
  functions `examples_from_trace`, `rejection`, `select`, `collect` plus `main`, each raising
  `NotImplementedError`. This lets each step below turn one test class green.
- [X] Run: `python -m unittest agent_from_scratch.tests.test_sft_export.ExamplesFromTraceTests -v`.
  Expected: 8 errors, all `NotImplementedError`.
- [X] Implement `examples_from_trace` to this contract.

```python
def examples_from_trace(trace: dict, template) -> list[dict]:
    """One example per completed actor request, in request order.

    Args:
        trace: the JSON object in `state/runs/<run_id>.jsonl` (schema 5).
        template: `compile_chat_template(...)` of prompts/qwen_chat.jinja.

    Returns:
        list of {"request_index", "prompt", "completion", "raw_matches_canonical"}.
    """
```

  What it must do, for each request with `purpose == "agent"` and `status == "completed"`:

  1. `prompt` is `render_chat(template, request["input_messages"], request["tools"])`. Use the
     recorded view and schemas, never `trace["messages"]`.
  2. The assistant turn is `trace["messages"][request["input_message_count"]]` (fact 2). If its
     role is not `assistant`, or its tool-call IDs differ from `request["call_ids"]`, raise
     `ValueError`. A misaligned trace must stop the export, not produce a wrong target.
  3. `completion` is what the template adds when that turn joins the conversation (fact 3),
     ending at `<|im_end|>` with the trailing newline removed. At inference the server stops on
     `<|im_end|>`, so the newline after it is never generated and must not be a target.
  4. Check the round trip before returning: `LLM.parse_response` on the completion without
     `<|im_end|>` must give the recorded tool names and arguments, and the same text. Raise
     `ValueError` otherwise. This is the guarantee that what you train the model to write parses
     to the action that was actually taken.
  5. `raw_matches_canonical` is `None` unless `request["raw_response"]["choices"][0]` has a
     `text` key (the vLLM completion shape); then it is `text + "<|im_end|>" == completion`.

  Hint for 3: render twice and take the difference. You already proved in Step 1 that the second
  rendering starts with the first.

- [X] Run the class again. Expected: 8 pass.
- [X] Before moving on, predict and then check: for `points_save_recovery-0-fault`, which request
  index has the injected failure in its *prompt* for the first time, and what is that request's
  completion? Print it. If the answer surprises you, the mental model needs fixing before the
  filter does.
- [X] Commit: `git add agent_from_scratch/evals/sft agent_from_scratch/tests/test_sft_export.py && git commit -m "feat: render SFT examples from schema-5 traces"`.

### Step 3 — `rejection`, which trajectories may be used

- [X] Run: `python -m unittest agent_from_scratch.tests.test_sft_export.RejectionTests -v`.
  Expected: 5 errors, `NotImplementedError`.
- [X] Implement it to this contract.

```python
def rejection(record: dict, task: dict, trace: dict) -> str | None:
    """Why a rollout may not become training data; None when it may.

    Args:
        record: the task's entry in results.json (verifier outcome plus metrics()).
        task: the task's task.json, read for `fault_signal` only.
        trace: the run trace.
    """
```

  Return the first reason that applies, in this order:

  | Reason | Condition |
  |---|---|
  | `"failed"` | `record["passed"]` is false |
  | `"own_error"` | `parse_errors` or `stuck_reminders` is non-zero, in both modes; or, in `reject` mode, `flawed_turns` marks any turn `own_error` |
  | `"edited_view"` | any request is not a completed actor request, or its `input_messages` differ from `trace["messages"][:input_message_count]` |
  | `"loose_answer"` | in `reject` mode, `flawed_turns` marks the final turn `loose_answer` |

  **Flawed turns (added after Step 9).** `rejection` takes a keyword `flawed`, `"reject"` (default)
  or `"drop"`, and a new function says which turns are flawed:

```python
def flawed_turns(record: dict, task: dict, trace: dict) -> dict[int, str]:
    """Request index -> why that turn must not be a target, in a rollout the verifier passed."""
```

  | Reason | The turn is flawed when |
  |---|---|
  | `"own_error"` | one of its `call_ids` has a tool result with `ok` false whose `error_code` is not `"skipped"` and which is not the injected fault. The injected fault is: `task["fault_signal"]` is set, the result's `tool_name` equals `fault_signal[0]`, and `fault_signal[1]` is in its `error_message` |
  | `"loose_answer"` | it is the last request, it made no call, and `record["checks"]["format_exact"]` is false |

  In `drop` mode `rejection` ignores both, and `collect` leaves those turns out of the examples.
  The rule names no model, tool, path or skeleton, so it applies to the RFT track in the same
  way when asked for.

  A fourth reason, `"fallback"`, detected a request served by another model through Anthropic's
  server-side fallback. DeepSeek has no such mechanism, so the reason and its test were removed
  in `5bda57e`; `RejectionTests` now has 4 tests, and `FlawedTurnTests` adds 4.

  Read `expect` from `task.json` nowhere in this function. The verifier already judged the run;
  export only decides whether a correct run is also a clean demonstration.

- [X] Run the class. Expected: 5 pass.
- [X] Commit. (Committed together with Step 4 as `633abe2`.)

### Step 4 — `select`, `collect` and the `export` command

- [X] Run: `python -m unittest agent_from_scratch.tests.test_sft_export.SelectTests agent_from_scratch.tests.test_sft_export.CollectTests -v`.
  Expected: 10 errors.
- [X] Implement `select`.

```python
def select(trajectories: list[dict], max_per_task: int) -> list[dict]:
    """Distinct trajectories per task, shortest first, at most `max_per_task` each.

    Each trajectory has at least `task_id`, `seed`, `run_dir` and `examples`.
    """
```

  Sort by `(task_id, own-error turns dropped, turns dropped, number of examples, seed, run_dir)`,
  so a trajectory with a clean history comes first. Two trajectories of one task are the
  same when their sequences of `completion` strings are equal. Keep the first of each, and stop
  adding to a task once it has `max_per_task`. The same actions on two different tasks are two
  trajectories.

- [X] Implement `collect`.

```python
def collect(run_dirs: list[Path], track: str, *, max_per_task: int = 2, pilot: bool = False,
            flawed: str = "reject") -> tuple[list[dict], dict]:
    """Read bench run directories and return (examples, metadata) for one track."""
```

  Abort with `ValueError` (nothing is written) when:

  - a run's `metadata.json` has `split != "train"`, or any record does;
  - `settings.backend` is `"scripted"`, always;
  - `settings.backend` is not the track's (`teacher` → `"deepseek"`, `rft` → `"vllm"`), unless `pilot`;
  - `settings.chat_template_sha256` is set and differs from the hash of the template file on disk;
  - the same `(task id, seed)` appears twice across the given run directories;
  - a trace's `schema_version` is not 5.

  For each record: load `<run>/<task id>/task.json` and
  `<run>/<task id>/state/runs/<run_id>.jsonl`, count the `rejection` reason or build a trajectory
  `{task_id, skeleton, family, condition, seed, run_dir, run_id, examples}`. Then `select`, then
  flatten with the prefix rule from D5: an example is a duplicate when the same task already
  contributed the same action history ending in the same completion. Because of fact 5 the key
  cannot be the prompt text. In `drop` mode a flawed turn is not written, but it stays in that
  action history, because what follows it was decided after it.

  Each output example:

```json
{"id": "<first 16 hex of sha256(prompt + NUL + completion)>",
 "prompt": "<|im_start|>system\n...<|im_start|>assistant\n",
 "completion": "<tool_call>\n{...}\n</tool_call><|im_end|>",
 "provenance": {"track": "rft", "pilot": false, "task_id": "...", "skeleton": "...",
                "family": "...", "condition": "fault", "seed": 3, "run_dir": "rft-train-s3",
                "run_id": "...", "dropped_turns": [], "request_index": 2,
                "raw_matches_canonical": true}}
```

  The trace stays the source of truth: `run_dir`, `task_id`, `run_id` and `request_index` are
  enough to find the original messages, so the example does not copy them. `dropped_turns`
  lists the request indexes of this rollout that were left out; when it is not empty and holds
  an own-error turn, later examples have that failed call in their prompt.

  `metadata`: `track`, `pilot`, `max_per_task`, `flawed_turns`, `chat_template_sha256`,
  `target: "canonical"`,
  `sources` (per run: `run_dir`, `seed`, `settings`, `git_revision`), and `funnel`:

  | `funnel` key | Meaning |
  |---|---|
  | `rollouts` | records read |
  | `rejected` | count per reason |
  | `eligible` | rollouts with no rejection |
  | `kept_trajectories`, `tasks_covered` | after `select` |
  | `dropped_turns` | flawed turns left out of kept trajectories, per reason |
  | `examples_before_dedup`, `examples` | before and after the prefix rule |
  | `by_skeleton` | `{name: {"trajectories", "tasks"}}` over kept trajectories |

- [X] Implement `main`: positional run directories, `--track {teacher,rft}` (required),
  `--output` (required, must not exist), `--max-per-task` (default 2), `--pilot`,
  `--flawed-turns {reject,drop}` (default `reject`). Turn
  `ValueError`/`OSError`/`KeyError` from `collect` into `parser.error`. Refuse to write when
  there are no examples. Write `examples.jsonl` with `utils.write_jsonl` and `metadata.json`
  with `evals.run.save`, then print the funnel.
- [X] Register the command in `evals/__main__.py`: add `"export": "sft.export"` to `COMMANDS`
  and one line to the module docstring.
- [X] Run the two classes. Expected: 10 pass.
- [X] Run the whole suite. Expected: `Ran 366 tests`, `OK` (342 + 1 + 23).
- [X] Commit: `git commit -am "feat: export verified train rollouts as SFT examples"`. (`633abe2`)

### Step 5 — free dry run on real traces (optional, recommended)

A local GGUF run exercises the exporter on real model text before any money is spent. It is a
pipeline check, not data: not bf16, not vLLM.

- [X] Run four tasks with the Qwen3-4B Q8_0 GGUF used in `docs/benchmark.md`'s local check:

```sh
python -m agent_from_scratch.evals bench --backend llama_cpp --split train --seed 0 \
  --model-path <Qwen3-4B-Instruct-2507 Q8_0 GGUF> \
  --tasks inventory_reconciliation-0-fault,eligibility_record_update-3,address_conflict_stop-1,points_save_recovery-0-fault \
  --output outputs/stage9-item2-pilot-gguf
python -m agent_from_scratch.evals export --track rft --pilot \
  --output outputs/stage9-item2-pilot-gguf-sft outputs/stage9-item2-pilot-gguf
```

- [X] If no task passed, `export` refuses to write. That is a result: note which checks failed in
  each `result.json`, since those are the behaviours the teacher track has to supply.

**Result (October 12, 2026; bench run at `cab1e03`, exported with the Step 3–4 code before it
was committed as `633abe2`).** Model file:
`unsloth/Qwen3-4B-Instruct-2507-GGUF @ a06e946`, `Qwen3-4B-Instruct-2507-Q8_0.gguf`. 2 of 4 tasks
passed; `export` wrote 4 examples, all `pilot: true`. One sample of four tasks: a pipeline
check, not a measurement of the base model.

| Task | Result | Failed checks | What the model did |
|---|---|---|---|
| `eligibility_record_update-3` | pass | none | read all three files in one turn, replied `BLOCKED` |
| `address_conflict_stop-1` | pass | none | read both files in one turn, replied `NEED_INPUT` |
| `inventory_reconciliation-0-fault` | fail | `format_exact` | recovered from the read fault and reached `NONE`, but wrapped it in three sentences |
| `points_save_recovery-0-fault` | fail | `files`, `answer_content`, `format_exact` | retried the failed save, but copied the `N| ` line prefixes from `read_file` output into the file content, then replied with an error message |

Funnel: 4 rollouts, 2 rejected as `failed`, 2 eligible, 2 kept, 4 examples before and after the
prefix rule. What it shows for the later steps:

- The exporter works on real model text: every prompt ends with `<|im_start|>assistant\n` and
  every target with one `<|im_end|>`.
- The base model batches calls, so a target can hold two or three `<tool_call>` blocks. When the
  injected fault hit the first call of a batch, the second was `skipped`; `rejection` does not
  count a skipped call as the model's own error.
- The two failures are the behaviours Step 8 checks in teacher targets: a bare-token final
  answer, and `write_file` content without the `N| ` prefix.
- `raw_matches_canonical` is `None` on a llama.cpp run. Only vLLM runs can fill it (Step 11).
- Without `--pilot` the same run is refused with the backend error.

### Step 6 — teacher backend

**Files:** create `evals/sft/teacher.py` (Appendix B) and `tests/test_teacher_backend.py`
(Appendix C); modify `evals/sft/export.py` and `tests/test_sft_export.py`.

The adapter is API plumbing, so Appendix B is complete. Read it against these points, because
they are the parts that are easy to get wrong:

- **The model ID is `deepseek-flash`.** There is no `deepseek-v4.1-flash` ID: `GET /models`
  lists `deepseek-flash` with the name "DeepSeek-V4.1-Flash". The ID is an alias that moves
  between releases, so the adapter reads the name at start-up and records it in
  `settings.model_name`.
- **The harness's messages are already in the API's shape**, so they are sent with only the
  fields the API defines. A failed tool result needs no extra marker.
- **Each turn's `reasoning_content` goes back on later requests.** The API documentation
  requires this whenever tools are in play and says a missing one returns a 400. A live probe
  without it returned 200 on October 12, 2026, so the rule was not enforced that day; the
  adapter follows the documentation. The harness's message list does not hold reasoning, so
  the adapter keeps it keyed by the turn's first tool-call ID. The harness keeps a non-empty
  unique call ID as is, so the key survives.
- **No sampling parameters, no seed, no `tool_choice`.** Thinking mode ignores `temperature`.
  Depth is set with `reasoning_effort` (`low`, `high`, `max`; the API default is `high`), and
  `--teacher-effort none` sends `thinking: {"type": "disabled"}`.
- **`max_tokens` is the teacher's own (8,192).** Thinking counts toward it, so the student's
  2,048 reserve would truncate replies. The budget's `response_reserve` still reports the
  student's.
- **The budget is counted locally, in student tokens** (D3). `student_token_counter` loads only
  the vocabulary of a student GGUF (0.2 s, no weights). A trace then holds two different counts
  per request: `budget.prompt_tokens` is the student's, `usage.prompt_tokens` is the teacher's.
- **Finish reasons need no mapping.** `stop` and `tool_calls` parse; `length` raises
  `TRUNCATED_RESPONSE`; `content_filter` and `insufficient_system_resource` raise
  `UNSUPPORTED_FINISH_REASON`.
- **This API has no server-side fallback**, so the `fallback` rejection reason was removed from
  `export` (Step 3).
- **Prompt caching is automatic.** `usage.prompt_cache_hit_tokens` in the trace shows it working.
- **The key is read from `DEEPSEEK_API_KEY`** and is never written to `settings()` or a trace.

- [X] No new dependency: the adapter uses `httpx`, which `VLLMClient` already needs. There is
  no `requirements-teacher.txt`.
- [X] Create the test file from Appendix C and run:
  `python -m unittest agent_from_scratch.tests.test_teacher_backend -v`.
  Expected: import error, `No module named 'agent_from_scratch.evals.sft.teacher'`.
- [X] Create `evals/sft/teacher.py` from Appendix B. In `export.py`, set
  `TRACK_BACKENDS["teacher"]` to `"deepseek"` and remove the `fallback` reason; in
  `tests/test_sft_export.py`, drop its test and use `backend="deepseek"` in the backend test.
- [X] Run again. Expected: 13 of 14 pass, and `BenchTests.test_bench_builds_the_teacher_backend`
  errors until Step 7. The other two bench tests pass early for the wrong reason (argparse
  rejects `--backend teacher` before the guard exists), so Step 7 checks the guards by message.
- [X] Commit. (Committed together with Step 7 as `5bda57e`, so no commit holds a failing test.)

### Step 7 — `bench --backend teacher`

**Files:** modify `evals/bench/run.py`.

- [X] In `load_model`, before the vLLM branch:

```python
    if args.backend == "teacher":
        from ..sft.teacher import TeacherClient, student_token_counter
        count, tokenizer = student_token_counter(args.student_gguf)
        return TeacherClient(args.teacher_model, count=count, budget_tokenizer=tokenizer,
                             effort=args.teacher_effort, sample=args.seed, n_ctx=BENCH_N_CTX,
                             reserve=max_tokens)
```

- [X] In `main`, extend the backend choices and add three options:

```python
    parser.add_argument("--backend", choices=("llama_cpp", "vllm", "teacher"), default="llama_cpp",
                        help="llama_cpp: the local GGUF, for pipeline checks; vllm: the frozen control; "
                             "teacher: an API model, --split train only")
    parser.add_argument("--teacher-model", default="deepseek-flash", help="API model ID (--backend teacher)")
    parser.add_argument("--teacher-effort", choices=("none", "low", "high", "max"), default="low",
                        help="reasoning_effort; none turns thinking off (--backend teacher)")
    parser.add_argument("--student-gguf", type=Path,
                        help="A GGUF of the student; its tokenizer counts the context budget (--backend teacher)")
```

- [X] Directly after `overrides = parse_limits(...)`, before the test-split checks:

```python
    if args.backend == "teacher" and args.split != "train":
        parser.error("The teacher may only see the train split.")
    if args.backend == "teacher" and (args.student_gguf is None or not args.student_gguf.is_file()):
        parser.error("--backend teacher needs --student-gguf, an existing GGUF of the student model.")
```

- [X] Update the usage line in the `bench/run.py` docstring.
- [X] Run the whole suite. Expected: `Ran 379 tests`, `OK` (366 − 1 + 14).
- [X] Check both guards by their messages: `bench --backend teacher --split val --output X`
  must end with "The teacher may only see the train split.", and
  `bench --backend teacher --split train --output X` with "--backend teacher needs --student-gguf".
- [X] Commit: `feat: run a DeepSeek teacher on the train split` (`5bda57e`).

### Step 8 — teacher smoke, then look at the data

This is the "next smallest gap" in STAGE.md: a few verified trajectories, inspected at request
level, before scaling.

- [X] Authenticate without putting a key in the repository: export `DEEPSEEK_API_KEY` in this
  shell only.
- [X] Run one task per skeleton, covering a read fault, a blocked update, a conflict stop and a
  write fault:

```sh
RUN=outputs/stage9-item2-$(date +%Y%m%d)
GGUF=gz-data/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q8_0.gguf
TMPDIR=/tmp python -m agent_from_scratch.evals bench --backend teacher --split train --seed 0 \
  --student-gguf $GGUF \
  --tasks inventory_reconciliation-0-fault,eligibility_record_update-3,address_conflict_stop-1,points_save_recovery-0-fault \
  --output $RUN-teacher-smoke
python -m agent_from_scratch.evals export --track teacher \
  --output $RUN-teacher-smoke-sft $RUN-teacher-smoke
```

- [X] Read every example, prompt tail and target:

```sh
python - "$RUN-teacher-smoke-sft/examples.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    example = json.loads(line)
    p = example["provenance"]
    print("=" * 100, p["task_id"], "request", p["request_index"], sep="  ")
    print(example["prompt"][-700:])
    print(">>> TARGET:", example["completion"])
PY
```

- [X] Check, and write down what you find:
  - Each prompt ends with `<|im_start|>assistant\n` and each target with one `<|im_end|>`.
  - In the fault tasks, the failed observation is in the prompt of the request that retries.
  - Final answers are the bare token (`NEED_INPUT`, `BLOCKED`, the ID list). This is the base
    model's weakest behaviour, so these targets matter most.
  - Whether the teacher batches calls in one turn, and what a multi-call target looks like.
  - Whether any target carries narration, and whether you want the student to learn it.
  - No target copies an `N| ` line prefix into `write_file` content.
- [X] Read the cost: each trace's `raw_response.usage` has the cache fields. Project the
  80-task pass before running it.
- [X] If fewer than 4 of 4 pass, read the failing `result.json` `checks` first. Raise
  `--teacher-effort` only if the failure is the teacher's reasoning, not the task wording.

**Result (October 12, 2026, at `5bda57e`; `$RUN` is `outputs/stage9-item2-20261012`).** 4 of 4
passed in 18 s. `export` kept 3 trajectories and 8 examples; the fourth,
`points_save_recovery-0-fault`, was rejected as `own_error` (see Step 9).

- Format: all 8 prompts end with the assistant header, all 8 targets end with one `<|im_end|>`,
  none holds an `N| ` prefix or an absolute workspace path.
- Final answers are bare: `NONE`, `BLOCKED`, `NEED_INPUT`.
- The teacher batches: targets hold one, two or three `<tool_call>` blocks. In
  `inventory_reconciliation-0-fault` the fault hit the first call of a two-call batch, the second
  was `skipped`, and the next turn retried the first alone.
- Narration: 3 of the 5 tool-call targets start with a sentence, for example
  "I'll read both files." and "The first read failed with a temporary error. Let me retry both."
- Cost: 14 requests, 24,768 teacher prompt tokens (17,920 cache hits, 6,848 misses) and 1,182
  output tokens, which is \$0.002. The student-token budget for the same 14 requests was 23,234.
  Projected to 80 tasks: under \$0.05.

### Step 9 — full teacher pass

- [X] Run all 80 train tasks, then retry the tasks that gave no usable trajectory as later
  samples. "No usable trajectory" is a verifier failure or a `rejection` reason, since either
  one leaves the task without data:

```sh
TMPDIR=/tmp python -m agent_from_scratch.evals bench --backend teacher --split train --seed 0 \
  --student-gguf $GGUF --output $RUN-teacher-s0
TMPDIR=/tmp python -m agent_from_scratch.evals bench --backend teacher --split train --seed 1 \
  --teacher-effort high --student-gguf $GGUF --tasks "$UNUSABLE" --output $RUN-teacher-s1
TMPDIR=/tmp python -m agent_from_scratch.evals bench --backend teacher --split train --seed 2 \
  --teacher-effort none --student-gguf $GGUF --tasks "$UNUSABLE" --output $RUN-teacher-s2
```

  For the teacher, `--seed` is only a sample label: the API takes no seed.
- [X] Export the track:
  `python -m agent_from_scratch.evals export --track teacher --flawed-turns drop --max-per-task 1 --output $RUN-sft-teacher-v3 $RUN-teacher-s0 $RUN-teacher-s1 $RUN-teacher-s2`.
- [X] Record the funnel, the pass count by skeleton, and the dollars spent.

**Result (October 12, 2026, at `5bda57e`).**

| Run | Effort | Tasks | Verifier passed | Eligible after `rejection` | Requests | Time | Cost |
|---|---|---|---|---|---|---|---|
| `teacher-smoke` | low | 4 | 4 | 3 | 14 | 18 s | \$0.002 |
| `teacher-s0` | low | 80 | 80 | 62 | 279 | 395 s | \$0.034 |
| `teacher-s1` | high | 18 | 18 | 0 | 111 | 191 s | \$0.017 |
| `teacher-s2` | none | 18 | 18 | 3 | 103 | 156 s | \$0.014 |

Cost is usage times the off-peak list price, about \$0.07 in total (at most \$0.14 if every
request had been billed at the peak rate). It is not an invoice.

**First export, strict filter.** Funnel of `$RUN-sft-teacher` (s0, s1 and s2;
`--flawed-turns reject`, which was the only behaviour at the time): 116 rollouts, 0 `failed`,
51 `own_error`, 0 `edited_view`; 65 eligible, 65 kept trajectories, 65 of 80 tasks covered;
187 examples before and after the prefix rule (171 from sample 0, 16 from sample 2).

| Skeleton | Tasks | Verifier passed (s0) | Tasks covered | Trajectories |
|---|---|---|---|---|
| `inventory_reconciliation` | 20 | 20 | 20 | 20 |
| `eligibility_record_update` | 20 | 20 | 20 | 20 |
| `address_conflict_stop` | 20 | 20 | 20 | 20 |
| `points_save_recovery` | 20 | 20 | **5** | 5 |

What the data looks like:

- 122 tool-call targets and 65 final answers. Calls per tool-call target: 44 single, 57 double,
  21 triple. No target copies an `N| ` prefix, and none holds an absolute workspace path.
- 40 of the 122 tool-call targets (33%) carry narration before the first call. The most common
  is "I'll read both files." (18 times).
- 62 of the 65 final answers are the bare token. The other 3 are `points_save_recovery` answers
  of the form `SAVED` plus a bulleted explanation; the verifier accepts them because that
  skeleton's answer check is not format-exact.

**Why `points_save_recovery` is covered on 5 tasks only.** All 51 rejections are this skeleton,
and all for one habit. The prompt says "Read points.json and both account records" without
naming the account files, and the task offers `read_file` and `write_file` but no listing tool.
The teacher explores by calling `read_file` on `"."`, which fails with "Not an existing regular
file". In sample 0 the first turn was `points.json` with `"."` on 11 tasks, `"."` alone on 6,
`"."` then `points.json` on 1, and `points.json` alone on the 2 that stayed clean. The verifier
passes all of them; `rejection` drops them under D4, because the failed read is the teacher's
own mistake. Resampling does not fix it: high effort gave 0 clean of 18 and thinking off gave 3
of 18.

**Decisions taken (October 12, 2026).**

1. **D4: drop the flawed turn, not the trajectory.** `export` gained `--flawed-turns` (Step 3).
2. **Narration stays.** Stripping it would also change later prompts, which hold each turn's
   narration as history.
3. **A loose final answer is a flawed turn**, read from the verifier's own `format_exact` check.
4. **One trajectory per task for the teacher track** (`--max-per-task 1`). The retries exist
   for one skeleton only, so with $K = 2$ that skeleton would get two trajectories per task and
   the others one: 178 of 337 examples would be `points_save_recovery`.

**Final teacher dataset** (`$RUN-sft-teacher-v3`; `pilot: false`, `flawed_turns: "drop"`,
`max_per_task: 1`): 116 rollouts, none rejected; 80 kept trajectories, 80 of 80 tasks; 15
`own_error` turns and 4 `loose_answer` turns dropped; 255 examples before and after the prefix
rule (201 from sample 0, 20 from sample 1, 34 from sample 2).

| Skeleton | Tasks covered | Examples |
|---|---|---|
| `inventory_reconciliation` | 20 of 20 | 59 |
| `eligibility_record_update` | 20 of 20 | 60 |
| `address_conflict_stop` | 20 of 20 | 40 |
| `points_save_recovery` | 20 of 20 | 96 |

- No target reads `"."`, copies an `N| ` prefix or holds an absolute workspace path. All 76 final
  answers are the bare token.
- 71 examples (28%), all `points_save_recovery`, have the teacher's failed `"."` read in their
  prompt as history. The student is never trained to write it. (An earlier version of this note
  said the student's own history would usually not contain a failed read. Step 10 showed the
  opposite: the base model guesses a failing path in the first turn of 158 of 160 rollouts of
  this skeleton, so these examples show the step it most needs, which is what to do next.)
- Known gaps in `points_save_recovery`: the first step (read `points.json` alone) is
  demonstrated by 5 examples only, because a dropped first turn cannot be recovered; and 4 of
  the 20 tasks have no final-answer example.
- 74 of the 179 tool-call targets (41%) carry narration.
- `$RUN-sft-teacher` (strict) is superseded. `$RUN-sft-teacher-v2` is an intermediate export
  made before `select` preferred clean histories, and can be deleted.

### Step 10 — RFT sampling on the rented GPU

Follow the runbook in [benchmark.md](benchmark.md#runbook-control-baseline-on-a-rented-gpu-as-run-september-27-2026)
exactly as run for Stage 8: same image, same `vllm==0.30.0`, same serve command including
`VLLM_USE_FLASHINFER_SAMPLER=0`. Only the commands after the server is up differ.

- [X] Before renting: commit everything, set a 2-hour cap, and start the cost guard from the Mac
  (`sleep <secs> && runpodctl pod stop <pod-id>`).
- [X] Ship a shallow clone of this branch's commit, as the runbook does.
- [X] Smoke four train tasks and check that measured prompt tokens equal the served ones:

```sh
python -m agent_from_scratch.evals bench --backend vllm --split train --seed 0 \
  --tasks inventory_reconciliation-0-fault,eligibility_record_update-3,address_conflict_stop-1,points_save_recovery-0-fault \
  --output outputs/rft-smoke
python - <<'PY'
import glob, json
rows = [q for f in glob.glob("outputs/rft-smoke/*/state/runs/*.jsonl") for line in open(f)
        for q in json.loads(line)["model_requests"] if q["status"] == "completed" and q["purpose"] == "agent"]
bad = [q for q in rows if q["budget"]["prompt_tokens"] != q["usage"]["prompt_tokens"]]
print(f"{len(bad)} of {len(rows)} requests where measured != served prompt tokens")
PY
```

  Expected: 0 mismatches. This is the token-identity evidence for the RFT track: the exporter
  re-renders with the same function and the same template file, and `collect` refuses a run
  whose recorded template hash differs.
- [X] Sample $N = 8$ and aggregate:

```sh
for s in 0 1 2 3 4 5 6 7; do
  python -m agent_from_scratch.evals bench --backend vllm --split train --seed $s \
    --output outputs/rft-train-s$s
done
python -m agent_from_scratch.evals aggregate outputs/rft-train-s? --output outputs/rft-train-aggregate.json
```

  `aggregate` gives pass@1, pass^k, pass@k and the mixed-outcome count per skeleton. The mixed
  count is also the first evidence for Stage 10's GRPO precondition.
- [X] Copy `outputs/` back, remove the pod, confirm `runpodctl pod list` is empty, record the cost.

**Result (October 12, 2026, at `11e4205`).** One A40 in CA-MTL-1 at \$0.59 per hour, pod
`stage9-rft`, created 12:49 UTC and deleted after the copy-back was verified;
`runpodctl pod list --all` was empty afterwards. Cost: **\$0.34**, read from the account balance
before and after, which is about 34 minutes of rental. The run
directories are in `outputs/stage9-item2-20261012-rft/` (`rft-smoke`, `rft-train-s0` … `s7`,
`rft-train-aggregate.json`).

- One trap beyond the runbook: the unpacked clone is owned by the Mac's user ID, and git on the
  pod refuses it ("dubious ownership"), which would fail `open_run`. `chown -R root:root` on the
  clone fixes it; `git status` stayed clean.
- Smoke: 0 of 12 requests where measured and served prompt tokens differ, 0 parse errors, 1 of 4
  passed. Over the eight full runs: **0 of 1,737** requests differ, 0 parse errors.
- Sampling: 640 rollouts in 26 minutes, about 0.9 s per request.

| Skeleton | Mean pass@1 | pass@8 | pass^8 | Mixed tasks |
|---|---|---|---|---|
| `inventory_reconciliation` | 0.044 | 0.15 | 0.00 | 3 |
| `eligibility_record_update` | 0.156 | 0.25 | 0.00 | 5 |
| `address_conflict_stop` | 0.988 | 1.00 | 0.90 | 2 |
| `points_save_recovery` | 0.006 | 0.05 | 0.00 | 1 |
| all 80 tasks | 0.298 | 0.36 | 0.23 | 11 |

Where the base model fails, from the `checks` of the 449 failed rollouts:

- `inventory_reconciliation`: all 153 failures miss `format_exact`, and only 33 of them also
  have the wrong content. The model finds the answer and wraps it in explanation.
- `eligibility_record_update`: 80 of 135 failures miss `evidence` and 57 miss `format_exact`.
  Split by expected answer: on the 80 `UPDATED` rollouts the model edited the file in 79 and the
  file was right in 78, but it read the record back in 0 of 79, so none passed. On the 80
  `BLOCKED` rollouts it never wrote and always named the right outcome, and 25 passed; the other
  55 wrapped `BLOCKED` in explanation.
- `points_save_recovery`: all 159 failures leave the files wrong, because the model almost never
  reaches a write. In 158 of 160 rollouts the first turn reads `points.json` together with a
  guessed account path (`accounts/1.json`, `accounts/account1.txt`), which fails. `points.json`
  was read successfully in all 160 and names the account files, but only 5 rollouts ever read
  `account_a.txt`; 155 end with a text answer saying the files cannot be located. The injected
  save fault is therefore almost never met.
- So the base model's failures on this pool are three specific behaviours: a final answer that
  is not the bare token, no read-back after a write, and giving up after its own failed read
  instead of using what it already read.
- 11 of 80 tasks have mixed outcomes across the eight samples, which is the first evidence for
  Stage 10's GRPO precondition. 96 of 320 completion claims were false completions.

### Step 11 — export RFT and compare the tracks

- [X] `python -m agent_from_scratch.evals export --track rft --flawed-turns drop --output $RUN-sft-rft $RUN-rft/rft-train-s?` (the same flawed-turn mode as the teacher track, so the two differ in data source only; $K = 2$, the default).
- [X] From the two `metadata.json` files, build one table per track by skeleton: tasks covered
  out of 20, kept trajectories, examples, and for RFT the base pass@1 from the aggregate.
- [X] Report the share of RFT examples with `raw_matches_canonical == false`, and read a few.
  A low share supports D2. A high share means the base model's own formatting differs from the
  template's, and you should decide D2 again with that evidence.
- [X] List the tasks covered by teacher only, by RFT only, by both, and by neither. "Neither" is
  a task-design question, not a data question.
- [X] Count completions that contain `tiny-agent-bench-`. Those are absolute-path arguments,
  where identical trajectories do not collapse (D5 errs toward keeping them).

**Result (October 12, 2026).** Both datasets have `pilot: false`, `flawed_turns: "drop"`, and
every example's provenance resolves to a completed request in an existing trace (255 of 255 and
63 of 63).

| Skeleton | Teacher: tasks | Teacher: examples | Base pass@1 | RFT: tasks | RFT: examples |
|---|---|---|---|---|---|
| `inventory_reconciliation` | 20 of 20 | 59 | 0.044 | 3 of 20 | 6 |
| `eligibility_record_update` | 20 of 20 | 60 | 0.156 | 5 of 20 | 10 |
| `address_conflict_stop` | 20 of 20 | 40 | 0.988 | 20 of 20 | 40 |
| `points_save_recovery` | 20 of 20 | 96 | 0.006 | 1 of 20 | 7 |
| total | 80 of 80 | 255 | 0.298 | 29 of 80 | 63 |

- RFT funnel: 640 rollouts, 449 `failed`, 0 `own_error`, 0 `edited_view`; 191 eligible, 29 kept
  trajectories on 29 tasks, 1 `loose_answer` turn dropped, 63 examples before and after the
  prefix rule. $K = 1$ gives the same 29 trajectories: on every covered task, all passing
  rollouts took the same actions. The strict mode gives 28 tasks and 56 examples.
- **Coverage: both tracks 29 tasks, teacher only 51, RFT only 0, neither 0.** The 51
  teacher-only tasks are 19 `points_save_recovery`, 17 `inventory_reconciliation` and 15
  `eligibility_record_update`. No task is uncovered, so there is no task-design question here.
- **`raw_matches_canonical`: 62 true, 1 false (1.6%), which supports D2.** The one difference is
  whitespace: the model wrote a blank line between its narration and the `<tool_call>` block,
  and the template writes a single newline.
- No completion in either dataset contains `tiny-agent-bench-`, so no absolute-path argument
  kept identical trajectories apart.
- RFT targets carry narration on 2 of 35 tool-call turns, against 74 of 179 for the teacher. 40
  of the 63 RFT examples are `address_conflict_stop`, the skeleton the base model already
  passes; an adapter trained on the RFT set alone can learn little about the other three.

### Step 12 — record the evidence

- [X] `evals/README.md`: add `export` and `--backend teacher` to the command table and file list.
- [X] `docs/STAGE.md`: tick Item 2 and add an evidence section in the form of the Item 1 section:
  commands as run, both funnels, the coverage table, the raw-versus-canonical rate, dollars
  spent against the budget lines, and the known limits (D3 budget in student tokens while usage is the teacher's, D8 path difference,
  D4 strictness, small sample labelled as a pilot).
- [X] State what the manifest now is. Measured with `manifest_drift` at `11e4205`: the Item 1
  dataset manifest has drifted in `source_sha256` only (`llm.py`, `evals/bench/run.py` and
  `evals/__main__.py` changed; the three `evals/sft/` files added). The Stage 8 manifest also
  shows `splits_sha256` and `tasks`, but those date from Item 1; Item 2 added source drift
  only. Do not refresh either. The versioned freeze happens once, before the test
  comparison at the end of Stage 9.
- [X] `git add -f agent_from_scratch/docs/STAGE9_ITEM2_PLAN.md` and commit the rest.
- [ ] Merge the branch. Left for the owner: it changes `main`.

## 8. Exit gate

Item 2 is done when all of these hold.

- [X] 388 tests pass.
- [X] `$RUN-sft-teacher-v3/` and `$RUN-sft-rft/` each hold `examples.jsonl` and `metadata.json`,
  with `pilot: false` and every source run on the train split.
- [X] Every example's provenance resolves to an existing trace and request.
- [X] RFT smoke showed 0 requests where measured and served prompt tokens differ.
- [ ] You have read at least one full trajectory per skeleton per track as rendered text.
  (Yours to do. The Step 8 snippet prints them from `$RUN-sft-teacher-v3/examples.jsonl` and
  `$RUN-sft-rft/examples.jsonl`.)
- [X] Coverage by skeleton, yield, rejections by reason and cost are written in STAGE.md,
  including tasks neither track covers.
- [X] No val or test task was run through the teacher or exported. (All 14 teacher
  and RFT run directories record `split: train` in their metadata and in every record.)

## 9. Handoff to Item 3

- An example is already split for the loss: mask everything in `prompt`, train on `completion`.
  Tokenize `prompt + completion` as one string and derive the mask from character offsets;
  tokenizing the two halves separately can differ at the boundary.
- Token counts per example are not in Item 2's output. Compute them when the tokenizer is
  loaded for training, and check the total against the 2–8M-token line in the budget.
- The val comparison needs base, teacher-SFT and RFT-SFT on one server command. The test
  comparison additionally needs a new dataset manifest generated at the commit that contains
  the training code (`evals dataset --output ... --check`), because test rejects drift.

---

## Appendix A — `tests/test_sft_export.py`

```python
"""Verified train rollouts become (prompt, completion) pairs that are exactly what the model saw."""

import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_from_scratch.config import CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import answer, call
from agent_from_scratch.evals.run import save
from agent_from_scratch.evals.sft import export
from agent_from_scratch.evals.sft.export import (collect, examples_from_trace, flawed_turns, rejection,
                                                 select)
from agent_from_scratch.llm import (LLM, LLMResponse, ResponseError, ResponseErrorCode, ResponseType,
                                    compile_chat_template, render_chat)
from agent_from_scratch.tests.test_bench_skeletons import _model
from agent_from_scratch.tools.base import ToolCall

TEMPLATE = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
END = "<|im_end|>"
ENTRIES = {ctx.id: (skeleton, ctx) for skeleton, ctx in specs("train")}


def rollout(root: Path, task_id: str, *, before=(), script=None, overrides=None, name="case"):
    """Run one train task with its scripted solution; returns (record, task, trace).

    `before` prepends responses; `script` rewrites the solution's response list.
    """
    skeleton, ctx = ENTRIES[task_id]
    with TemporaryDirectory() as temporary:
        task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"),
                              Path(temporary), ctx)
    out = root / name
    responses = [*before, *(script or list)(skeleton.solution(task))]
    record = run_case(_model(responses), skeleton, ctx, out, overrides=overrides)
    trace = json.loads((out / "state" / "runs" / f"{record['run_id']}.jsonl").read_text())
    return record, json.loads((out / "task.json").read_text()), trace


def handmade(arguments: dict, *, content: str = "", raw_text: str | None = None) -> dict:
    """The smallest schema-5 trace with one completed tool-call request."""
    messages = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Go."},
                {"role": "assistant", "content": content, "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "write_file", "arguments": json.dumps(arguments)}}]}]
    return {"schema_version": 5, "messages": messages, "model_requests": [{
        "purpose": "agent", "status": "completed", "input_message_count": 2, "call_ids": ["c1"],
        "input_messages": messages[:2], "tools": {},
        "raw_response": None if raw_text is None else {"choices": [{"text": raw_text}]}}]}


class ExamplesFromTraceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_one_example_per_completed_actor_request(self):
        record, _, trace = rollout(self.root, "inventory_reconciliation-0-fault")
        examples = examples_from_trace(trace, TEMPLATE)
        self.assertEqual(len(examples), record["actor_requests"])
        self.assertEqual([e["request_index"] for e in examples], list(range(len(examples))))
        self.assertEqual(examples[0]["completion"],
                         '<tool_call>\n{"name": "read_file", "arguments": {"path": "inventory.json"}}\n</tool_call>' + END)
        self.assertEqual(examples[-1]["completion"], record["final_answer"] + END)
        for example in examples:
            self.assertTrue(example["prompt"].endswith("<|im_start|>assistant\n"))
            self.assertEqual(example["completion"].count(END), 1)

    def test_prompt_is_byte_identical_to_the_inference_prompt(self):
        _, _, trace = rollout(self.root, "points_save_recovery-0-fault")
        for example in examples_from_trace(trace, TEMPLATE):
            request = trace["model_requests"][example["request_index"]]
            self.assertEqual(example["prompt"],
                             render_chat(TEMPLATE, request["input_messages"], request["tools"]))

    def test_each_target_is_how_the_next_prompt_shows_that_turn(self):
        _, _, trace = rollout(self.root, "points_save_recovery-0-fault")
        examples = examples_from_trace(trace, TEMPLATE)
        for earlier, later in zip(examples, examples[1:]):
            self.assertTrue(later["prompt"].startswith(earlier["prompt"] + earlier["completion"] + "\n"))

    def test_target_parses_back_to_the_action_that_was_taken(self):
        _, _, trace = rollout(self.root, "points_save_recovery-0-fault")
        for example in examples_from_trace(trace, TEMPLATE):
            request = trace["model_requests"][example["request_index"]]
            recorded = trace["messages"][request["input_message_count"]]
            parsed = LLM.parse_response({"choices": [{"message": {
                "role": "assistant", "content": example["completion"].removesuffix(END)}}]})
            self.assertEqual([(c.name, c.arguments) for c in parsed.tool_calls],
                             [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                              for c in recorded.get("tool_calls", [])])

    def test_a_parse_error_request_is_never_a_target(self):
        record, _, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                   before=[ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)])
        self.assertTrue(record["passed"], record["checks"])
        examples = examples_from_trace(trace, TEMPLATE)
        self.assertEqual(len(examples), record["actor_requests"] - 1)
        self.assertNotIn(0, [e["request_index"] for e in examples])
        self.assertIn("[Runtime feedback]", examples[0]["prompt"])

    def test_escaped_arguments_and_narration_survive_the_round_trip(self):
        arguments = {"path": "a<b.txt", "content": "值 <tool_call> \"q\"\n"}
        example = examples_from_trace(handmade(arguments, content="Saving it."), TEMPLATE)[0]
        self.assertTrue(example["completion"].startswith("Saving it.\n<tool_call>\n"))
        self.assertIn("a\\u003cb.txt", example["completion"])
        parsed = LLM.parse_response({"choices": [{"message": {
            "role": "assistant", "content": example["completion"].removesuffix(END)}}]})
        self.assertEqual((parsed.content, parsed.tool_calls[0].arguments), ("Saving it.", arguments))

    def test_a_request_that_does_not_line_up_with_its_message_is_an_error(self):
        trace = handmade({"path": "a.txt", "content": "x"})
        trace["model_requests"][0]["call_ids"] = ["other"]
        with self.assertRaises(ValueError):
            examples_from_trace(trace, TEMPLATE)

    def test_sampled_text_is_compared_with_the_canonical_target(self):
        arguments = {"path": "a.txt", "content": "x"}
        same = '<tool_call>\n{"name": "write_file", "arguments": ' + json.dumps(arguments) + '}\n</tool_call>'
        for raw_text, expected in ((same, True), (same.replace(": {", ":{", 1), False), (None, None)):
            with self.subTest(raw_text=raw_text):
                example = examples_from_trace(handmade(arguments, raw_text=raw_text), TEMPLATE)[0]
                self.assertIs(example["raw_matches_canonical"], expected)


class RejectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_clean_and_injected_fault_runs_are_eligible(self):
        for task_id in ("inventory_reconciliation-0-clean", "inventory_reconciliation-0-fault",
                        "points_save_recovery-1-fault"):
            with self.subTest(task_id=task_id):
                self.assertIsNone(rejection(*rollout(self.root, task_id, name=task_id)))

    def test_a_failed_run_is_rejected(self):
        record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean")
        self.assertEqual(rejection({**record, "passed": False}, task, trace), "failed")

    def test_a_passed_run_with_the_models_own_mistakes_is_rejected(self):
        cases = {"parse": [ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)],
                 "wrong_path": [call("read_file", path="no-such-file.json")],
                 "bad_arguments": [call("read_file", nope=1)]}
        for name, before in cases.items():
            with self.subTest(name=name):
                record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                              before=before, name=name)
                self.assertTrue(record["passed"], record["checks"])
                self.assertEqual(rejection(record, task, trace), "own_error")

    def test_a_view_that_is_not_the_raw_prefix_is_rejected(self):
        record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                      overrides={"planning": True})
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(rejection(record, task, trace), "edited_view")


def batch(*paths):
    """One turn that reads several paths at once."""
    return LLMResponse("assistant", "", ResponseType.tool_call,
                       tool_calls=[ToolCall("read_file", {"path": path}) for path in paths])


def wrong_first(solution):
    """The scripted solution after one read of a file that does not exist."""
    return [call("read_file", path="no-such-file.json"), *solution]


class FlawedTurnTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_only_the_turn_holding_the_models_own_failed_call_is_flawed(self):
        # The shape seen in teacher data: one batch with a good read and a read of a directory.
        record, task, trace = rollout(
            self.root, "inventory_reconciliation-0-clean",
            script=lambda solution: [batch("inventory.json", "."), *solution[1:]])
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(flawed_turns(record, task, trace), {0: "own_error"})
        self.assertEqual(rejection(record, task, trace), "own_error")
        self.assertIsNone(rejection(record, task, trace, flawed="drop"))

    def test_the_injected_fault_and_a_call_skipped_after_it_are_not_flaws(self):
        # The fault hits the first call of the batch, so the second is skipped.
        record, task, trace = rollout(
            self.root, "inventory_reconciliation-0-fault",
            script=lambda solution: [batch("inventory.json", "shipment.json"), *solution[1:]])
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual([error["error_code"] for error in record["tool_errors"]],
                         ["execution_error", "skipped"])
        self.assertEqual(flawed_turns(record, task, trace), {})
        self.assertIsNone(rejection(record, task, trace))

    def test_a_loose_final_answer_flaws_the_last_turn_only(self):
        loose = lambda solution: [*solution[:-1], answer(solution[-1].content + "\n\nBoth balances verified.")]
        record, task, trace = rollout(self.root, "points_save_recovery-0-clean", script=loose)
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(flawed_turns(record, task, trace), {record["actor_requests"] - 1: "loose_answer"})
        self.assertEqual(rejection(record, task, trace), "loose_answer")
        self.assertIsNone(rejection(record, task, trace, flawed="drop"))

    def test_a_parse_error_rejects_the_whole_rollout_in_either_mode(self):
        record, task, trace = rollout(self.root, "inventory_reconciliation-0-clean",
                                      before=[ResponseError(ResponseErrorCode.INVALID_TOOL_CALL)])
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(rejection(record, task, trace, flawed="drop"), "own_error")


def trajectory(task_id, seed, completions, run_dir="r"):
    return {"task_id": task_id, "seed": seed, "run_dir": run_dir,
            "examples": [{"completion": text} for text in completions]}


class SelectTests(unittest.TestCase):
    def test_identical_trajectories_of_a_task_collapse_to_the_lowest_seed(self):
        kept = select([trajectory("t", 2, ["a", "b"]), trajectory("t", 1, ["a", "b"])], 5)
        self.assertEqual([(t["task_id"], t["seed"]) for t in kept], [("t", 1)])

    def test_cap_keeps_the_shortest_distinct_trajectories_per_task(self):
        pool = [trajectory("t", 0, ["a", "b", "c"]), trajectory("t", 1, ["a", "c"]),
                trajectory("t", 2, ["b", "c"]), trajectory("u", 0, ["a", "b", "c"])]
        kept = select(pool, 2)
        self.assertEqual([(t["task_id"], t["seed"]) for t in kept], [("t", 1), ("t", 2), ("u", 0)])

    def test_the_same_actions_on_different_tasks_are_both_kept(self):
        self.assertEqual(len(select([trajectory("t", 0, ["a"]), trajectory("u", 0, ["a"])], 1)), 2)

    def test_a_trajectory_without_dropped_turns_comes_first(self):
        flawed = trajectory("t", 0, ["x", "a"])
        flawed["examples"][0]["dropped"] = "own_error"
        kept = select([flawed, trajectory("t", 1, ["a", "b", "c"])], 1)
        self.assertEqual([t["seed"] for t in kept], [1])

    def test_a_dropped_answer_ranks_before_a_dropped_failure(self):
        # A failed call stays in the history of every later example; a loose answer touches nothing else.
        loose = trajectory("t", 0, ["a", "b", "x"])
        loose["examples"][2]["dropped"] = "loose_answer"
        failed = trajectory("t", 1, ["x", "a"])
        failed["examples"][0]["dropped"] = "own_error"
        self.assertEqual([t["seed"] for t in select([failed, loose], 1)], [0])


class CollectTests(unittest.TestCase):
    TASKS = ("inventory_reconciliation-0-clean", "inventory_reconciliation-1-clean")

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def run_dir(self, name, seed, *, split="train", backend="vllm", tasks=TASKS, script=None):
        """A bench-shaped run directory whose rollouts come from scripted solutions."""
        out = self.root / name
        out.mkdir()
        records = [rollout(out, task_id, name=task_id, script=script)[0] for task_id in tasks]
        save(out / "results.json", records)
        save(out / "metadata.json", {"suite": f"bench-{split}-v2", "split": split, "seed": seed,
                                     "git_revision": "0" * 40,
                                     "settings": {"backend": backend, "chat_template_sha256": None}})
        return out

    def test_repeated_rollouts_collapse_and_examples_carry_provenance(self):
        examples, metadata = collect([self.run_dir("s0", 0), self.run_dir("s1", 1)], "rft")
        funnel = metadata["funnel"]
        self.assertEqual((funnel["rollouts"], funnel["eligible"], funnel["kept_trajectories"],
                          funnel["tasks_covered"]), (4, 4, 2, 2))
        self.assertEqual((funnel["examples_before_dedup"], funnel["examples"], len(examples)), (6, 6, 6))
        self.assertEqual(len({e["id"] for e in examples}), len(examples))
        provenance = examples[0]["provenance"]
        self.assertEqual((provenance["track"], provenance["seed"], provenance["run_dir"],
                          provenance["skeleton"]), ("rft", 0, "s0", "inventory_reconciliation"))
        trace = json.loads((self.root / "s0" / provenance["task_id"] / "state" / "runs"
                            / f"{provenance['run_id']}.jsonl").read_text())
        self.assertEqual(trace["model_requests"][provenance["request_index"]]["status"], "completed")
        self.assertEqual(metadata["funnel"]["by_skeleton"],
                         {"inventory_reconciliation": {"trajectories": 2, "tasks": 2}})

    def test_two_distinct_trajectories_share_their_common_prefix_once(self):
        # Seed 1 re-reads the second source before answering: same first two decisions, then it
        # diverges. Prompts never match across runs (each names its own temporary workspace),
        # so duplicates are found by task and action history, not by prompt text.
        task = self.TASKS[:1]
        again = lambda solution: [*solution[:2], solution[1], solution[2]]
        examples, metadata = collect([self.run_dir("s0", 0, tasks=task),
                                      self.run_dir("s1", 1, tasks=task, script=again)], "rft")
        funnel = metadata["funnel"]
        self.assertEqual((funnel["kept_trajectories"], funnel["examples_before_dedup"],
                          funnel["examples"]), (2, 7, 5))
        self.assertEqual([e["provenance"]["seed"] for e in examples], [0, 0, 0, 1, 1])
        self.assertEqual(collect([self.run_dir("t0", 0, tasks=task),
                                  self.run_dir("t1", 1, tasks=task, script=again)], "rft",
                                 max_per_task=1)[1]["funnel"]["examples"], 3)

    def test_drop_mode_keeps_a_flawed_rollout_without_its_flawed_turn(self):
        run = self.run_dir("s0", 0, tasks=self.TASKS[:1], script=wrong_first)
        examples, metadata = collect([run], "rft")
        self.assertEqual((examples, metadata["funnel"]["rejected"]["own_error"], metadata["flawed_turns"]),
                         ([], 1, "reject"))
        examples, metadata = collect([run], "rft", flawed="drop")
        funnel = metadata["funnel"]
        self.assertEqual((metadata["flawed_turns"], funnel["eligible"], funnel["examples_before_dedup"],
                          funnel["examples"]), ("drop", 1, 3, 3))
        self.assertEqual(funnel["dropped_turns"], {"own_error": 1, "loose_answer": 0})
        self.assertEqual([e["provenance"]["request_index"] for e in examples], [1, 2, 3])
        self.assertEqual([e["provenance"]["dropped_turns"] for e in examples], [[0]] * 3)
        # The failed call stays in the history the later turns saw; it is never a target.
        self.assertIn("no-such-file.json", examples[0]["prompt"])
        self.assertFalse(any("no-such-file.json" in e["completion"] for e in examples))

    def test_a_flawed_trajectory_ranks_after_a_clean_one_and_is_not_its_duplicate(self):
        task = self.TASKS[:1]
        runs = [self.run_dir("s0", 0, tasks=task, script=wrong_first), self.run_dir("s1", 1, tasks=task)]
        examples, metadata = collect(runs, "rft", flawed="drop")
        self.assertEqual((metadata["funnel"]["kept_trajectories"], metadata["funnel"]["examples"]), (2, 6))
        self.assertEqual([e["provenance"]["seed"] for e in examples], [1, 1, 1, 0, 0, 0])
        self.assertEqual([e["provenance"]["dropped_turns"] for e in examples], [[]] * 3 + [[0]] * 3)
        only = collect(runs, "rft", flawed="drop", max_per_task=1)[0]
        self.assertEqual({e["provenance"]["seed"] for e in only}, {1})

    def test_only_the_train_split_can_become_training_data(self):
        for split in ("val", "test"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train"):
                collect([self.run_dir(split, 0, split=split)], "rft")

    def test_the_backend_must_match_the_track_unless_marked_pilot(self):
        local = self.run_dir("local", 0, backend="llama_cpp")
        with self.assertRaisesRegex(ValueError, "backend"):
            collect([local], "rft")
        self.assertTrue(collect([local], "rft", pilot=True)[1]["pilot"])
        with self.assertRaisesRegex(ValueError, "backend"):
            collect([self.run_dir("api", 0, backend="deepseek")], "rft")
        with self.assertRaisesRegex(ValueError, "[Ss]cripted"):
            collect([self.run_dir("gate", 0, backend="scripted")], "rft", pilot=True)

    def test_the_same_task_at_the_same_seed_twice_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            collect([self.run_dir("a", 0), self.run_dir("b", 0)], "rft")

    def test_a_run_sampled_under_another_template_is_an_error(self):
        run = self.run_dir("old", 0)
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["settings"]["chat_template_sha256"] = "f" * 64
        save(run / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "template"):
            collect([run], "rft")

    def test_command_writes_once_and_never_overwrites(self):
        run, out = self.run_dir("s0", 0), self.root / "sft"
        argv = ["export", "--track", "rft", "--output", str(out), str(run)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            export.main()
        lines = (out / "examples.jsonl").read_text().splitlines()
        metadata = json.loads((out / "metadata.json").read_text())
        self.assertEqual(len(lines), metadata["funnel"]["examples"])
        self.assertEqual(set(json.loads(lines[0])), {"id", "prompt", "completion", "provenance"})
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as error:
            export.main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((out / "examples.jsonl").read_text().splitlines(), lines)

    def test_command_takes_the_flawed_turn_mode(self):
        run, out = self.run_dir("s0", 0), self.root / "sft"
        argv = ["export", "--track", "rft", "--flawed-turns", "drop", "--output", str(out), str(run)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            export.main()
        self.assertEqual(json.loads((out / "metadata.json").read_text())["flawed_turns"], "drop")


if __name__ == "__main__":
    unittest.main()
```

The error messages your `ValueError`s need, because the tests match on them: the split error
contains `train`, the backend error contains `backend`, the scripted error contains `Scripted`
or `scripted`, the repeat error contains `Duplicate` or `duplicate`, the template error contains
`template`.

## Appendix B — `evals/sft/teacher.py` (reference, plumbing)

```python
"""A stronger API model behind the same interface as `VLLMClient`, for Stage 9 teacher rollouts.

The teacher is DeepSeek's OpenAI-compatible chat API. The harness's message list is already in
that shape, so it is sent nearly as it is and the reply goes straight through
`LLM.parse_response`, the one parser. The API has no token-count endpoint, so the context
budget is the student's: the request rendered with the project template and counted with the
student's own vocabulary. The harness triggers then fire exactly where the student's would.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Callable

from ...config import BENCH_DECODING, BENCH_N_CTX, CHAT_TEMPLATE_PATH, MAX_TOOL_CALLS_PER_RESPONSE
from ...llm import (LLM, QWEN_END, LLMResponse, ResponseError, ResponseErrorCode,
                    compile_chat_template, render_chat)

BASE_URL = "https://api.deepseek.com"
API_KEY_VARIABLE = "DEEPSEEK_API_KEY"


def student_token_counter(gguf: Path) -> tuple[Callable[[str], int], str]:
    """(count, label) from a GGUF's vocabulary alone; no weights are loaded.

    Args:
        gguf: a GGUF of the student model. Only its tokenizer is read.
    """
    from llama_cpp import Llama

    vocabulary = Llama(model_path=str(gguf), vocab_only=True, verbose=False)
    label = f"{vocabulary.metadata.get('general.name')} ({Path(gguf).name})"
    return (lambda text: len(vocabulary.tokenize(text.encode("utf-8"), add_bos=False, special=True)),
            label)


class TeacherClient:
    """DeepSeek's chat API as the model backend, with the same interface as `LLM`.

    Args:
        model: API model ID. An alias such as `deepseek-flash` moves between releases, so the
            release name the server reports is recorded in `settings()`.
        count: student tokenizer, text -> token count (see `student_token_counter`).
        budget_tokenizer: label for `count`, recorded in `settings()`.
        effort: `reasoning_effort` (low, high, max); "none" turns thinking off.
        sample: bookkeeping label for this pass. The API takes no seed.
        max_output_tokens: the teacher's own reply limit. Thinking counts toward it.
        n_ctx, reserve: the student's window and reply reserve, used for the budget only.
    """

    def __init__(self, model: str = "deepseek-flash", *, count: Callable[[str], int],
                 budget_tokenizer: str, effort: str = "low", sample: int = 0,
                 max_output_tokens: int = 8192, n_ctx: int = BENCH_N_CTX,
                 reserve: int = BENCH_DECODING["max_tokens"], base_url: str = BASE_URL,
                 timeout: float = 600.0, retries: int = 3, backoff: float = 2.0, client=None):
        import httpx

        if client is None:
            key = os.environ.get(API_KEY_VARIABLE)
            if not key:
                raise ValueError(f"Set {API_KEY_VARIABLE} to run the teacher.")
            client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  headers={"Authorization": f"Bearer {key}"})
        self.http, self.base_url, self.retries, self.backoff = client, base_url, retries, backoff
        self.model, self.effort, self.sample = model, effort, sample
        self.max_output_tokens, self.n_ctx, self.reserve = max_output_tokens, n_ctx, reserve
        self._count, self.budget_tokenizer = count, budget_tokenizer
        self._template = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
        served = {entry["id"]: entry for entry in self._request("GET", "/models")["data"]}
        if model not in served:
            raise ValueError(f"The API does not serve {model!r}; it serves {sorted(served)}.")
        self.model_name = served[model].get("name")
        if count("<|im_start|>" + QWEN_END) != 2:
            raise ValueError("The chat template's <|im_start|>/<|im_end|> are not single tokens "
                             "in the student tokenizer.")
        # First tool-call ID of a turn -> the reasoning that came with it. The API asks for a
        # turn's reasoning_content back whenever tools are in play; the harness does not keep it.
        self._reasoning: dict[str, str] = {}

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        import httpx

        for attempt in range(self.retries):
            try:
                response = self.http.request(method, path, json=body)
                if response.status_code < 500 and response.status_code != 429:
                    response.raise_for_status()
                    return response.json()
                error: Exception = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            except (httpx.TransportError, httpx.TimeoutException) as transport:
                error = transport
            if attempt == self.retries - 1:
                raise error
            time.sleep(self.backoff * 2 ** attempt)
        raise AssertionError("unreachable")

    def _messages(self, messages: list) -> list[dict]:
        """The harness's messages with only the fields the API defines."""
        converted = []
        for message in messages:
            entry = {"role": message["role"], "content": message.get("content") or ""}
            if message["role"] == "tool":
                entry["tool_call_id"] = message["tool_call_id"]
            elif message["role"] == "assistant" and message.get("tool_calls"):
                entry["tool_calls"] = message["tool_calls"]
                reasoning = self._reasoning.get(message["tool_calls"][0]["id"])
                if reasoning is not None:
                    entry["reasoning_content"] = reasoning
            converted.append(entry)
        return converted

    def measure_context(self, messages, tools, *, max_tokens: int | None = None) -> dict[str, Any]:
        """Same contract as `LLM.measure_context`, in student tokens against the student's window."""
        configured = self.reserve if max_tokens is None else max_tokens
        reserve = configured if configured is not None and configured > 0 else None
        count = self._count(render_chat(self._template, messages, tools))
        return {"count_method": "exact", "prompt_tokens": count, "window_tokens": self.n_ctx,
                "response_reserve": reserve,
                "remaining_tokens": self.n_ctx - count - reserve if reserve is not None else None}

    def generate(self, messages, tools, *, max_tokens: int | None = None,
                 max_tool_calls: int | None = None) -> LLMResponse:
        # Thinking counts toward max_tokens, so the caller's reserve (sized for the student) is
        # not used here. Thinking mode ignores sampling parameters and the API takes no seed.
        body: dict[str, Any] = {"model": self.model, "messages": self._messages(messages),
                                "max_tokens": self.max_output_tokens, "stream": False}
        if self.effort == "none":
            body["thinking"] = {"type": "disabled"}
        else:
            body.update(thinking={"type": "enabled"}, reasoning_effort=self.effort)
        if tools:
            body["tools"] = list(tools.values())
        raw = self._request("POST", "/chat/completions", body)
        try:
            parsed = LLM.parse_response(
                raw, MAX_TOOL_CALLS_PER_RESPONSE if max_tool_calls is None else max_tool_calls)
        except ResponseError as error:
            error.raw_response = raw
            raise
        reasoning = raw["choices"][0]["message"].get("reasoning_content")
        if parsed.tool_calls and isinstance(reasoning, str):
            self._reasoning[parsed.tool_calls[0].call_id] = reasoning
        parsed.raw_response = raw
        return parsed

    def settings(self) -> dict[str, Any]:
        return {"backend": "deepseek", "base_url": self.base_url, "model": self.model,
                "model_name": self.model_name, "effort": self.effort, "sample": self.sample,
                "max_tokens": self.max_output_tokens, "n_ctx": self.n_ctx,
                "response_reserve": self.reserve, "budget_tokenizer": self.budget_tokenizer,
                # The template is used for the budget only; the teacher never sampled under it.
                "chat_template_sha256": None}

    def close(self) -> None:
        self.http.close()
```

This code has met the live API: the Step 8 smoke and the Step 9 passes ran through it.

## Appendix C — `tests/test_teacher_backend.py`

```python
"""The teacher backend speaks DeepSeek's chat API, shares the one parser, and budgets in student tokens."""

import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from agent_from_scratch.config import BENCH_DECODING, BENCH_N_CTX, CHAT_TEMPLATE_PATH
from agent_from_scratch.evals.bench import run
from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.sft.export import TRACK_BACKENDS, examples_from_trace, rejection
from agent_from_scratch.evals.sft.teacher import TeacherClient
from agent_from_scratch.llm import ResponseError, ResponseErrorCode, compile_chat_template, render_chat
from agent_from_scratch.tools.register import default_tool_schemas

TEMPLATE = compile_chat_template(CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
CONVERSATION = [
    {"role": "system", "content": "System rules."},
    {"role": "user", "content": "Read both files."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "a.json"})}},
        {"id": "t2", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "b.json"})}}]},
    {"role": "tool", "tool_call_id": "t1", "content": json.dumps({"ok": False, "error_message": "Temporary failure."})},
    {"role": "tool", "tool_call_id": "t2", "content": json.dumps({"ok": False, "error_code": "skipped"})},
    {"role": "user", "content": "[Runtime feedback] Try again."},
]


def count(text: str) -> int:
    """A stand-in student tokenizer: one token per control marker, one per other character."""
    specials = text.count("<|im_start|>") + text.count("<|im_end|>")
    return specials + len(text.replace("<|im_start|>", "").replace("<|im_end|>", ""))


def reply(content="", *calls, reasoning="Thinking it through.", finish=None):
    """One chat completion as the API returns it; `calls` are (id, name, arguments)."""
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning,
               "tool_calls": [{"index": index, "id": call_id, "type": "function",
                               "function": {"name": name, "arguments": json.dumps(arguments)}}
                              for index, (call_id, name, arguments) in enumerate(calls)] or None}
    return {"model": "deepseek-flash",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170,
                      "prompt_cache_hit_tokens": 50, "prompt_cache_miss_tokens": 100,
                      "completion_tokens_details": {"reasoning_tokens": 6}}}


class FakeDeepSeek:
    """A scripted server: GET /models and POST /chat/completions."""

    def __init__(self, replies=(), *, failures=0):
        self.replies, self.failures, self.sent = list(replies), failures, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash"}]})
        if self.failures:
            self.failures -= 1
            return httpx.Response(503, text="busy")
        self.sent.append(json.loads(request.content))
        return httpx.Response(200, json=self.replies.pop(0))


def teacher(server: FakeDeepSeek, model: str = "deepseek-flash", **kwargs) -> TeacherClient:
    http = httpx.Client(base_url="https://deepseek.test", transport=httpx.MockTransport(server))
    return TeacherClient(model, client=http, **{"count": count, "budget_tokenizer": "fake", "backoff": 0, **kwargs})


class ConversionTests(unittest.TestCase):
    def test_request_passes_the_harness_messages_and_schemas_through(self):
        server = FakeDeepSeek([reply("Done.")])
        teacher(server).generate(CONVERSATION, default_tool_schemas)
        sent = server.sent[0]
        self.assertEqual(sent["model"], "deepseek-flash")
        self.assertEqual(sent["messages"], CONVERSATION)
        self.assertEqual(sent["tools"], list(default_tool_schemas.values()))

    def test_request_sets_thinking_effort_and_no_sampling_parameters(self):
        server = FakeDeepSeek([reply("Done."), reply("Done.")])
        teacher(server, effort="high").generate(CONVERSATION, default_tool_schemas, max_tokens=512)
        sent = server.sent[0]
        for unused in ("temperature", "top_p", "top_k", "seed", "tool_choice"):
            self.assertNotIn(unused, sent)
        self.assertEqual((sent["thinking"], sent["reasoning_effort"]), ({"type": "enabled"}, "high"))
        self.assertEqual(sent["max_tokens"], 8192)  # thinking counts toward it; not the student's reserve
        teacher(server, effort="none").generate(CONVERSATION, default_tool_schemas)
        self.assertEqual(server.sent[1]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", server.sent[1])

    def test_reply_goes_through_the_shared_parser(self):
        server = FakeDeepSeek([reply("Reading.", ("call_00_x", "read_file", {"path": "a.json"}))])
        response = teacher(server).generate(CONVERSATION[:2], default_tool_schemas)
        self.assertEqual([(c.call_id, c.name, c.arguments) for c in response.tool_calls],
                         [("call_00_x", "read_file", {"path": "a.json"})])
        self.assertEqual((response.content, response.finish_reason), ("Reading.", "tool_calls"))
        self.assertEqual(response.usage, {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170})
        self.assertEqual(response.raw_response["model"], "deepseek-flash")
        self.assertEqual(response.raw_response["usage"]["prompt_cache_hit_tokens"], 50)
        json.dumps(response.raw_response)  # the trace must be able to store it

    def test_own_turns_are_replayed_with_their_reasoning(self):
        first = reply("", ("call_00_x", "read_file", {"path": "a.json"}), reasoning="Read a.json first.")
        server = FakeDeepSeek([first, reply("Done.")])
        model = teacher(server)
        message = model.generate(CONVERSATION[:2], default_tool_schemas).to_message()
        model.generate([*CONVERSATION[:2], message,
                        {"role": "tool", "tool_call_id": "call_00_x", "content": json.dumps({"ok": True})}],
                       default_tool_schemas)
        replayed = server.sent[1]["messages"][2]
        self.assertEqual(replayed, {**message, "reasoning_content": "Read a.json first."})
        # A turn this client did not produce has no reasoning to return, and none is invented.
        self.assertNotIn("reasoning_content", server.sent[0]["messages"][1])

    def test_finish_reasons_reach_the_shared_parser(self):
        cases = {"length": ResponseErrorCode.TRUNCATED_RESPONSE,
                 "content_filter": ResponseErrorCode.UNSUPPORTED_FINISH_REASON,
                 "insufficient_system_resource": ResponseErrorCode.UNSUPPORTED_FINISH_REASON}
        for finish, code in cases.items():
            with self.subTest(finish=finish), self.assertRaises(ResponseError) as caught:
                teacher(FakeDeepSeek([reply("partial", finish=finish)])).generate(
                    CONVERSATION[:2], default_tool_schemas)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.raw_response["usage"]["prompt_tokens"], 150)
            self.assertEqual(caught.exception.raw_response["choices"][0]["finish_reason"], finish)

    def test_a_reply_without_a_choice_is_an_invalid_response(self):
        with self.assertRaises(ResponseError) as caught:
            teacher(FakeDeepSeek([{"error": "unexpected"}])).generate(CONVERSATION[:2], default_tool_schemas)
        self.assertEqual(caught.exception.code, ResponseErrorCode.INVALID_RESPONSE)
        self.assertEqual(caught.exception.raw_response, {"error": "unexpected"})

    def test_budget_is_the_student_token_count_of_the_student_rendering(self):
        server = FakeDeepSeek()
        measured = teacher(server).measure_context(CONVERSATION, default_tool_schemas)
        tokens = count(render_chat(TEMPLATE, CONVERSATION, default_tool_schemas))
        self.assertEqual(measured, {"count_method": "exact", "prompt_tokens": tokens,
                                    "window_tokens": BENCH_N_CTX,
                                    "response_reserve": BENCH_DECODING["max_tokens"],
                                    "remaining_tokens": BENCH_N_CTX - tokens - BENCH_DECODING["max_tokens"]})
        self.assertEqual(server.sent, [])  # counted locally; the API has no count endpoint

    def test_settings_identify_the_teacher(self):
        model = teacher(FakeDeepSeek(), sample=3)
        settings = model.settings()
        self.assertEqual((settings["backend"], settings["model"], settings["model_name"], settings["effort"],
                          settings["sample"], settings["budget_tokenizer"]),
                         ("deepseek", "deepseek-flash", "DeepSeek-V4.1-Flash", "low", 3, "fake"))
        self.assertEqual(settings["backend"], TRACK_BACKENDS["teacher"])
        self.assertIsNone(settings["chat_template_sha256"])  # the teacher never sampled under it
        model.close()
        self.assertTrue(model.http.is_closed)

    def test_an_unserved_model_a_wrong_tokenizer_or_a_missing_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "deepseek-v9"):
            teacher(FakeDeepSeek(), "deepseek-v9")
        with self.assertRaisesRegex(ValueError, "single tokens"):
            teacher(FakeDeepSeek(), count=len)
        with patch.dict(os.environ, clear=True), self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            TeacherClient(count=count, budget_tokenizer="fake")

    def test_server_errors_are_retried(self):
        server = FakeDeepSeek([reply("Done.")], failures=2)
        self.assertEqual(teacher(server).generate(CONVERSATION, default_tool_schemas).content, "Done.")
        with self.assertRaises(RuntimeError):
            teacher(FakeDeepSeek([reply("Done.")], failures=3)).generate(CONVERSATION, default_tool_schemas)


class HarnessTests(unittest.TestCase):
    def scripted(self, skeleton, ctx):
        """The skeleton's scripted solution, spoken as chat completions."""
        with TemporaryDirectory() as temporary:
            task = skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"),
                                  Path(temporary), ctx)
        return [reply(response.content, *[(f"call_{index:02d}_{n}", c.name, c.arguments)
                                          for n, c in enumerate(response.tool_calls)],
                      reasoning=f"step {index}")
                for index, response in enumerate(skeleton.solution(task))]

    def test_a_teacher_rollout_passes_the_verifier_and_exports(self):
        skeleton, ctx = next((s, c) for s, c in specs("train") if c.id == "points_save_recovery-0-fault")
        server = FakeDeepSeek(self.scripted(skeleton, ctx))
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            record = run_case(teacher(server), skeleton, ctx, out)
            self.assertTrue(record["passed"], record["checks"])
            trace = json.loads((out / "state" / "runs" / f"{record['run_id']}.jsonl").read_text())
            task = json.loads((out / "task.json").read_text())
        self.assertEqual(record["usage"]["prompt_tokens"], 150 * record["model_requests"])
        self.assertIsNone(rejection(record, task, trace))
        examples = examples_from_trace(trace, TEMPLATE)
        self.assertEqual(len(examples), record["model_requests"])
        self.assertTrue(examples[0]["completion"].startswith('<tool_call>\n{"name": "read_file"'))
        self.assertEqual(examples[-1]["completion"], "SAVED<|im_end|>")
        # The budget the harness acted on is the student's count of exactly the exported prompt.
        for example in examples:
            budget = trace["model_requests"][example["request_index"]]["budget"]
            self.assertEqual(budget["prompt_tokens"], count(example["prompt"]))
        # Every earlier turn went back with the reasoning the model produced for it.
        for index, sent in enumerate(server.sent):
            turns = [m for m in sent["messages"] if m["role"] == "assistant"]
            self.assertEqual([m["reasoning_content"] for m in turns], [f"step {n}" for n in range(index)])


class BenchTests(unittest.TestCase):
    def test_the_teacher_may_only_see_the_train_split(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            for split in ("val", "test"):
                args = ["bench", "--backend", "teacher", "--split", split, "--final",
                        "--student-gguf", __file__, "--output", str(Path(temporary) / split)]
                with self.subTest(split=split), patch.object(sys, "argv", args), patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_the_teacher_needs_the_student_tokenizer_file(self):
        with TemporaryDirectory() as temporary, patch.object(run, "load_model") as model:
            for extra in ([], ["--student-gguf", str(Path(temporary) / "missing.gguf")]):
                args = ["bench", "--backend", "teacher", "--split", "train", *extra,
                        "--output", str(Path(temporary) / "out")]
                with self.subTest(extra=extra), patch.object(sys, "argv", args), patch("sys.stderr"):
                    with self.assertRaises(SystemExit) as error:
                        run.main()
                    self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_bench_builds_the_teacher_backend(self):
        args = SimpleNamespace(backend="teacher", teacher_model="deepseek-flash", teacher_effort="low",
                               seed=4, student_gguf=Path("student.gguf"))
        with patch("agent_from_scratch.evals.sft.teacher.TeacherClient") as client, \
                patch("agent_from_scratch.evals.sft.teacher.student_token_counter",
                      return_value=(count, "Qwen3 (student.gguf)")) as counter:
            run.load_model(args)
        counter.assert_called_once_with(Path("student.gguf"))
        client.assert_called_once_with("deepseek-flash", count=count, budget_tokenizer="Qwen3 (student.gguf)",
                                       effort="low", sample=4, n_ctx=BENCH_N_CTX,
                                       reserve=BENCH_DECODING["max_tokens"])


if __name__ == "__main__":
    unittest.main()
```
