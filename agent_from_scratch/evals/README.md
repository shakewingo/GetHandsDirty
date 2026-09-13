# Foundation dev checks

Run from the repository root in the `transformer-practice` environment:

```sh
python -m unittest discover -s agent_from_scratch/tests -v
python -m agent_from_scratch.agent
```

In the REPL, `/read llm.py` explicitly requests a full-file read with a coverage
check. Paths may contain spaces. A plain sentence still uses the ordinary agent
loop: it has no automatic task-condition extraction. A successful coverage check
does not verify the accuracy of the final summary.

The Python equivalent is:

```python
from pathlib import Path
from agent_from_scratch.agent import Agent
from agent_from_scratch.llm import LLM
from agent_from_scratch.verification import full_file_check

workspace = Path("agent_from_scratch").resolve()
llm = LLM(temperature=0, max_tokens=2048, n_ctx=8000,
          chat_template_path=workspace / "prompts/qwen_chat.jinja")
agent = Agent(llm)
result = agent.run_turn(
    f"Read file {workspace / 'llm.py'}",
    completion_check=full_file_check(workspace, "llm.py"),
)
print(result.stop_reason, result.completion_check, result.final_answer)
```

For a real-model comparison, copy a fixed source file into a disposable workspace
first. Keep the same fixture for every run; changing `llm.py` during implementation
otherwise changes both the agent and the task. Use a new output directory each time:

```sh
mkdir -p outputs/foundation-demo/workspace
cp agent_from_scratch/llm.py outputs/foundation-demo/workspace/llm.py
python -m agent_from_scratch.evals.foundation \
  --workspace outputs/foundation-demo/workspace \
  --output outputs/foundation-demo/results \
  --cases exact,guarded,guided,partial --seeds 11,22,33 --read-bytes 8192
```

`exact` and `guarded` use the same initial sentence. Only `guarded` attaches the
coverage checker. `guided` explicitly asks for all chunks; `partial` asks for the
first 1024 bytes. `history` adds an actual prior exchange. `edit` resets
`config.json` inside the supplied workspace, then asks for a read/change/read-back
task. Run `edit` only in a disposable workspace.

`--readonly` blocks writes but keeps their schemas exposed and records attempted
writes. For a writable dev run, `changed_files` records file hash differences.
Changing the target aborts the remaining comparison. Output must be outside the
workspace so recording evidence is not counted as a task side effect.

Results contain raw model response files and run traces, fixture/template/code
hashes, model settings, coverage, tool errors, token counts and elapsed time.
`answer_relevant` starts as null: review the final answer against the fixture and
save a separate review record. Do not infer overall task success from
`final_response` or `read_coverage_passed` alone. Failed and interrupted comparisons
are evidence too; do not overwrite them or count them as completed trials.
