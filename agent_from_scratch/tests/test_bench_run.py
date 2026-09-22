"""The bench runner works end to end for a hand-built skeleton, before any real one exists."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.evals.bench.faults import FaultyTool
from agent_from_scratch.evals.bench.run import registry_for, run_case, specs
from agent_from_scratch.evals.bench.skeletons import Skeleton, answer, call
from agent_from_scratch.evals.bench.spec import Answer, BuildContext, Expect, Fault, Task
from agent_from_scratch.llm import LLM


def _write_config(rng, workspace, ctx):
    correct = {"output": "report.json", "retries": 3}
    if ctx.condition == "fault":
        (workspace / "config.json").write_text('{"output": "old.json", "retries": 3}\n')
    else:
        (workspace / "config.json").write_text('{"output": "report.json", "retries": 3}\n')
    fault = Fault("shell", 1, "forced check failure") if ctx.condition == "fault" else None
    return Task(id=ctx.id, skeleton=ctx.name, family=ctx.family, split=ctx.split,
                pair_id=ctx.pair_id, condition=ctx.condition, max_iterations=5,
                prompt="Run check_fixture, fix config.json's output if needed, reply CHECKED.",
                tools=("read_file", "write_file", "shell"),
                expect=Expect(answer=Answer(value="CHECKED", format="required"),
                             files={"config.json": '{"output": "report.json", "retries": 3}'},
                             process=(("check_before_write", "passing_check_after_last_write")
                                      if ctx.condition == "fault" else ("no_write_attempts",))),
                claim_tokens=("CHECKED",), fault=fault,
                fault_signal=("shell", "forced") if fault else None,
                debug={"correct": correct})


def _solve(task):
    if task.condition == "fault":
        # A diagnostic check comes first (forced to fail by the fault), then the fix, then a
        # second, passing check — satisfying both "check_before_write" and
        # "passing_check_after_last_write" declared in _write_config's expect.process.
        return [call("shell", command_id="check_fixture"),
                call("read_file", path="config.json"),
                call("write_file", path="config.json", content='{"output": "report.json", "retries": 3}'),
                call("shell", command_id="check_fixture"), answer("CHECKED")]
    return [call("shell", command_id="check_fixture"), answer("CHECKED")]


PROBE = Skeleton(name="probe_check", family="recovery", split="dev", seeds=(0,),
                 recovery=True, build=_write_config, solution=_solve)


def _model(responses):
    model = Mock(spec=LLM)
    model.settings.return_value = {}
    model.read_usage.side_effect = LLM.read_usage
    model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 1000,
        "window_tokens": 32768, "response_reserve": 2048, "remaining_tokens": 29720}
    model.generate.side_effect = list(responses)
    return model


class RunFrameworkTests(unittest.TestCase):
    def test_registry_for_restricts_to_the_tasks_declared_tools(self):
        with TemporaryDirectory() as private:
            workspace = Path(private) / "workspace"
            workspace.mkdir()
            ctx = BuildContext("probe_check", "recovery", "dev", 0, None)
            task = _write_config(None, workspace, ctx)
            registry = registry_for(task, workspace, Path(private))
            self.assertEqual(set(registry.schemas()), {"read_file", "write_file", "shell"})

    def test_run_case_scores_a_clean_recovery_task(self):
        ctx = BuildContext("probe_check", "recovery", "dev", 0, "clean")
        with TemporaryDirectory() as directory:
            record = run_case(_model(_solve(_write_config(None, Path(directory), ctx))),
                              PROBE, ctx, Path(directory) / "run")
        self.assertTrue(record["passed"], record["checks"])
        self.assertEqual(record["id"], "probe_check-0-clean")

    def test_run_case_exercises_the_shell_fault_and_recovers(self):
        ctx = BuildContext("probe_check", "recovery", "dev", 0, "fault")
        with TemporaryDirectory() as directory:
            probe_workspace = Path(directory) / "probe"
            probe_workspace.mkdir()
            task = _write_config(None, probe_workspace, ctx)
            record = run_case(_model(_solve(task)), PROBE, ctx, Path(directory) / "run")
        self.assertTrue(record["passed"], record["checks"])
        self.assertTrue(record["fault_encountered"])

    def test_specs_lists_every_seed_and_condition_pair(self):
        from agent_from_scratch.evals.bench import skeletons
        original = list(skeletons.SKELETONS)
        skeletons.SKELETONS[:] = [PROBE]
        try:
            entries = specs()
            self.assertEqual([(s.name, c.seed, c.condition) for s, c in entries],
                             [("probe_check", 0, "clean"), ("probe_check", 0, "fault")])
        finally:
            skeletons.SKELETONS[:] = original


class DispatcherTests(unittest.TestCase):
    def test_bench_is_a_registered_command(self):
        from agent_from_scratch.evals.__main__ import COMMANDS
        self.assertEqual(COMMANDS["bench"], "bench.run")


class FinalGuardTests(unittest.TestCase):
    def test_split_test_without_final_is_rejected(self):
        import sys
        from agent_from_scratch.evals.bench.run import main
        old_argv = sys.argv
        sys.argv = ["bench", "--output", "/tmp/should-not-be-created", "--split", "test"]
        try:
            with self.assertRaises(SystemExit):
                main()
        finally:
            sys.argv = old_argv

    def test_drift_without_allow_drift_is_rejected(self):
        import sys
        from unittest.mock import patch
        from agent_from_scratch.evals.bench.run import main
        old_argv = sys.argv
        sys.argv = ["bench", "--output", "/tmp/should-not-be-created-2", "--split", "test", "--final"]
        try:
            # Patch where manifest_drift is defined: `main()` does `from .manifest import
            # manifest_drift` fresh on every call, so the patched module attribute is what
            # it picks up, with no need for run.py to import manifest.py at module level.
            with patch("agent_from_scratch.evals.bench.manifest.manifest_drift",
                      return_value={"memory": ("off", "on")}):
                with self.assertRaises(SystemExit):
                    main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
