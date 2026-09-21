"""The pressure suite builds deterministic workspaces whose reads stay whole and answers unique."""

from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.evals import pressure
from agent_from_scratch.llm import LLM
from agent_from_scratch.tests.test_behavior_eval import answer, call
from agent_from_scratch.tools.files import ReadFileTool


class PressureTests(unittest.TestCase):
    def build(self, spec, seed=11):
        root = Path(self.enterContext(TemporaryDirectory()))
        prompt, expected = spec["build"](random.Random(f"{seed}:{spec['id']}"), root, **spec["args"])
        return root, prompt, expected

    def test_workspaces_are_deterministic_and_each_file_reads_in_one_window(self):
        for spec in pressure.TASKS:
            with self.subTest(task=spec["id"]):
                first, _, expected = self.build(spec)
                second, _, again = self.build(spec)
                self.assertEqual(expected, again)
                files = sorted(p.name for p in first.iterdir())
                self.assertEqual(len(files), pressure.FILES)
                self.assertEqual([(first / n).read_bytes() for n in files],
                                 [(second / n).read_bytes() for n in files])
                reader = ReadFileTool(first, restrict_to_workspace=True)
                for name in files:
                    result = reader.invoke({"path": name})
                    self.assertTrue(result.ok and result.output["eof"], result)

    def test_the_answer_appears_where_the_task_says_it_does(self):
        root, _, code = self.build(pressure.TASKS[0])
        holders = [p.name for p in sorted(root.iterdir()) if f"ACCESS CODE: {code}" in p.read_text()]
        self.assertEqual(holders, ["report_07.txt"])
        root, prompt, code = self.build(pressure.TASKS[2])
        first = prompt.split()[2].rstrip(".")
        chain = [first]
        while "NEXT FILE" in (text := (root / chain[-1]).read_text()):
            chain.append(text.splitlines()[-1].removeprefix("NEXT FILE: "))
        self.assertEqual(len(chain), pressure.FILES)
        self.assertTrue(text.endswith(f"FINAL ANSWER: {code}\n"))
        root, _, codes = self.build(pressure.TASKS[3])
        found = {p.name: line.removeprefix("ACCESS CODE: ") for p in sorted(root.iterdir())
                 for line in p.read_text().splitlines() if line.startswith("ACCESS CODE: ")}
        self.assertEqual(list(found), ["report_03.txt", "report_08.txt"])
        self.assertEqual(",".join(found.values()), codes)

    def test_run_case_scores_the_final_answer_and_records_trajectory_fields(self):
        model = Mock(spec=LLM)
        model.settings.return_value = {}
        model.read_usage.side_effect = LLM.read_usage
        model.measure_context.return_value = {"count_method": "exact", "prompt_tokens": 15000,
            "window_tokens": 32768, "response_reserve": 2048, "remaining_tokens": 15720}
        spec = pressure.TASKS[0]
        _, _, code = self.build(spec)
        model.generate.side_effect = [call("read_file", path="report_01.txt"), answer(code)]
        with TemporaryDirectory() as directory:
            record = pressure.run_case(model, spec, Path(directory) / "run", seed=11)
        self.assertTrue(record["passed"], record["checks"])
        self.assertTrue(record["answer_found"])
        self.assertEqual(record["actions"], ["explore", "answer"])
        self.assertEqual(record["peak_context_ratio"], 0.4883)


if __name__ == "__main__":
    unittest.main()
