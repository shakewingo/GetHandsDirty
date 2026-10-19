"""Stage 9 Item 7 shapes: act/ask pairs differ only in whether the fact is established, asking
where acting was expected is failed and counted, and JSON edits must leave valid JSON."""

import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.bench.run import run_case, specs
from agent_from_scratch.evals.bench.skeletons import answer, call
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.tests.test_bench_skeletons import _model

PAIRS = ("followup_same_record", "followup_file_from_listing", "lookup_then_act", "followup_move_meeting")


class Item7Tests(unittest.TestCase):
    def entries(self, name):
        return [(s, c) for s, c in specs() if s.name == name]

    def build(self, skeleton, ctx):
        root = Path(self.enterContext(TemporaryDirectory()))
        return root, skeleton.build(random.Random(f"bench:{ctx.name}:{ctx.seed}:{ctx.condition}"), root, ctx)

    def run_script(self, skeleton, ctx, responses):
        return run_case(_model(responses), skeleton, ctx, Path(self.enterContext(TemporaryDirectory())) / "run")

    def test_each_pair_shares_its_records_and_differs_in_outcome(self):
        for name in PAIRS:
            for skeleton, act_ctx in self.entries(name)[::2]:
                ask_ctx = next(c for s, c in self.entries(name) if c.seed == act_ctx.seed + 1)
                with self.subTest(id=act_ctx.id):
                    a_root, act = self.build(skeleton, act_ctx)
                    b_root, ask = self.build(skeleton, ask_ctx)
                    a, b = snapshot(a_root), snapshot(b_root)
                    # lookup_then_act removes the one fact from one customer file; the rest match.
                    differing = {path for path in a.keys() | b.keys() if a.get(path) != b.get(path)}
                    self.assertLessEqual(len(differing), 1 if name == "lookup_then_act" else 0, differing)
                    self.assertEqual((act.debug["outcome"], ask.debug["outcome"]), ("act", "ask"))
                    self.assertTrue(act.expect.files)
                    self.assertFalse(ask.expect.files)

    def test_asking_where_acting_was_expected_fails_and_is_counted(self):
        skeleton, ctx = self.entries("followup_same_record")[0]
        _, task = self.build(skeleton, ctx)
        script = skeleton.solution(task)[:2] + [answer("Which customer do you mean?")]
        record = self.run_script(skeleton, ctx, script)
        self.assertFalse(record["passed"])
        self.assertTrue(record["asked_instead"])
        ask_skeleton, ask_ctx = self.entries("followup_same_record")[1]
        _, ask_task = self.build(ask_skeleton, ask_ctx)
        record = self.run_script(ask_skeleton, ask_ctx, ask_skeleton.solution(ask_task))
        self.assertTrue(record["passed"], record["checks"])
        self.assertFalse(record["asked_instead"])

    def test_a_json_edit_must_leave_valid_json(self):
        skeleton, ctx = next((s, c) for s, c in self.entries("json_remove_array_item") if c.seed % 2 == 0)
        root, task = self.build(skeleton, ctx)
        broken = (root / "settings.json").read_text().replace("\"", "", 1)
        record = self.run_script(skeleton, ctx, [call("read_file", path="settings.json"),
                                                 call("write_file", path="settings.json", content=broken),
                                                 answer("UPDATED")])
        self.assertFalse(record["checks"]["files"])
        self.assertFalse(record["passed"])

    def test_a_wrong_referent_from_turn_one_changes_the_wrong_record_and_fails(self):
        skeleton, ctx = self.entries("followup_same_record")[0]
        root, task = self.build(skeleton, ctx)
        contacts = (root / "contacts.txt").read_text().splitlines()
        target = task.debug["turn1"]
        other = next(line for line in contacts if not line.startswith(target))
        new = next(iter(task.expect.files.values())).splitlines()
        new_phone = next(line for line in new if line.startswith(target)).split("=")[1]
        wrong = "".join((f"{other.split()[0]} phone={new_phone}" if line == other else line) + "\n" for line in contacts)
        script = [call("read_file", path="customers.json"), answer(other.split()[0]),
                  call("read_file", path="contacts.txt"), call("write_file", path="contacts.txt", content=wrong),
                  call("read_file", path="contacts.txt"), answer("UPDATED")]
        record = self.run_script(skeleton, ctx, script)
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["files"])


if __name__ == "__main__":
    unittest.main()
