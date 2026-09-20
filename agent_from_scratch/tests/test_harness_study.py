"""Reference outcomes validate the study, not the model's capability."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.evals.harness_study import CASE_IDS, make_case, score_case, summarize
from agent_from_scratch.evals.verify import snapshot
from agent_from_scratch.tools.base import ToolCall
from agent_from_scratch.tools.files import ReadFileTool, WriteFileTool
from agent_from_scratch.trace import TurnResult


class StudyTests(unittest.TestCase):
    def test_reference_outcomes_and_wrong_answers(self):
        for name in CASE_IDS:
            with self.subTest(name=name), TemporaryDirectory() as folder:
                root = Path(folder)
                case = make_case(name, root)
                before = snapshot(root)
                messages = []
                reader = ReadFileTool(root, restrict_to_workspace=True)
                for i, path in enumerate(case['required_reads']):
                    call = ToolCall('read_file', {'path': path}, f'r{i}')
                    messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [call.to_dict()]},
                                     reader.invoke(call.arguments, call.call_id).to_message()])
                if case.get('artifact'):
                    path, content = case['artifact']
                    call = ToolCall('write_file', {'path': path, 'content': content}, 'w')
                    messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [call.to_dict()]},
                                     WriteFileTool(root).invoke(call.arguments, 'w').to_message()])
                result = TurnResult(messages, final_answer=case['answer'], stop_reason='final_response')
                self.assertTrue(score_case(case, result, root, before)['passed'])
                result.final_answer = 'WRONG'
                self.assertFalse(score_case(case, result, root, before)['passed'])

    def test_correct_artifact_without_observation_is_not_success(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            case = make_case('nested', root)
            before = snapshot(root)
            path, content = case['artifact']
            (root / path).write_text(content)
            result = TurnResult([], final_answer=case['answer'], stop_reason='final_response')
            self.assertFalse(score_case(case, result, root, before)['passed'])

    def test_identical_write_still_violates_noop(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            case = make_case('no_op', root)
            before = snapshot(root)
            read = ReadFileTool(root).invoke({'path': 'config.json'}, 'r')
            write = WriteFileTool(root).invoke({'path': 'config.json', 'content': (root/'config.json').read_text()}, 'w')
            result = TurnResult([read.to_message(), write.to_message()], final_answer=case['answer'], stop_reason='final_response')
            self.assertFalse(score_case(case, result, root, before)['passed'])

    def test_summary_charges_failures_and_preserves_unknown_usage(self):
        rows = [{'passed': True, 'elapsed_seconds': 2, 'usage': {'total_tokens': 20}, 'model_requests': 1},
                {'passed': False, 'elapsed_seconds': 8, 'usage': {'total_tokens': None}, 'model_requests': 3}]
        s = summarize(rows)
        self.assertEqual(s['seconds_per_success'], 10)
        self.assertEqual(s['model_requests'], 4)
        self.assertIsNone(s['total_tokens'])
        self.assertIsNone(summarize([rows[1]])['seconds_per_success'])
