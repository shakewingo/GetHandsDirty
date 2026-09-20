from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_from_scratch.support import RepeatMonitor, attach_diagnostics
from agent_from_scratch.tools.base import ToolRegistry, ToolResult
from agent_from_scratch.tools.files import WriteFileTool
from agent_from_scratch.tools.search import SearchFilesTool


class SupportTests(unittest.TestCase):
    def test_search_matches_bounds_and_symlink_escape(self):
        with TemporaryDirectory() as folder, TemporaryDirectory() as outside:
            root = Path(folder)
            (root/'a.txt').write_text('needle one\nneedle two\n')
            (Path(outside)/'secret.txt').write_text('needle secret')
            (root/'escape').symlink_to(outside)
            tool = SearchFilesTool(root)
            result = tool.invoke({'query': 'needle', 'max_matches': 1})
            self.assertTrue(result.ok)
            self.assertEqual(result.output['matches'][0]['path'], 'a.txt')
            self.assertEqual(result.output['matches'][0]['line'], 1)
            self.assertTrue(result.output['truncated'])
            self.assertFalse(tool.invoke({'query': 'needle', 'path': 'escape'}).ok)
            self.assertNotIn('secret', str(tool.invoke({'query': 'needle'}).output))

    def test_search_total_byte_budget_and_query_validation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'large.txt').write_text('x' * 70000)
            tool = SearchFilesTool(root)
            result = tool.invoke({'query': 'missing'})
            self.assertTrue(result.output['truncated'])
            self.assertFalse(tool.invoke({'query': ''}).ok)
            self.assertFalse(tool.invoke({'query': 'x'*201}).ok)
            self.assertFalse(tool.invoke({'query': ' '*20000+'x'}).ok)

    def test_repeat_reminder_once_and_reset_for_new_evidence(self):
        monitor = RepeatMonitor(3)
        row = ToolResult('id', 'read_file', True, {'path':'a','content':'same'})
        self.assertIsNone(monitor.observe(row, {'path':'a'}))
        self.assertIsNone(monitor.observe(row, {'path':'a'}))
        self.assertIn('unchanged', monitor.observe(row, {'path':'a'}))
        self.assertIsNone(monitor.observe(row, {'path':'a'}))
        self.assertIsNone(monitor.observe(row, {'path':'a','offset':2}))
        row.output['content'] = 'new'
        self.assertIsNone(monitor.observe(row, {'path':'a'}))

    def test_diagnostics_do_not_fail_or_replay_a_completed_write(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            writer = WriteFileTool(root, restrict_to_workspace=True)
            registry = ToolRegistry([writer])
            for path, content, expected in [('a.json', "{'x': 1}", 'error'),
                                             ('b.json', '{"x": 1}', 'ok'),
                                             ('a.py', 'def broken(', 'error'),
                                             ('b.py', 'x = 1\n', 'ok')]:
                args = {'path':path,'content':content}
                result = writer.invoke(args)
                attach_diagnostics(registry, result, args)
                self.assertTrue(result.ok)
                self.assertEqual(result.output['diagnostics']['status'], expected)
                self.assertEqual((root/path).read_text(), content)

    def test_diagnostics_skip_large_files_and_non_code(self):
        with TemporaryDirectory() as folder:
            writer = WriteFileTool(folder)
            registry = ToolRegistry([writer])
            for path, content in [('large.json',' '*140000), ('a.txt','hello')]:
                args={'path':path,'content':content}
                result=writer.invoke(args)
                attach_diagnostics(registry,result,args)
                if path.endswith('.json'):
                    self.assertEqual(result.output['diagnostics']['status'],'skipped')
                else:
                    self.assertNotIn('diagnostics',result.output)

    def test_runtime_reminder_follows_whole_batch_and_diagnostics_reach_model(self):
        from dataclasses import replace
        from unittest.mock import Mock
        from agent_from_scratch.agent import Agent
        from agent_from_scratch.config import AgentLimits
        from agent_from_scratch.context import ContextState
        from agent_from_scratch.llm import LLMResponse, ResponseType
        from agent_from_scratch.tools.base import ToolCall
        from agent_from_scratch.tools.files import ReadFileTool
        from agent_from_scratch.tests.test_turn import answer
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'a.txt').write_text('same')
            registry = ToolRegistry([ReadFileTool(root), WriteFileTool(root)])
            model = Mock()
            model.settings.return_value = {}
            model.measure_context.return_value = {'count_method':'exact','prompt_tokens':100,
                'remaining_tokens':5000,'window_tokens':8000,'response_reserve':2048}
            calls = [ToolCall('read_file', {'path':'a.txt'}, str(i)) for i in range(3)]
            calls.append(ToolCall('write_file', {'path':'b.json','content':"{'x':1}"}, 'w'))
            model.generate.side_effect = [LLMResponse('assistant','',ResponseType.tool_call,calls), answer('done')]
            result = Agent(model,registry=registry,limits=replace(AgentLimits(),
                repeat_reminder_enabled=True,diagnostics_enabled=True)).run_turn('work')
            sent = result.model_requests[-1].input_messages
            self.assertEqual([m['role'] for m in sent[-5:]], ['tool','tool','tool','tool','user'])
            self.assertIn('"status": "error"', sent[-2]['content'])
            self.assertIn('unchanged', sent[-1]['content'])
            # A single recent batch is intentionally protected, even when structurally valid.
            self.assertEqual(ContextState(result.messages,1,len(result.messages)).compact_boundary(),1)
