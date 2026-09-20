from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from agent_from_scratch.config import AgentLimits
from agent_from_scratch.context import ContextState
from agent_from_scratch.elision import elide_old_outputs
from agent_from_scratch.tools.base import ToolCall, ToolResult


def history():
    raw = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'inspect'}]
    for i in range(5):
        call = ToolCall('read_file', {'path': f'{i}.txt'}, str(i))
        result = ToolResult(str(i), 'read_file', True, {'path': f'{i}.txt', 'content': 'long material ' * 200,
                           'version': 'v1', 'next_offset': None, 'eof': True, 'truncated': False})
        raw.extend([{'role': 'assistant', 'content': '', 'tool_calls': [call.to_dict()]}, result.to_message()])
    return raw


def measure(messages, schemas):
    count = len(json.dumps(messages))
    return {'count_method': 'exact', 'prompt_tokens': count, 'window_tokens': 20000,
            'response_reserve': 2000, 'remaining_tokens': 18000-count}


class ElisionTests(unittest.TestCase):
    def setUp(self):
        self.raw = history()
        self.state = ContextState(self.raw, 1, len(self.raw)-1)
        self.llm = Mock()
        self.llm.measure_context.side_effect = measure
        self.limits = replace(AgentLimits(), elision_enabled=True, elision_soft_ratio=0.1)

    def test_raw_envelopes_recent_and_unsent_evidence_preserved(self):
        original = deepcopy(self.raw)
        before = measure(self.state.messages(), {})
        after = elide_old_outputs(self.state, self.llm, {}, self.limits, before)
        self.assertLess(after['prompt_tokens'], before['prompt_tokens'])
        self.assertEqual(self.raw, original)
        self.assertEqual(self.state.messages()[-4:], original[-4:])
        output = json.loads(self.state.messages()[3]['content'])
        self.assertEqual(output['output']['version'], 'v1')
        self.assertEqual(output['call_id'], '0')
        self.assertTrue(output['output']['elided'])
        self.assertEqual(len(self.state.elided), 3)

    def test_errors_partial_results_and_small_outputs_not_elided(self):
        for index in (3, 5, 7):
            row = json.loads(self.raw[index]['content'])
            if index == 3:
                row['ok'] = False
            elif index == 5:
                row['output']['truncated'] = True
                row['output']['next_offset'] = 20
            else:
                row['output']['content'] = 'small'
            self.raw[index]['content'] = json.dumps(row)
        before = self.state.messages()
        elide_old_outputs(self.state, self.llm, {}, self.limits, measure(before, {}))
        self.assertEqual(self.state.messages(), before)

    def test_disabled_below_threshold_and_nonshrinking_are_noops(self):
        budget = measure(self.state.messages(), {})
        for limits in (AgentLimits(), replace(self.limits, elision_soft_ratio=0.99)):
            elide_old_outputs(self.state, self.llm, {}, limits, budget)
            self.assertFalse(self.state.elided)
        self.llm.measure_context.side_effect = None
        self.llm.measure_context.return_value = budget
        elide_old_outputs(self.state, self.llm, {}, self.limits, budget)
        self.assertFalse(self.state.elided)

    def test_malformed_history_is_unchanged(self):
        self.raw[3]['tool_call_id'] = 'unpaired'
        elide_old_outputs(self.state, self.llm, {}, self.limits, measure(self.raw, {}))
        self.assertFalse(self.state.elided)

    def test_agent_elides_without_summary_call_and_traces_actual_input(self):
        from agent_from_scratch.agent import Agent
        from agent_from_scratch.tests.test_turn import answer
        from agent_from_scratch.tools.base import ToolRegistry
        self.llm.settings.return_value = {}
        self.llm.generate.return_value = answer('done')
        original = deepcopy(self.raw[1:])
        result = Agent(self.llm, registry=ToolRegistry([]), limits=self.limits).run_turn('finish', original)
        self.assertEqual(result.stop_reason, 'final_response')
        self.assertEqual(len(result.model_requests), 1)
        self.assertEqual(result.model_requests[0].elided_call_ids, ['0', '1', '2'])
        self.assertEqual(result.messages[1:-2], original)
        self.assertIn('elision_note', result.model_requests[0].input_messages[3]['content'])
