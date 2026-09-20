from dataclasses import replace
import unittest
from unittest.mock import Mock

from agent_from_scratch.agent import Agent
from agent_from_scratch.config import AgentLimits
from agent_from_scratch.llm import LLMResponse, ResponseType
from agent_from_scratch.planning import PlanState, UpdatePlanTool
from agent_from_scratch.tests.test_turn import answer
from agent_from_scratch.tools.base import ToolCall, ToolRegistry


class PlanningTests(unittest.TestCase):
    def test_validation_does_not_destroy_previous_plan(self):
        state = PlanState()
        tool = UpdatePlanTool(state)
        valid = {'steps': [{'text': 'Inspect target', 'status': 'in_progress'}], 'completion': 'Observed expected value'}
        self.assertTrue(tool.invoke(valid).ok)
        before = state.render()
        for steps in ([], [{'text': 'x', 'status': 'pending'}], valid['steps'] * 6,
                      [{'text': 'x' * 161, 'status': 'in_progress'}]):
            self.assertFalse(tool.invoke({**valid, 'steps': steps}).ok)
            self.assertEqual(state.render(), before)
        self.assertTrue(tool.invoke({'steps': [{'text': 'Inspected target', 'status': 'completed'}], 'completion': 'Observed expected value'}).ok)

    def test_latest_only_ephemeral_injection_and_turn_isolation(self):
        model = Mock()
        model.settings.return_value = {}
        model.measure_context.return_value = {'count_method': 'exact', 'prompt_tokens': 100,
            'response_reserve': 2048, 'remaining_tokens': 5852, 'window_tokens': 8000}
        registry = ToolRegistry([])
        agent = Agent(model, registry=registry, limits=replace(AgentLimits(), planning_enabled=True))
        def update(text, status):
            return LLMResponse('assistant', '', ResponseType.tool_call,
                [ToolCall('update_plan', {'steps': [{'text': text, 'status': status}], 'completion': 'Observed correct value'})])
        model.generate.side_effect = [update('inspect', 'in_progress'), update('inspected', 'completed'), answer('done')]
        result = agent.run_turn('Inspect and change and verify.')
        self.assertEqual(result.stop_reason, 'final_response')
        self.assertNotIn('update_plan', registry.schemas())
        self.assertIn('update_plan', result.model_requests[0].tools)
        reminders = [m['content'] for m in result.model_requests[-1].input_messages if m.get('content', '').startswith('[Current task plan]')]
        self.assertEqual(len(reminders), 1)
        self.assertIn('inspected', reminders[0])
        self.assertFalse(any(m.get('content', '').startswith('[Current task plan]') for m in result.messages))
        model.generate.side_effect = [answer('READY')]
        second = agent.run_turn('Reply READY')
        self.assertNotIn('inspected', str(second.model_requests[0].input_messages))

    def test_plan_candidate_measurement_survives_compaction(self):
        from agent_from_scratch.context import ContextState
        from agent_from_scratch.compact import Compactor
        from agent_from_scratch.trace import ModelRequest
        state = ContextState([{'role':'system','content':'rules'}, {'role':'user','content':'old'},
                              {'role':'assistant','content':'large'}, {'role':'user','content':'new'}], 3, 3,
                             plan_text='[Current task plan]\nKeep target and verify')
        model = Mock()
        model.measure_context.return_value = {'count_method':'exact','prompt_tokens':100,'remaining_tokens':3000}
        request = ModelRequest(1, 4)
        self.assertTrue(Compactor(model, AgentLimits())._publish(state, 3, 'summary', {}, request, {'prompt_tokens':1000}))
        sent = model.measure_context.call_args.args[0]
        self.assertEqual(sent[-1]['content'], state.plan_text)
