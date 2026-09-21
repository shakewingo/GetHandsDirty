"""Foundation compaction must preserve the optional harness view and protocol."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from agent_from_scratch.agent import Agent
from agent_from_scratch.compact import Compactor
from agent_from_scratch.config import AgentLimits
from agent_from_scratch.context import ContextState, InstructionConfig
from agent_from_scratch.llm import LLM
from agent_from_scratch.planning import PLAN_RULES
from agent_from_scratch.tests.test_turn import answer
from agent_from_scratch.tools.base import ToolCall, ToolRegistry, ToolResult
from agent_from_scratch.trace import ModelRequest


class CompactHarnessTests(unittest.TestCase):
    def test_rule_reload_keeps_enabled_planning_protocol_in_actual_actor_input(self):
        # A reload that bypasses harness instruction assembly loses the plan protocol.
        for enabled in (False, True):
            with self.subTest(planning=enabled), TemporaryDirectory() as directory:
                rules = Path(directory, 'AGENTS.md')
                rules.write_text('Original workspace rule.')
                model = Mock(spec=LLM)
                model.settings.return_value = {}

                def measure(messages, schemas, **kwargs):
                    summarized = any(m.get('content', '').startswith('[Conversation summary:') for m in messages)
                    return {'count_method': 'exact', 'prompt_tokens': 100 if summarized else 2000,
                            'response_reserve': kwargs.get('max_tokens', 2048),
                            'remaining_tokens': 4000, 'window_tokens': 8192}

                def generate(messages, schemas, **kwargs):
                    if not schemas and messages[0]['content'].startswith('Summarize the supplied'):
                        rules.write_text('Revised workspace rule.')
                        return answer('Goal: report 7. Prior evidence retained.')
                    return answer('7')

                model.measure_context.side_effect = measure
                model.generate.side_effect = generate
                agent = Agent(model, registry=ToolRegistry([]),
                              limits=replace(AgentLimits(), planning_enabled=enabled),
                              instruction_config=InstructionConfig(workspace=Path(directory)))
                result = agent.run_turn('Report 7.', [{'role': 'user', 'content': 'old'},
                    {'role': 'assistant', 'content': 'previous evidence'}], compact=True)
                self.assertEqual(result.stop_reason, 'final_response')
                self.assertEqual([r.purpose for r in result.model_requests], ['compact', 'agent'])
                actor = result.model_requests[-1]
                self.assertIn('Revised workspace rule.', actor.input_messages[0]['content'])
                self.assertIn('Original workspace rule.', result.messages[0]['content'])
                self.assertEqual(actor.input_messages[0]['content'].count(PLAN_RULES), int(enabled))
                self.assertEqual('update_plan' in actor.tools, enabled)

    def test_candidate_fit_includes_plan_elision_and_reloaded_rules_atomically(self):
        # Dropping plan_text can admit an oversized candidate; dropping elision can
        # reject a fitting one; dropping instructions republishes outdated rules.
        call = ToolCall('read_file', {'path': 'evidence.txt'}, 'read-1')
        observation = ToolResult('read-1', 'read_file', True,
                                 {'path': 'evidence.txt', 'content': 'evidence' * 500})
        raw = [{'role': 'system', 'content': 'original'},
               {'role': 'user', 'content': 'old'},
               {'role': 'assistant', 'content': 'old history' * 500},
               {'role': 'user', 'content': 'continue'},
               {'role': 'assistant', 'content': '', 'tool_calls': [call.to_dict()]},
               observation.to_message()]
        original = deepcopy(raw)
        stub = ToolResult('read-1', 'read_file', True,
                          {'path': 'evidence.txt', 'content': '', 'elided': True}).to_message()['content']
        state = ContextState(raw, 3, len(raw), elided={'read-1': stub}, plan_text='plan' * 600)
        model = Mock(spec=LLM)

        def measure(messages, schemas, **kwargs):
            count = len(json.dumps(messages))  # Scripted units, not tokenizer claims.
            return {'count_method': 'exact', 'prompt_tokens': count,
                    'response_reserve': 200, 'window_tokens': 1800, 'remaining_tokens': 1600-count}

        model.measure_context.side_effect = measure
        compactor = Compactor(model, AgentLimits(),
                              reload_instructions=lambda: ('revised', {'status': 'loaded'}))
        request = ModelRequest(1, len(raw))
        self.assertFalse(compactor._publish(state, 3, 'summary', {}, request, measure(state.messages(), {})))
        self.assertEqual((state.covered, state.summary, state.instructions), (1, '', None))
        state.plan_text = '[Current task plan]\nVerify observed evidence'
        self.assertTrue(compactor._publish(state, 3, 'summary', {}, request, measure(state.messages(), {})))
        actual = state.messages()
        self.assertEqual(actual[0]['content'], 'revised')
        self.assertEqual(actual[-1]['content'], '[Current task plan]\nVerify observed evidence')
        self.assertEqual(json.loads(actual[-2]['content'])['output']['elided'], True)
        self.assertEqual(request.compact_after, measure(actual, {}))
        self.assertEqual(raw, original)
