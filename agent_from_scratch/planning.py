"""Small current-turn plan; model assertions guide work but never verify success."""
from copy import deepcopy
from dataclasses import dataclass, field
import json

from .tools.base import Tool, ToolErrorCode, ToolExecutionError


PLAN_RULES = '''
## Task planning
For a task with three or more dependent steps, first use update_plan to keep a short
plan and its observable completion condition. Skip plans for trivial requests.
Keep exactly one step in_progress until all steps are completed; update after actual
progress. The latest plan is shown each turn. A plan is your working hypothesis,
not evidence of completion: verify using actual tool observations before reporting.
'''


@dataclass
class PlanState:
    steps: list[dict] = field(default_factory=list)
    completion: str = ''

    def render(self):
        if not self.steps:
            return ''
        return '[Current task plan]\n' + json.dumps({'steps': self.steps, 'completion': self.completion}, ensure_ascii=False)


class UpdatePlanTool(Tool):
    name = 'update_plan'
    description = ('Replace the current task plan with 1-5 short steps and an observable completion condition. '
                   'Exactly one step in_progress unless all completed. Skip simple tasks. This does not execute steps.')
    parameters = {'type': 'object', 'properties': {
        'steps': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'text': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed']}},
            'required': ['text', 'status'], 'additionalProperties': False}},
        'completion': {'type': 'string'}}, 'required': ['steps', 'completion'], 'additionalProperties': False}

    def __init__(self, state):
        self.state = state

    def execute(self, steps, completion):
        if (not 1 <= len(steps) <= 5 or not completion.strip() or len(completion) > 240
                or any(not s['text'].strip() or len(s['text']) > 160 for s in steps)):
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS,
                                     'Use 1-5 steps of 1-160 characters and a completion condition of 1-240 characters.')
        active = sum(s['status'] == 'in_progress' for s in steps)
        if active != 1 and not (active == 0 and all(s['status'] == 'completed' for s in steps)):
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS,
                                     'Exactly one step must be in_progress unless all steps are completed.')
        self.state.steps, self.state.completion = deepcopy(steps), completion
        return {'updated': True, 'step_count': len(steps)}
