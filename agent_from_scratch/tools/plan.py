"""A todo list the harness shows back to the model before every request."""

from .base import Tool


class PlanTool(Tool):
    name = "update_plan"
    description = (
        "Create or replace your task plan (a todo list). The harness shows the current plan "
        "back to you before every request. For any task of about 3+ steps, call this FIRST. "
        "Pass the COMPLETE list every time; it replaces the previous one. Keep exactly one "
        "task in_progress and mark a task completed as soon as it is done. Skip planning for a "
        "single trivial step or a purely informational request. Touches no files."
    )
    parameters = {
        "type": "object", "properties": {"plan": {"type": "array", "items": {
            "type": "object", "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            }, "required": ["content", "status"], "additionalProperties": False}}},
        "required": ["plan"], "additionalProperties": False,
    }

    def __init__(self):
        self.plan: list[dict] = []

    def execute(self, plan: list[dict]) -> dict:
        self.plan = plan
        return {"tasks": len(plan), "in_progress": sum(task["status"] == "in_progress" for task in plan)}

    def render(self) -> str:
        """Return the plan as one '[status] content' line per task."""
        return "\n".join(f"[{task['status']}] {task['content']}" for task in self.plan)
