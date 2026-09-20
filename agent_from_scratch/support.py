"""Optional observations that help agents recover without declaring task success."""
import ast
import json

from .tools.files import EditFileTool, WriteFileTool


class RepeatMonitor:
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.last = None
        self.count = 0

    def observe(self, result, arguments):
        if not result.ok or result.tool_name not in {'read_file', 'list_files', 'search_files'}:
            self.last, self.count = None, 0
            return None
        key = json.dumps([result.tool_name, arguments, result.output], sort_keys=True, ensure_ascii=False)
        self.count = self.count + 1 if key == self.last else 1
        self.last = key
        if self.count == self.threshold:
            return ('[Runtime feedback] Repeated read-only calls returned unchanged evidence. '
                    'Choose the next necessary action, change the query, or report the actual blocker. '
                    'Do not claim completion unless the requested outcome is observed.')
        return None


def attach_diagnostics(registry, result, arguments):
    """Annotate a completed file operation; diagnostic failure never undoes its success."""
    if not result.ok or result.tool_name not in {'write_file', 'edit_file'}:
        return
    tool = registry.get_tool(result.tool_name)
    if not isinstance(tool, (WriteFileTool, EditFileTool)) or not isinstance(result.output, dict):
        return
    try:
        path = tool._resolve(arguments['path'])
        if path.suffix not in {'.py', '.json'}:
            return
        with path.open('rb') as stream:
            content = stream.read(131073)
        if len(content) > 131072:
            diagnostic = {'status': 'skipped', 'detail': 'Syntax check size limit: 128 KiB.'}
        else:
            try:
                if path.suffix == '.json':
                    json.loads(content)
                else:
                    ast.parse(content, filename=str(path))
                diagnostic = {'status': 'ok', 'detail': 'Syntax only; task correctness is not verified.'}
            except (ValueError, SyntaxError, UnicodeError) as error:
                diagnostic = {'status': 'error', 'detail': str(error)[:500]}
    except Exception as error:
        diagnostic = {'status': 'unavailable', 'detail': str(error)[:300]}
    result.output = {**result.output, 'diagnostics': diagnostic}
