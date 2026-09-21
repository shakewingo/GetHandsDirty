"""Find files by glob pattern or by regular expression over their contents."""

from pathlib import Path
import re

from .base import ToolErrorCode, ToolExecutionError
from .files import ListFilesTool, _FileTool


class _SearchTool(_FileTool):
    def _directory(self, path: str) -> Path:
        root = self._resolve(path)
        if not root.is_dir():
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, f"Not an existing directory: {root}.")
        return root

    def _files(self, root: Path, pattern: str):
        """Yield regular files under root matching pattern, outside ignored and escaped paths."""
        for item in sorted(root.glob(pattern)):
            if (item.is_file() and not ListFilesTool._IGNORED & set(item.relative_to(root).parts)
                    and (not self.restrict_to_workspace
                         or item.resolve().is_relative_to(self.workspace))):
                yield item


class GlobFilesTool(_SearchTool):
    name = "glob_files"
    description = ("Find files whose path matches a glob pattern such as '**/*.json' or 'src/*.py'. "
                   "Returns sorted paths relative to path. Use grep_text to search file contents.")
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string", "description": "Directory to search; default '.'."},
        "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000},
    }, "required": ["pattern"], "additionalProperties": False}

    def execute(self, pattern: str, path: str = ".", max_matches: int = 200) -> dict:
        root = self._directory(path)
        matches = [item.relative_to(root).as_posix() for item in self._files(root, pattern)]
        return {"path": self._display(root), "matches": matches[:max_matches],
                "truncated": len(matches) > max_matches}


class GrepTextTool(_SearchTool):
    name = "grep_text"
    description = ("Search file contents with a Python regular expression. Returns 'path:line: text' "
                   "matches under a directory. include filters file names by glob, e.g. '*.json'. "
                   "Binary files and common build/cache directories are skipped.")
    parameters = {"type": "object", "properties": {
        "query": {"type": "string"},
        "path": {"type": "string", "description": "Directory to search; default '.'."},
        "include": {"type": ["string", "null"]}, "case_sensitive": {"type": "boolean"},
        "max_matches": {"type": "integer", "minimum": 1, "maximum": 200},
    }, "required": ["query"], "additionalProperties": False}

    def execute(self, query: str, path: str = ".", include: str | None = None,
                case_sensitive: bool = True, max_matches: int = 30) -> dict:
        try:
            regex = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error as error:
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, f"Invalid regular expression: {error}")
        root = self._directory(path)
        matches = []
        for item in self._files(root, f"**/{include or '*'}"):
            data = item.read_bytes()[:self.max_file_bytes]
            if b"\0" in data[:8192]:
                continue
            for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{item.relative_to(root).as_posix()}:{number}: {line[:200]}")
                    if len(matches) > max_matches:
                        return {"path": self._display(root), "matches": matches[:max_matches],
                                "truncated": True}
        return {"path": self._display(root), "matches": matches, "truncated": False}
