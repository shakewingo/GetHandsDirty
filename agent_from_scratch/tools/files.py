"""General local read/write/edit/list tools, following nanobot's filesystem interface.

Paths may be absolute, home-relative or relative to workspace. Optional confinement
checks resolved paths; it is not an OS sandbox against concurrent path changes.
Historical byte-offset tools live in evals/legacy_files.py.
"""

from collections.abc import Iterable
import difflib
from itertools import islice
import json
import os
from pathlib import Path
import stat
from tempfile import NamedTemporaryFile

from .base import Tool, ToolErrorCode, ToolExecutionError
from ..utils import file_version


class _FileTool(Tool):
    def __init__(self, workspace: str | Path = ".", *, restrict_to_workspace: bool = False,
                 max_file_bytes: int = 100 * 1024 * 1024):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        if not self.workspace.is_dir() or max_file_bytes < 1:
            raise ValueError("Use an existing working directory and a positive file-size limit.")
        self.restrict_to_workspace = restrict_to_workspace
        self.max_file_bytes = max_file_bytes
        self.description += f" Relative paths start at {self.workspace}; absolute and ~ paths are supported."
        if restrict_to_workspace:
            self.description += " Paths must remain inside this workspace."

    def _resolve(self, path: str) -> Path:
        if not path.strip() or "\0" in path:
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "Provide a nonempty file or directory path.")
        target = (self.workspace / Path(path).expanduser()).resolve()
        if self.restrict_to_workspace and not target.is_relative_to(self.workspace):
            raise ToolExecutionError(ToolErrorCode.DENIED, "Path must stay inside the configured workspace.")
        return target

    def _display(self, path: Path) -> str:
        return str(path.relative_to(self.workspace)) if path.is_relative_to(self.workspace) else str(path)

    def _read(self, target: Path) -> tuple[bytes, os.stat_result]:
        if not target.is_file():
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"Not an existing regular file: {target}. Use list_files to find the correct path.")
        with target.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if before.st_size > self.max_file_bytes:
                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, f"File exceeds {self.max_file_bytes} bytes.")
            data = stream.read(self.max_file_bytes + 1)
            if (len(data) > self.max_file_bytes or file_version(os.fstat(stream.fileno())) != file_version(before)
                    or file_version(target.stat()) != file_version(before)):
                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, "File changed during read; read it again.")
        return data, before

    def _replace(self, target: Path, data: bytes, before: os.stat_result | None) -> dict:
        if len(data) > self.max_file_bytes:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, f"Content exceeds {self.max_file_bytes} bytes.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if before is not None:
                temporary.chmod(stat.S_IMODE(before.st_mode))
            current = target.stat() if target.exists() else None
            if (file_version(current) if current else None) != (file_version(before) if before else None):
                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, "File changed before replacement; read it again.")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        result = {"path": self._display(target), "bytes_written": len(data), "changed": True,
                  "created": before is None, "version": file_version(target.stat())}
        # Parse-only checks surface a broken file now instead of on the next run.
        try:
            if target.suffix.lower() == ".json":
                json.loads(data)
            elif target.suffix.lower() == ".py":
                compile(data, str(target), "exec")
        except (ValueError, SyntaxError) as error:
            result["diagnostics"] = f"{type(error).__name__}: {error}"
        return result

    @staticmethod
    def _check_version(before: os.stat_result | None, expected_version: str | None) -> None:
        if expected_version is not None and (before is None or file_version(before) != expected_version):
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     "File version differs from expected_version; read it again before editing.")


def _line_window(lines: Iterable[str], offset: int, limit: int, column: int, max_chars: int) -> dict:
    """Number a bounded range; preserve a cursor even when one line exceeds the output cap."""
    iterator = iter(lines)
    rendered: list[str] = []
    used = 0
    number = 0
    next_offset, next_column = None, 0
    try:
        for number, line in enumerate(iterator, 1):
            if number < offset:
                continue
            if len(rendered) >= limit:
                next_offset = number
                break
            start_column = column if number == offset else 0
            if start_column > len(line):
                raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "column exceeds the selected line length.")
            prefix = f"{number}| "
            room = max_chars - used - len(prefix) - bool(rendered)
            if room <= 0:
                next_offset = number
                break
            part = line[start_column:start_column + room]
            rendered.append(prefix + part)
            used += len(prefix) + len(part) + (1 if len(rendered) > 1 else 0)
            if start_column + len(part) < len(line):
                next_offset, next_column = number, start_column + len(part)
                break
        if not rendered and (number > 0 or offset != 1 or column != 0):
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS,
                                     f"offset {offset} is beyond the available {number} lines.")
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    return {"content": "\n".join(rendered), "offset": offset, "column": column,
            "lines_returned": len(rendered), "next_offset": next_offset, "next_column": next_column,
            "eof": next_offset is None, "truncated": next_offset is not None}


class ReadFileTool(_FileTool):
    name = "read_file"
    description = (
        "Read local text/code, PDF or Office documents. Returns line-numbered content and a version. "
        "offset is a 1-based LINE number, limit is lines (default 2000). "
        "For more content use returned next_offset and next_column as offset and column. "
        "Copy text without the 'N| ' prefix for edit_file. PDF pages accepts '2' or '2-5'. "
        "Document extraction is text-only; images have metadata only, no OCR/visual understanding."
    )
    parameters = {
        "type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1, "description": "1-based line number; default 1."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
            "column": {"type": "integer", "minimum": 0, "description": "Character cursor for continuing a clipped line; default 0."},
            "pages": {"type": ["string", "null"], "description": "PDF page or inclusive range, at most 20 pages."},
            "encoding": {"type": "string", "description": "Text encoding, default utf-8; e.g. utf-16 or latin-1."},
        }, "required": ["path"], "additionalProperties": False,
    }

    def __init__(self, workspace: str | Path = ".", *, max_chars: int = 16000, **kwargs):
        super().__init__(workspace, **kwargs)
        if max_chars < 32:
            raise ValueError("max_chars must be at least 32.")
        self.max_chars = max_chars
        self.description += f" Each result is capped at {max_chars} characters; use the continuation cursor."

    def execute(self, path: str, offset: int = 1, limit: int = 2000, column: int = 0,
                pages: str | None = None, encoding: str = "utf-8") -> dict:
        from .file_documents import document_lines, image_metadata

        target = self._resolve(path)
        raw, before = self._read(target)
        metadata = {"path": self._display(target), "size_bytes": len(raw), "version": file_version(before)}
        suffix = target.suffix.lower()
        if pages is not None and suffix != ".pdf":
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "pages applies only to PDF files.")
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
            return {**metadata, **image_metadata(raw)}
        if suffix in {".pdf", ".docx", ".xlsx", ".pptx"}:
            lines, extra = document_lines(raw, suffix, pages)
            metadata.update(extra)
        else:
            text = raw.decode(encoding)
            if "\0" in text:
                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, "Binary data is not readable text; choose the correct encoding or a document format.")
            lines = iter(text.splitlines())
            metadata.update(format="text", encoding=encoding)
        result = _line_window(lines, offset, limit, column, self.max_chars)
        result["read_status"] = (
            f"Continue with offset={result['next_offset']}, column={result['next_column']}."
            if result["truncated"] else "End of selected text."
        )
        if suffix == ".pdf":
            result["document_eof"] = result["eof"] and metadata.get("next_pages") is None
            if result["eof"] and metadata.get("next_pages"):
                result["read_status"] += f" More pages available: pages={metadata['next_pages']!r}; reset offset=1, column=0."
        return {**metadata, **result}


class WriteFileTool(_FileTool):
    name = "write_file"
    description = (
        "Create or replace a local text file, creating parent directories. Set append=true to append. "
        "Use edit_file for targeted changes. Optionally pass expected_version from read_file. "
        "Unchanged content is left untouched; existing file permissions are preserved. "
        "Results include diagnostics when a written .py or .json file does not parse."
    )
    parameters = {
        "type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
            "append": {"type": "boolean"}, "expected_version": {"type": ["string", "null"]},
            "encoding": {"type": "string"},
        }, "required": ["path", "content"], "additionalProperties": False,
    }

    def execute(self, path: str, content: str, append: bool = False,
                expected_version: str | None = None, encoding: str = "utf-8") -> dict:
        target = self._resolve(path)
        old, before = self._read(target) if target.exists() else (b"", None)
        self._check_version(before, expected_version)
        data = (old if append else b"") + content.encode(encoding)
        if before is not None and data == old:
            return {"path": self._display(target), "bytes_written": 0, "changed": False,
                    "created": False, "version": file_version(before)}
        return self._replace(target, data, before)


class EditFileTool(_FileTool):
    name = "edit_file"
    description = (
        "Replace exact text in one local UTF-8 file; copy old_text from read_file without line numbers. "
        "This edits contents, not the filename. To rename or move a file, use shell with mv. "
        "Ambiguous matches are rejected. Select occurrence (1-based), line_hint, or replace_all; "
        "these are mutually exclusive. expected_replacements and expected_version add optional checks. "
        "Empty old_text creates a missing/empty file. Preserves other bytes and file permissions. "
        "Results include diagnostics when a written .py or .json file does not parse."
    )
    parameters = {
        "type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
            "replace_all": {"type": "boolean"},
            "occurrence": {"type": ["integer", "null"], "minimum": 1},
            "line_hint": {"type": ["integer", "null"], "minimum": 1,
                          "description": "Exact target line from read_file; omit or null when unknown."},
            "expected_replacements": {"type": ["integer", "null"], "minimum": 1},
            "expected_version": {"type": ["string", "null"],
                                 "description": "Optional version returned by read_file; omit or null if unknown."},
        }, "required": ["path", "old_text", "new_text"], "additionalProperties": False,
    }

    def execute(self, path: str, old_text: str, new_text: str, replace_all: bool = False,
                occurrence: int | None = None, line_hint: int | None = None,
                expected_replacements: int | None = None, expected_version: str | None = None) -> dict:
        if sum((replace_all, occurrence is not None, line_hint is not None)) > 1:
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, "Choose only one of replace_all, occurrence or line_hint.")
        target = self._resolve(path)
        raw, before = self._read(target) if target.exists() else (b"", None)
        self._check_version(before, expected_version)
        content = raw.decode("utf-8")
        if not old_text:
            if content:
                raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                    "Empty old_text requires a missing or empty file. "
                    "edit_file changes contents, not filenames; to rename a file, use shell with mv.")
            matches = [0]
        else:
            # Match LF text copied from a line-numbered read against a CRLF file.
            if "\r\n" in content and "\n" not in content.replace("\r\n", ""):
                old_text = old_text.replace("\r\n", "\n").replace("\n", "\r\n")
                new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
            matches, start = [], 0
            while (index := content.find(old_text, start)) != -1:
                matches.append(index)
                start = index + len(old_text)
        locations = [content.count("\n", 0, index) + 1 for index in matches]
        if not matches:
            nearby = difflib.get_close_matches(old_text.splitlines()[0], content.splitlines()[:10000], n=3)
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     "old_text not found in file contents. Read the target region and copy its exact text. "
                                     "To change the filename, use shell with mv; edit_file cannot rename files.",
                                     {"path": self._display(target), "nearby_lines": [line[:300] for line in nearby]})
        if occurrence is not None:
            selected = matches[occurrence - 1:occurrence]
        elif line_hint is not None:
            selected = [index for index, line in zip(matches, locations)
                        if line <= line_hint <= line + old_text.rstrip("\r\n").count("\n")]
        else:
            selected = matches
        if not selected or (len(selected) > 1 and not replace_all):
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     "No unique match. Use more old_text, occurrence, line_hint or replace_all.",
                                     {"match_count": len(matches), "matching_lines": locations[:20]})
        if expected_replacements is not None and len(selected) != expected_replacements:
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR,
                                     f"Expected {expected_replacements} replacements; found {len(selected)}.")
        updated = content
        for index in reversed(selected):
            updated = updated[:index] + new_text + updated[index + len(old_text):]
        data = updated.encode("utf-8")
        if before is not None and data == raw:
            return {"path": self._display(target), "changed": False, "replacements": 0,
                    "version": file_version(before)}
        return {**self._replace(target, data, before), "replacements": len(selected)}


class ListFilesTool(_FileTool):
    name = "list_files"
    description = (
        "List local directory entries with names, types and file sizes. recursive=true explores subdirectories. "
        "Common build/cache directories are omitted unless include_ignored=true. "
        "Does not recurse through symlinks. Use next_offset to continue a truncated listing."
    )
    parameters = {
        "type": "object", "properties": {
            "path": {"type": "string"}, "recursive": {"type": "boolean"},
            "max_entries": {"type": "integer", "minimum": 1, "maximum": 2000},
            "offset": {"type": "integer", "minimum": 0}, "include_ignored": {"type": "boolean"},
        }, "required": ["path"], "additionalProperties": False,
    }
    _IGNORED = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", "htmlcov"}

    def execute(self, path: str, recursive: bool = False, max_entries: int = 200,
                offset: int = 0, include_ignored: bool = False) -> dict:
        directory = self._resolve(path)
        if not directory.is_dir():
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, f"Not an existing directory: {directory}.")

        def walk(root):
            for item in sorted(root.iterdir()):
                if not include_ignored and item.name in self._IGNORED:
                    continue
                link = item.is_symlink()
                is_dir = not link and item.is_dir()
                kind = "symlink" if link else "directory" if is_dir else "file" if item.is_file() else "other"
                yield {"name": str(item.relative_to(directory)), "type": kind,
                       "size_bytes": item.stat().st_size if kind == "file" else None}
                if recursive and is_dir:
                    yield from walk(item)

        entries = list(islice(walk(directory), offset, offset + max_entries + 1))
        truncated = len(entries) > max_entries
        return {"path": self._display(directory), "entries": entries[:max_entries], "truncated": truncated,
                "next_offset": offset + max_entries if truncated else None}
