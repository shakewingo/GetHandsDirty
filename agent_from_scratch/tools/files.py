"""Small UTF-8 file tools for a single-writer scratch workspace.

Resolved paths prevent ordinary traversal and symlink escapes. This is not an
OS sandbox against another process changing paths during a call.
"""

from codecs import getincrementaldecoder
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from os import fstat

from .base import Tool
from ..utils import resolve_path, file_version


class ListFilesTool(Tool):
    name = "list_files"
    description = "List immediate workspace directory entries. Output may be truncated."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path; use '.' for root."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | Path, max_entries: int = 100):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir() or max_entries < 1:
            raise ValueError("Workspace must be a directory and max_entries must be positive.")
        self.max_entries = max_entries

    def execute(self, path: str) -> dict:
        directory = resolve_path(self.workspace, path)
        entries = list(islice(directory.iterdir(), self.max_entries + 1))
        # Do not follow symlinks to inspect targets while listing their names.
        return {
            "entries": sorted(entry.name for entry in entries[:self.max_entries]),
            "truncated": len(entries) > self.max_entries,
        }


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a bounded UTF-8 file chunk. offset and chunk_size are bytes. "
        "For a full-file read, start at 0 and continue using next_offset until eof. "
        "For a partial read, stop after the requested range."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0,
                       "description": "Byte offset; omit to start at 0, then use returned next_offset."},
            "chunk_size": {"type": "integer", "minimum": 4,
                           "description": "Maximum bytes to read."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | Path, max_bytes: int = 8192):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir() or max_bytes < 4:
            raise ValueError("Workspace must be a directory and max_bytes must be at least 4.")
        self.max_bytes = max_bytes
        self.parameters = {**self.parameters, "properties": {
            **self.parameters["properties"],
            "chunk_size": {**self.parameters["properties"]["chunk_size"], "maximum": max_bytes,
                           "default": max_bytes,
                           "description": f"Maximum bytes to read; defaults to {max_bytes} bytes."},
        }}

    def execute(self, path: str, offset: int = 0, chunk_size: int | None = None) -> dict:
        target = resolve_path(self.workspace, path)
        if not target.is_file():
            raise ValueError(f"Not an existing regular file: {target.relative_to(self.workspace)}. "
                             "Use list_files to inspect the workspace.")
        size = self.max_bytes if chunk_size is None else chunk_size
        with target.open("rb") as stream:
            before = fstat(stream.fileno())
            version = file_version(before)
            file_size = before.st_size
            if offset > file_size:
                raise ValueError(f"offset exceeds file size {file_size}; expected 0..{file_size}. "
                                 "Use the previous next_offset.")
            stream.seek(offset)
            raw = stream.read(size + 1)
            if file_version(fstat(stream.fileno())) != version or file_version(target.stat()) != version:
                raise ValueError("File changed during the read; this chunk cannot be used.")
        eof = len(raw) <= size
        decoder = getincrementaldecoder("utf-8")()
        content = decoder.decode(raw[:size], final=eof)
        # Leave an incomplete UTF-8 character for the next read, without data loss.
        consumed = len(raw[:size]) - len(decoder.getstate()[0])
        return {"path": str(target.relative_to(self.workspace)), "content": content,
                "offset": offset, "next_offset": None if eof else offset + consumed, "eof": eof,
                "size_bytes": file_size, "version": version}


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or replace a small UTF-8 workspace file, creating parent directories."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: str | Path, max_bytes: int = 8192):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir() or max_bytes < 1:
            raise ValueError("Workspace must be a directory and max_bytes must be positive.")
        self.max_bytes = max_bytes

    def execute(self, path: str, content: str) -> dict:
        target = resolve_path(self.workspace, path)
        data = content.encode("utf-8")
        if len(data) > self.max_bytes:
            raise ValueError(f"Content exceeds the {self.max_bytes}-byte write limit.")
        if target.exists() and not target.is_file():
            raise ValueError("Path must refer to a regular file.")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after writing succeeds; do not truncate an existing file.
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"path": str(target.relative_to(self.workspace)), "bytes_written": len(data)}
