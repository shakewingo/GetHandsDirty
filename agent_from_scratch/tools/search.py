"""Bounded literal search for small-model file localization."""
import os

from .base import ToolErrorCode, ToolExecutionError
from .files import _FileTool, ListFilesTool


class SearchFilesTool(_FileTool):
    name = 'search_files'
    description = ('Search text files for a literal query under a workspace directory. Returns paths, '
                   '1-based line numbers and matching text. Bounded scan; truncated means coverage is incomplete. '
                   'Use read_file on matching paths before editing.')
    parameters = {'type': 'object', 'properties': {
        'query': {'type':'string'}, 'path': {'type':'string'},
        'max_matches': {'type':'integer','minimum':1,'maximum':40}},
        'required':['query'], 'additionalProperties':False}

    def __init__(self, workspace='.'):
        super().__init__(workspace, restrict_to_workspace=True)

    def execute(self, query, path='.', max_matches=20):
        if not query.strip() or len(query) > 200:
            raise ToolExecutionError(ToolErrorCode.INVALID_ARGUMENTS, 'query must contain 1-200 characters.')
        root = self._resolve(path)
        if not root.is_dir():
            raise ToolExecutionError(ToolErrorCode.EXECUTION_ERROR, 'path must be an existing directory.')
        matches, entries, scanned, total_bytes, incomplete = [], 0, 0, 0, False

        def walk(directory):
            nonlocal entries, incomplete
            # Stream directory entries so a very wide tree cannot allocate an unbounded list.
            with os.scandir(directory) as iterator:
                for item in iterator:
                    entries += 1
                    if entries > 2000:
                        incomplete = True
                        return
                    if item.is_symlink() or item.name in ListFilesTool._IGNORED:
                        continue
                    if item.is_dir(follow_symlinks=False):
                        yield from walk(item.path)
                    elif item.is_file(follow_symlinks=False):
                        yield self._resolve(item.path)

        for file in walk(root):
            if scanned >= 200 or total_bytes >= 1024 * 1024:
                incomplete = True
                break
            scanned += 1
            limit = min(65536, 1024 * 1024 - total_bytes)
            try:
                with file.open('rb') as stream:
                    raw = stream.read(limit + 1)
                incomplete |= len(raw) > limit
                raw = raw[:limit]
                total_bytes += len(raw)
                if b'\0' in raw:
                    incomplete = True
                    continue
                text = raw.decode('utf-8')
            except (OSError, UnicodeError):
                incomplete = True
                continue
            for number, line in enumerate(text.splitlines(), 1):
                at = line.find(query)
                if at >= 0:
                    if len(matches) >= max_matches:
                        incomplete = True
                        break
                    start = max(0, at - 40)
                    matches.append({'path': self._display(file), 'line': number,
                                    'text': line[start:start+280]})
            if len(matches) >= max_matches:
                incomplete = True  # Conservative: more matches may exist in later files.
                break
        return {'matches': matches, 'truncated': incomplete, 'files_scanned': scanned,
                'bytes_scanned': total_bytes, 'query': query, 'path': self._display(root)}
