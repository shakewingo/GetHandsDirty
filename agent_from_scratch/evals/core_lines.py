"""Audit core Python size at committed checkpoints (or the staged INDEX).

Excluded from the runtime itself. See docs/STAGE.md for checkpoint selection.
"""

import argparse
import ast
import hashlib
import io
import json
from pathlib import PurePosixPath
import subprocess
import tokenize


def count_source(source, *, exclude_demo):
    lines = source.splitlines()
    tree = ast.parse(source)
    omitted = set()
    if exclude_demo:
        for node in tree.body:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
                assert node.end_lineno is not None
                omitted.update(range(node.lineno, node.end_lineno + 1))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                first = node.body[0]
                assert first.end_lineno is not None
                docstrings.update(range(first.lineno, first.end_lineno + 1))
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
    code_lines = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in ignored:
            code_lines.update(line for line in range(token.start[0], token.end[0] + 1)
                              if line <= len(lines) and lines[line - 1].strip())
    return {"physical": len(lines) - len(omitted),
            "code": len(code_lines - omitted - docstrings)}


def audit(revision, git="git"):
    def run(*args):
        return subprocess.check_output([git, *args]).decode("utf-8")

    paths = (run("ls-files", "--cached", "-z", "--", "agent_from_scratch") if revision == "INDEX"
             else run("ls-tree", "-r", "--name-only", "-z", revision, "--", "agent_from_scratch"))
    files = {}
    fingerprint = hashlib.sha256()
    for path in sorted(filter(None, paths.split("\0"))):
        relative = PurePosixPath(path).relative_to("agent_from_scratch")
        if relative.suffix != ".py" or not (len(relative.parts) == 1 or relative.parts[0] == "tools"):
            continue
        if relative.name in {"verify.py", "verification.py"}:
            continue
        source = run("show", f":{path}" if revision == "INDEX" else f"{revision}:{path}")
        fingerprint.update(path.encode() + b"\0" + source.encode() + b"\0")
        files[str(relative)] = count_source(source, exclude_demo=relative.name != "agent.py")
    return {"revision": revision, "source_sha256": fingerprint.hexdigest(), "files": files,
            "nonempty_files": sum(count["code"] > 0 for count in files.values()),
            "physical": sum(count["physical"] for count in files.values()),
            "code": sum(count["code"] for count in files.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revisions", nargs="*", default=["HEAD"])
    parser.add_argument("--git", default="git", help="Git executable path")
    args = parser.parse_args()
    print(json.dumps([audit(revision, args.git) for revision in args.revisions], indent=2))
