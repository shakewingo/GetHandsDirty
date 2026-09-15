"""Trusted fixture command, outside the agent's writable workspace. No package imports."""

import json
from pathlib import Path
import sys


def main():
    expected = json.loads(Path(sys.argv[1]).read_text())
    root = Path.cwd().resolve()
    target = (root / expected["path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Check target must stay inside the workspace.")
    try:
        with target.open("rb") as stream:
            actual = json.loads(stream.read(8193))
    except (OSError, ValueError) as error:
        print(f"FAIL: cannot read configuration: {error}")
        return 1
    if json.dumps(actual, sort_keys=True) != json.dumps(expected["value"], sort_keys=True):
        print("FAIL: configuration differs; expected " + json.dumps(expected["value"], sort_keys=True))
        return 1
    print("PASS: configuration matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
