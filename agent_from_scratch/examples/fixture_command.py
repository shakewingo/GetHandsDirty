"""Trusted local script: inspect JSON or check the demo's requested configuration."""

import json
from pathlib import Path
import sys


def main():
    try:
        with Path("config.json").open("rb") as stream:
            raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("config.json is too large")
        config = json.loads(raw)
        if sys.argv[1] == "inspect":
            print(json.dumps(config, ensure_ascii=False))
        elif sys.argv[1] == "check":
            if config != {"output": "report.txt", "retries": 3}:
                raise ValueError('Expected output="report.txt" and retries=3, with no other fields changed.')
            print("PASS: config.json has the requested output and preserves retries.")
        else:
            raise ValueError("Expected inspect or check mode")
    except (OSError, ValueError, IndexError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
