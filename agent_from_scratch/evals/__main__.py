"""One entry point for evaluation: python -m agent_from_scratch.evals COMMAND [options]

Commands:
  dev       17 small tasks: reading, editing, recovery, stopping   (real model, ~2 min)
  pressure  9-file tasks that push the window past the elision and summary triggers (real model)
  compare   profile one run, or pair two runs by task ID          (no model)

Each command takes --help. Start with evals/README.md.
"""

import importlib
import sys

COMMANDS = {"dev": "run", "pressure": "pressure", "compare": "trajectory"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command = sys.argv.pop(1)
    importlib.import_module(f"{__package__}.{COMMANDS[command]}").main()


if __name__ == "__main__":
    main()
