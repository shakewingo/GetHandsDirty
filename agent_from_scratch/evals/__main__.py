"""One entry point for evaluation: python -m agent_from_scratch.evals COMMAND [options]

Commands:
  bench     Stage 8 generated benchmark: --split dev (64 tasks) or --split test (60, needs --final)
  pressure  9-file tasks that push the window past the elision and summary triggers (real model)
  freeze    write/update the Stage 8 benchmark's frozen manifest (no model)
  aggregate combine k sampled bench runs into pass@1 / pass^k / pass@k      (no model)
  compare   profile one run, or pair two runs by task ID          (no model)

Each command takes --help. Start with evals/README.md.
"""

import importlib
import sys

COMMANDS = {"pressure": "pressure", "bench": "bench.run",
           "freeze": "bench.manifest",
           "aggregate": "bench.aggregate", "compare": "trajectory"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command = sys.argv.pop(1)
    importlib.import_module(f"{__package__}.{COMMANDS[command]}").main()


if __name__ == "__main__":
    main()
