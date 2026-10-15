"""One entry point for evaluation: python -m agent_from_scratch.evals COMMAND [options]

Commands:
  bench     generated benchmark: --split train (320 tasks), val (64) or test (140, needs --final)
  pressure  9-file tasks that push the window past the elision and summary triggers (real model)
  freeze    write/update the Stage 8 benchmark's frozen manifest (no model)
  aggregate combine k sampled bench runs into pass@1 / pass^k / pass@k      (no model)
  dataset   generate train/test fixtures and their manifest; --check runs scripted gates (no model)
  export    turn verifier-passed train rollouts into Stage 9 SFT examples     (no model)
  public    turn rows of a public tool-use dataset into SFT examples           (no model)
  mix       combine our SFT examples with public ones into one training set   (no model)
  compare   profile one run, or pair two runs by task ID          (no model)
  train     train a LoRA adapter on an exported SFT dataset (needs torch, transformers, peft)
  replay    continue each SFT prompt greedily on a vLLM server and compare with its target

Each command takes --help. Start with evals/README.md.
"""

import importlib
import sys

COMMANDS = {"pressure": "pressure", "bench": "bench.run",
           "freeze": "bench.manifest",
           "aggregate": "bench.aggregate", "compare": "trajectory", "dataset": "bench.dataset",
           "export": "sft.export", "public": "sft.public", "mix": "sft.mix",
           "train": "sft.train", "replay": "sft.replay"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command = sys.argv.pop(1)
    importlib.import_module(f"{__package__}.{COMMANDS[command]}").main()


if __name__ == "__main__":
    main()
