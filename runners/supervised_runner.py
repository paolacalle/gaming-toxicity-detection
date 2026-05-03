#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        "src/eval_training_eval.py",
        "--model-task",
        "supervised",
        "--models",
        "all",
        "--model-configs",
        "all",
    ]
    command.extend(sys.argv[1:])
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
