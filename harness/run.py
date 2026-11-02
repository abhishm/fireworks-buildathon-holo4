"""Run auto-bench with the harness env swapped in. Same CLI flags as `auto-bench`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import automationbench.scripts.eval as ev  # noqa: E402
from harness_env import HarnessEnv  # noqa: E402

ev.AutomationBenchEnv = HarnessEnv

if __name__ == "__main__":
    ev.main()
