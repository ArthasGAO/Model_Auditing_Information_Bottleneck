"""Compatibility alias; all planning and training now live in main_at.py.

Prefer: python -u main_at.py --epochs 30 50 --seeds 0 1 2
This alias selects only the two original 25k plans, then forwards CLI options.
"""

import sys
from pathlib import Path

from main_at import main as run_at


def main(argv=None):
    plan_dir = Path(__file__).resolve().parent / "saved_exp_plan/at_plan"
    plans = [
        str(plan_dir / f"CIFAR10_RES18_Extraction_PGD_25k_{epochs}epochs_off.yaml")
        for epochs in (30, 50)
    ]
    args = sys.argv[1:] if argv is None else list(argv)
    print("run_at_25k.py is a compatibility alias. Prefer main_at.py.")
    return run_at(["--plans", *plans, *args])


if __name__ == "__main__":
    main()
