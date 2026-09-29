import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # see main_victim.py

import torch

from main_victim import run

STAGE = "negative"

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run(STAGE)
