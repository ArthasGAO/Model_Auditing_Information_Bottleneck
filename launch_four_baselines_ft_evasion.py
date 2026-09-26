"""Windows launcher: retry transient reader locks without changing evaluation settings."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import run_four_baselines_ft_evasion as runner


def retry_file_write(function):
    def wrapped(*args, **kwargs):
        for attempt in range(120):
            try:
                return function(*args, **kwargs)
            except PermissionError:
                if attempt == 119:
                    raise
                time.sleep(0.5)
    return wrapped


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    runner.write_master = retry_file_write(runner.write_master)
    runner.atomic_json = retry_file_write(runner.atomic_json)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, default=runner.ROOT/'saved_logs/at_eval/combined_ft_evasion_2026-09-09')
    known, _ = parser.parse_known_args()
    if '--preflight' not in sys.argv:
        known.output.mkdir(parents=True, exist_ok=True)
        runner.atomic_json(known.output/'launcher.json', dict(
            launcher=Path(__file__).resolve().as_posix(),
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            note='UTF-8 console and up to 60 seconds retry on Windows file locks. Baseline computation unchanged.',
            launched_at=datetime.now().isoformat()))
    runner.main()
