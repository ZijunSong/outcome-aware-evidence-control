"""Select from a frozen BM25 pool. The candidate list is not retrieved again."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oec.phase1c as phase1c
from oec.frozen_pool import install


def main():
    if "--pool" not in sys.argv:
        raise SystemExit("Pass --pool DIR before the phase3 arguments")
    index = sys.argv.index("--pool")
    pool = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    original = None
    try:
        original, _ = install(pool, phase1c)
        from scripts.run_phase3_selection import main as run
        run()
    finally:
        if original is not None:
            phase1c.fixed_candidates = original


if __name__ == "__main__":
    main()
