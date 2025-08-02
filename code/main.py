# main.py
"""
main.py
────────
Drive parallel PII extraction over an email CSV using ProcessPoolExecutor.
Now supports both HF-style models **and** GLiNER ONNX snapshots.
"""

from __future__ import annotations
import os
import sys
import csv
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, get_context

import polars as pl
from tqdm import tqdm

# −−− helpers ────────────────────────────────────────────────────────────────
def _install_sigint_handler():
    import signal
    def handler(signum, frame):
        # cooperative interrupt flag for worker loop
        global _interrupted
        _interrupted = True
        print("\n[main] SIGINT received; attempting graceful shutdown…", file=sys.stderr)

    signal.signal(signal.SIGINT, handler)


def count_rows(path: str) -> int:
    """
    Count data rows (excluding header) in a CSV.
    Raises csv.field_size_limit to avoid 'field larger than field limit' errors.
    Falls back to naive line counting if the csv module still fails.
    """
    try:
        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int); break
            except OverflowError:
                max_int //= 10

        with open(path, newline="") as f:
            reader = csv.reader(f); next(reader, None)          # skip header
            return sum(1 for _ in reader)

    except csv.Error as e:
        print(f"[main] csv.Error during row-count ({e}); using slow line count.",
              file=sys.stderr)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)


def pick_slice_rows(total: int, workers: int) -> int:
    """Pick slice size so that each worker gets ≈ ≥3 slices (rounded ↑1 000)."""
    target = max(total // (workers * 3), 1)
    return int((target + 999) // 1000) * 1000


def plan_slices(total: int, slice_rows: int) -> list[tuple[int, int]]:
    """Return [(start,count),…] covering [0,total)."""
    return [(i, min(slice_rows, total - i)) for i in range(0, total, slice_rows)]


# −−− GLiNER detection ───────────────────────────────────────────────────────
def _is_gliner_model(model_dir: Path) -> bool:
    """True when directory looks like a GLiNER ONNX export."""
    return (model_dir / "onnx" / "model.onnx").exists()


# −−− main ───────────────────────────────────────────────────────────────────
_interrupted = False

def main():
    global _interrupted

    ap = argparse.ArgumentParser(
        description="Parallel PII extraction over emails CSV (HF or GLiNER)"
    )
    ap.add_argument("csv", help="Path to input CSV file")
    ap.add_argument("-m", "--model-dir", required=True,
                    help="HF checkpoint directory **or** GLiNER ONNX directory")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="Worker processes (default: logical cores)")
    ap.add_argument("-n", "--num-rows", type=int,
                    help="Process only the first N rows (quick smoke-test)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="Force device for HF models (GLiNER decides itself)")
    ap.add_argument("-e", "--encrypt", action="store_true",
                    help="Extract entities *and* rewrite email bodies with FPE")
    ap.add_argument("--partial-dir",
                    help="Write each processed slice to DIR/part_<first>_<last>.csv")
    ap.add_argument("--slice-rows", type=int,
                    help="Manually fix slice size (useful with --partial-dir)")
    args = ap.parse_args()

    model_dir = Path(os.path.expanduser(args.model_dir)).resolve()
    if not model_dir.exists():
        sys.exit(f"ERROR: model directory {model_dir} does not exist.")

    # accept either HF (config.json) or GLiNER (onnx/model.onnx)
    if not ((model_dir / "config.json").exists() or _is_gliner_model(model_dir)):
        sys.exit("ERROR: model dir lacks config.json *and* onnx/model.onnx – "
                 "not a HF checkpoint nor a GLiNER snapshot.")

    # environment → children
    os.environ["PII_MODEL_DIR"] = str(model_dir)
    if args.encrypt:
        os.environ["PII_DO_ENCRYPT"] = "1"

    if args.device == "cpu":
        os.environ["PII_FORCE_CPU"] = "1"
    elif args.device == "cuda":
        os.environ["PII_FORCE_CPU"] = "0"

    _install_sigint_handler()

    # late imports (respect env-vars)
    from email_processing import preload_pipe
    from chunk_worker import process_rows

    if args.encrypt:        # share model RAM via fork
        preload_pipe()

    total_all = count_rows(args.csv)
    total = min(total_all, args.num_rows) if args.num_rows else total_all
    slice_rows = args.slice_rows or pick_slice_rows(total, args.jobs)
    slices = plan_slices(total, slice_rows)

    manager = Manager(); prog_q = manager.Queue()
    dfs: list[pl.DataFrame] = []

    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx)

    start_ts = time.time()
    try:
        futures = [
            pool.submit(process_rows, args.csv, s, n, prog_q, args.partial_dir)
            for s, n in slices
        ]
        pbar = tqdm(total=total, desc="Processing rows", unit="rows")
        completed = 0
        while completed < len(slices):
            if _interrupted: raise KeyboardInterrupt
            try:
                pbar.update(prog_q.get(timeout=0.5))
            except Exception:
                pass
            completed = sum(1 for f in futures if f.done())
        pbar.close()
    except KeyboardInterrupt:
        print("[main] Interrupted – cancelling remaining tasks…", file=sys.stderr)
        for f in futures: f.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    for f in futures:
        if f.done() and not f.cancelled():
            try: dfs.append(f.result())
            except Exception as e:
                print(f"[main] slice failed: {e}", file=sys.stderr)

    if dfs:
        print(pl.concat(dfs, how="vertical"))
    else:
        print("[main] No processed data (all slices failed or cancelled).", file=sys.stderr)

    print(f"Done in {time.time() - start_ts:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
