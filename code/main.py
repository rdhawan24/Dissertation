# main.py
"""
main.py
────────
Drive parallel PII extraction over an email CSV using ProcessPoolExecutor.
Supports GPU auto-detect, smoke-test mode, dynamic slicing, and clean Ctrl+C shutdown.
"""

from __future__ import annotations
import os
import sys
import csv
import time
import argparse
import signal
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager, get_context

import polars as pl
from tqdm import tqdm

# Flag for interruption
_interrupted = False


def _install_sigint_handler():
    def handler(signum, frame):
        global _interrupted          # <-- use global, not nonlocal
        _interrupted = True
        print(
            "\n[main] SIGINT received; attempting graceful shutdown...",
            file=sys.stderr,
        )

    import signal

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
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except csv.Error as e:
        print(
            f"Warning: csv.Error during row count ({e}); falling back to line count. "
            "Embedded newlines may cause inaccuracy.",
            file=sys.stderr,
        )
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for _ in f)
        return max(total_lines - 1, 0)


def pick_slice_rows(total_rows: int, workers: int) -> int:
    """
    Pick slice size so that each worker gets ~3+ slices.
    Rounded up to nearest 1 000 for I/O efficiency.
    """
    target = max(total_rows // (workers * 3), 1)
    return int((target + 999) // 1000) * 1000


def plan_slices(total: int, slice_rows: int) -> list[tuple[int, int]]:
    """Return [(start, count), …] that cover [0,total)."""
    return [(i, min(slice_rows, total - i)) for i in range(0, total, slice_rows)]


def main():
    global _interrupted
    ap = argparse.ArgumentParser(description="Parallel PII extraction over emails CSV")
    ap.add_argument("csv", help="Path to input CSV file")
    ap.add_argument(
        "-m", "--model-dir",
        required=True,
        help="Path to ab-ai/pii_model snapshot (required)"
    )
    ap.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (default: logical cores)",
    )
    ap.add_argument(
        "-n", "--num-rows",
        type=int,
        help="Process only the first N rows (for quick tests)",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override device choice (auto = GPU if available)",
    )
    ap.add_argument("-e", "--encrypt",
                    action="store_true",
                    help="Extract entities *and* rewrite the e-mail bodies "
                         "using format-preserving encryption")
    ap.add_argument(
        "--partial-dir",
        help="If set, each processed slice is immediately written to this "
             "directory as part_<firstrow>_<lastrow>.csv for live inspection.",
    )
    ap.add_argument(
        "--slice-rows",
        type=int,
        help="Override automatic slice sizing; useful together with --partial-dir "
             "to get smaller, more frequent chunks.",
    )
    args = ap.parse_args()

    # Export model path for email_processing and validate
    model_dir_resolved = os.path.expanduser(args.model_dir)
    if not os.path.exists(model_dir_resolved):
        sys.exit(f"ERROR: model directory {model_dir_resolved} does not exist.")
    os.environ["PII_MODEL_DIR"] = model_dir_resolved

    if args.encrypt:
        os.environ["PII_DO_ENCRYPT"] = "1"

    # install interrupt handler
    _install_sigint_handler()

    # Force CPU if requested
    if args.device == "cpu":
        os.environ["PII_FORCE_CPU"] = "1"
    elif args.device == "cuda":
        os.environ["PII_FORCE_CPU"] = "0"

    # ── defer these imports until env-vars are ready ───────────────────
    from email_processing import preload_pipe
    from chunk_worker import process_rows

    if args.encrypt:
        preload_pipe()

    total_all = count_rows(args.csv)
    total = min(total_all, args.num_rows) if args.num_rows else total_all

    # determine slice size
    slice_size = (
        args.slice_rows
        if args.slice_rows                 # honour manual override
        else pick_slice_rows(total, args.jobs)
    )
    slices = plan_slices(total, slice_size)

    manager = Manager()
    prog_q = manager.Queue()

    workers = args.jobs
    ctx = get_context("spawn")  # safe for PyTorch
    dfs: list[pl.DataFrame] = []

    start_ts = time.time()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    try:
        futures = [
            pool.submit(process_rows, args.csv, start, count, prog_q, args.partial_dir)
            for start, count in slices
        ]
        pbar = tqdm(total=total, desc="Processing rows", unit="rows")
        try:
            completed = 0
            while completed < len(slices):
                if _interrupted:
                    raise KeyboardInterrupt
                try:
                    n = prog_q.get(timeout=0.5)
                    pbar.update(n)
                except Exception:
                    # timeout or empty; just loop to check interrupt
                    pass
                completed = sum(1 for f in futures if f.done())
        except KeyboardInterrupt:
            print("[main] Interrupt detected; cancelling remaining tasks...", file=sys.stderr)
            # cancel pending futures
            for f in futures:
                if not f.done():
                    f.cancel()
            # attempt to terminate worker processes aggressively
            for p in getattr(pool, "_processes", {}).values():
                try:
                    p.terminate()
                except Exception:
                    pass
            # shutdown executor without waiting
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            pbar.close()

        # collect results from completed futures
        for f in futures:
            if f.cancelled():
                continue
            if f.done():
                try:
                    dfs.append(f.result())
                except Exception as e:
                    print(f"Warning: slice failed with {e}", file=sys.stderr)
    finally:
        # ensure executor is shut down if not already
        try:
            pool.shutdown(wait=False)
        except Exception:
            pass

    if dfs:
        result_df = pl.concat(dfs, how="vertical")
        print(result_df)
    else:
        if _interrupted:
            print("Processing was interrupted by user; partial results may exist.", file=sys.stderr)
        else:
            print("No data processed; all slices failed.", file=sys.stderr)

    elapsed = time.time() - start_ts
    print(f"Done in {elapsed:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
