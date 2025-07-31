#!/usr/bin/env python3
"""
main.py – orchestrate multi-process PII extraction with smooth, row-level progress
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import multiprocessing as mp
import polars as pl
from tqdm import tqdm

from chunk_worker import process_rows
from email_processing import preload_pipe

# ── config ────────────────────────────────────────────────────────────────────
SLICE_ROWS = 5_000          # keep large slices for efficient I/O

# ── helpers ───────────────────────────────────────────────────────────────────
def count_rows(p: Path) -> int:
    """Return number of *data* rows (header excluded)."""
    if not p.exists():
        raise FileNotFoundError(p)

    csv.field_size_limit(sys.maxsize)
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            raise ValueError(f"{p} appears empty")
        return sum(1 for _ in rdr)


def plan_slices(total: int) -> List[Tuple[int, int]]:
    """Return (start, n_rows) pairs that cover *total* rows."""
    return [(i, min(SLICE_ROWS, total - i)) for i in range(0, total, SLICE_ROWS)]


def gather(path: Path, slices: List[Tuple[int, int]], workers: int) -> pl.DataFrame:
    """
    Run workers in parallel *and* stream row-level progress back through a
    multiprocessing.Manager queue (which is pickle-safe under 'spawn').
    """
    dfs: List[pl.DataFrame] = []

    with mp.Manager() as manager:
        prog_q = manager.Queue()                      # proxy object → pickle-able

        ctx = mp.get_context()                        # uses the 'spawn' start method
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = [
                pool.submit(process_rows, path, s, n, prog_q)
                for s, n in slices
            ]

            total_rows = sum(n for _, n in slices)
            remaining = len(futs)

            with tqdm(total=total_rows, desc="rows", unit="row",
                      mininterval=0.5, smoothing=0.2) as bar:
                while remaining:
                    # drain queued progress
                    try:
                        while True:
                            bar.update(prog_q.get_nowait())
                    except queue.Empty:
                        pass

                    # collect finished futures
                    done_now = [f for f in futs if f.done()]
                    for f in done_now:
                        dfs.append(f.result())
                        futs.remove(f)
                        remaining -= 1

                    time.sleep(0.2)

                # flush any last progress items
                try:
                    while True:
                        bar.update(prog_q.get_nowait())
                except queue.Empty:
                    pass

    return pl.concat(dfs, how="vertical", rechunk=True)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)          # fork-safe with PyTorch
    preload_pipe()                                    # optional warm-up

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="CSV file with the Enron e-mails")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="Number of worker processes (default: logical cores)")
    args = ap.parse_args()

    total = count_rows(args.csv)
    slices = plan_slices(total)

    print(f"→ {total:,} rows, {len(slices)} slices, {args.jobs} workers")
    t0 = time.perf_counter()
    df = gather(args.csv, slices, args.jobs)
    print(f"Done {df.height:,} rows in {time.perf_counter()-t0:.1f}s")
