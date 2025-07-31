#!/usr/bin/env python3
"""
main.py – orchestrate multi-process PII extraction with smooth, row-level progress and partial CSV flushing
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
import math

import multiprocessing as mp
import polars as pl
from tqdm import tqdm
import pandas as pd

from chunk_worker import process_rows
from email_processing import preload_pipe, _extract_body

# ── config ──────────────────────────────
SLICE_ROWS = 1_000          # maximum slice size for efficient I/O
FLUSH_BATCH = 200         # flush to temp CSV every this many rows
TMP_DIR = Path("tmp")      # directory for partial CSV files
TEST_ROWS = 70               # number of rows to write for testing

# ── helpers ────────────────────

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


def plan_slices(total: int, workers: int) -> List[Tuple[int, int]]:
    """Return (start, n_rows) pairs that cover *total* rows,
    aiming for at most *workers* parallel slices and a max of SLICE_ROWS per slice."""
    ideal = math.ceil(total / max(workers, 1))
    slice_size = min(SLICE_ROWS, ideal)
    return [(i, min(slice_size, total - i)) for i in range(0, total, slice_size)]


def gather(path: Path, slices: List[Tuple[int, int]], workers: int) -> pl.DataFrame:
    """
    Run workers in parallel *and* stream row-level progress.
    Flush every FLUSH_BATCH rows of each completed slice to a partial CSV in TMP_DIR.
    """
    dfs: List[pl.DataFrame] = []
    first_write = False  # tracks header write for partial file

    partial_file = TMP_DIR / f"{path.stem}_partial.csv"
    partial_file_str = str(partial_file)

    with mp.Manager() as manager:
        prog_q = manager.Queue()
        ctx = mp.get_context()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(process_rows, str(path), start, n, prog_q)
                for start, n in slices
            ]

            total_rows = sum(n for _, n in slices)
            remaining = len(futures)

            with tqdm(total=total_rows, desc="rows", unit="row",
                      mininterval=0.5, smoothing=0.2) as bar:
                while remaining:
                    # update progress bar from queue
                    try:
                        while True:
                            bar.update(prog_q.get_nowait())
                    except queue.Empty:
                        pass

                    # handle completed slices
                    done_now = [f for f in futures if f.done()]
                    for f in done_now:
                        df_slice = f.result()
                        offset = 0
                        # flush in FLUSH_BATCH chunks
                        while offset < df_slice.height:
                            chunk = df_slice.slice(offset, FLUSH_BATCH)
                            pd_chunk = chunk.to_pandas()
                            pd_chunk.to_csv(
                                partial_file_str,
                                mode='a',
                                header=not first_write,
                                index=False,
                                encoding='utf-8'
                            )
                            first_write = True
                            offset += FLUSH_BATCH

                        dfs.append(df_slice)
                        futures.remove(f)
                        remaining -= 1

                    time.sleep(0.2)
                # drain any remaining progress
                try:
                    while True:
                        bar.update(prog_q.get_nowait())
                except queue.Empty:
                    pass

    return pl.concat(dfs, how="vertical", rechunk=True)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    preload_pipe()

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="CSV file with the Enron e-mails")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="Number of worker processes (default: logical cores)")
    ap.add_argument("-n", "--num-records", type=int,
                    help="Process only the first N records (for testing)")
    args = ap.parse_args()

    # prepare tmp directory and truncate partial file
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    partial_file = TMP_DIR / f"{args.csv.stem}_partial.csv"
    partial_file.open("w").close()

    total = count_rows(args.csv)
    if args.num_records is not None:
        total = min(total, args.num_records)

    # plan slices based on workers
    slices = plan_slices(total, args.jobs)

    print(f"→ {total:,} rows, {len(slices)} slices, {args.jobs} workers")
    t_start = time.perf_counter()
    df = gather(args.csv, slices, args.jobs)
    elapsed = time.perf_counter() - t_start
    print(f"Done {df.height:,} rows in {elapsed:.1f}s")

    # write test CSV with PII + body for verification
    raw_pd = pd.read_csv(
        args.csv,
        usecols=[0, 1],
        names=["file", "message"],
        header=0,
        nrows=TEST_ROWS,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"  # skip malformed lines
    )
    raw_pd["body"] = raw_pd["message"].apply(lambda x: _extract_body(x))

    pii_pd = df.head(TEST_ROWS).to_pandas()
    merged = pd.merge(
        raw_pd[["file", "body"]],
        pii_pd,
        on="file",
        how="right",
    )
    cols = ["file", "body"] + [c for c in merged.columns if c not in ["file", "body"]]
    merged = merged[cols]

    test_file = args.csv.with_name(f"{args.csv.stem}_test_{TEST_ROWS}.csv")
    merged.to_csv(test_file, index=False)
    print(f"Test output ({TEST_ROWS} rows + body) written to {test_file}")

