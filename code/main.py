#!/usr/bin/env python3
"""
main.py – multi‑process loader & thread builder for the Enron e‑mail CSV.

  • counts data rows safely (even with embedded newlines)
  • splits work across N CPU processes
  • concatenates enriched slices
  • groups into conversation threads (subject + participants)

Run:
    python main.py ./dataset/emails.csv              # use all CPU cores
    python main.py ./dataset/emails.csv -j 12        # force 12 processes
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import polars as pl
from tqdm import tqdm

from chunk_worker import process_rows


# ─── helpers ─────────────────────────────────────────────────────────────────
def count_data_rows(path: Path) -> int:
    """True record count, honouring quoted newlines (header excluded)."""
    csv.field_size_limit(sys.maxsize)
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        rdr = csv.reader(f)
        next(rdr, None)  # discard header
        return sum(1 for _ in rdr)


def plan_work(total: int, n_procs: int) -> list[tuple[int, int]]:
    """Return list of (start_row, n_rows) slices."""
    per = math.ceil(total / n_procs)
    return [
        (i * per, min(per, total - i * per))
        for i in range(n_procs)
        if (i * per) < total
    ]


def gather_slices(path: Path, ranges: list[tuple[int, int]], workers: int) -> pl.DataFrame:
    """Spawn workers -> gather processed frames -> concat."""
    dfs: list[pl.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(process_rows, path, s, n) for s, n in ranges]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="workers"):
            dfs.append(fut.result())
    return pl.concat(dfs, how="vertical", rechunk=True)


def build_threads(df: pl.DataFrame) -> pl.DataFrame:
    """Group e‑mails into conversation threads."""
    return (
        df.with_columns(
            (pl.col("subj_norm") + "||" + pl.col("participants_sig")).alias("thread_key")
        )
        .group_by("thread_key")
        .agg(
            [
                pl.len().alias("email_count"),
                pl.min("date").alias("first_seen"),
                pl.max("date").alias("last_seen"),
                pl.concat_list("file").alias("files"),
            ]
        )
        .sort("email_count", descending=True)
    )


# ─── CLI & orchestration ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Enron multi‑process CSV loader")
    ap.add_argument("csv", type=Path, help="Path to emails.csv")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                    help="Worker processes (default: CPU cores)")
    args = ap.parse_args()

    total_rows = count_data_rows(args.csv)
    slices = plan_work(total_rows, args.jobs)

    print(f"→ {total_rows:,} e‑mails, {len(slices)} slices, {args.jobs} workers")
    df = gather_slices(args.csv, slices, args.jobs)

    threads = build_threads(df)

   
    # after: df = gather_slices(...)
    print(
        df.select([
            (pl.col("first_name") != "").sum().alias("first_name_hits"),
            (pl.col("last_name")  != "").sum().alias("last_name_hits"),
            (pl.col("money")      != "").sum().alias("money_hits"),
            (pl.col("card_number")!= "").sum().alias("card_hits"),
        ])
    )

    # ── sample output ──
    print(f"\nLoaded {df.height:,} rows × {df.width} cols")
    print(f"Grouped into {threads.height:,} conversation threads:\n")
    print(threads.head(10))

    # Save 20 rows from the final dataframe for testing
    df.sample(20).write_csv("../dataset/sample_20rows.csv")
    print("→ Sample of 20 rows saved to dataset/sample_20rows.csv")

if __name__ == "__main__":
    main()
