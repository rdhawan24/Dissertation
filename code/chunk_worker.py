"""
chunk_worker.py – worker for ProcessPoolExecutor.
Loads one ROW slice from the Enron CSV, then extracts e‑mail metadata.

Polars‑version‑agnostic: no Series.apply / Expr.apply required.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
from email_processing import parse_email_raw


def process_rows(path: str | Path, start_row: int, n_rows: int) -> pl.DataFrame:
    """
    Parameters
    ----------
    start_row : data‑row offset (0‑based, header excluded)
    n_rows    : number of rows assigned to this slice
    """
    if n_rows <= 0:
        return pl.DataFrame()

    # 1. Load the slice (fast CSV reader, no email parsing yet)
    df = pl.read_csv(
        path,
        has_header=True,
        skip_rows=start_row,        # header already parsed
        n_rows=n_rows,
        new_columns=["file", "message"],
        infer_schema_length=0,
        low_memory=True,
    )

    # 2. Python‑level metadata extraction (releases GIL per loop iteration)
    messages = df["message"].to_list()           # plain Python list of strings
    meta_dicts = [parse_email_raw(m) for m in messages]
    meta_df = pl.from_dicts(meta_dicts)          # fast → Polars DataFrame

    # 3. Combine original "file" with extracted metadata
    return pl.concat([df.drop("message"), meta_df], how="horizontal")
