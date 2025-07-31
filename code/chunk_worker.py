"""
chunk_worker.py – worker process code.

Each worker:
1. Reads *n_rows* rows starting at *start_row* from the CSV.
2. Extracts plain-text e-mail bodies (fast).
3. Runs batched PII NER, emitting progress to *progress_q* every batch.
4. Returns a Polars DataFrame with the original path plus 5 PII columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl

from email_processing import ner_batch, _extract_body, BATCH_SIZE

BODY_LIMIT = 2_000   # trim huge messages


def _stream_batches(bodies: List[str], progress_q, step: int) -> List[dict]:
    """NER in *step*-sized chunks; stream progress after each chunk."""
    out: List[dict] = []
    for i in range(0, len(bodies), step):
        batch = bodies[i : i + step]
        out.extend(ner_batch(batch))
        if progress_q is not None:
            progress_q.put(len(batch))      # rows processed
    return out


def process_rows(
    path: str | Path,
    start_row: int,
    n_rows: int,
    progress_q: Optional[object] = None,    # manager.Queue proxy
) -> pl.DataFrame:
    """Worker entry point – must be pickle-able under the 'spawn' context."""
    if n_rows <= 0:
        return pl.DataFrame()

    # 1. Load slice
    df = pl.read_csv(
        path,
        has_header=True,
        skip_rows=start_row,
        n_rows=n_rows,
        new_columns=["file", "message"],
        infer_schema_length=0,
        low_memory=True,
    )

    # 2. Extract bodies
    bodies: List[str] = [_extract_body(m, BODY_LIMIT) for m in df["message"]]

    # 3. Batched NER with incremental progress
    meta_dicts = _stream_batches(bodies, progress_q, step=BATCH_SIZE)

    # 4. Merge & return
    meta_df = pl.from_dicts(meta_dicts)
    return pl.concat([df.select("file"), meta_df], how="horizontal")
