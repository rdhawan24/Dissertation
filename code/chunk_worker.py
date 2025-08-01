# chunk_worker.py
"""
chunk_worker.py
────────────────
Process a slice of rows: read CSV, extract bodies, run NER, return a Polars DataFrame.
"""

from __future__ import annotations
import traceback
from pathlib import Path
from typing import List, Dict

import polars as pl

from email_processing import ner_batch, _extract_body, BATCH_SIZE


def _stream_batches(bodies: List[str], progress_q, step: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i in range(0, len(bodies), step):
        batch = bodies[i : i + step]
        out.extend(ner_batch(batch))
        if progress_q is not None:
            progress_q.put(len(batch))
    return out


def process_rows(
    path: str | Path,
    start: int,
    n: int,
    progress_q,
) -> pl.DataFrame:
    """
    Read 'n' rows from CSV at 'path', starting at 'start'.  
    Returns a DataFrame with columns:
      file, first_names, last_names, currencies, credit_cards, cc_issuers
    """
    path = Path(path)
    # skip_rows includes header + 'start' data rows; disable header parsing
    df = pl.read_csv(
        path,
        has_header=False,
        skip_rows=start + 1,
        n_rows=n,
    )
    df.columns = ["file", "message"]

    # extract full bodies (no trimming)
    bodies = [_extract_body(m) for m in df["message"]]

    try:
        meta_dicts = _stream_batches(bodies, progress_q, step=BATCH_SIZE)
    except Exception:
        traceback.print_exc()
        return pl.DataFrame()  # fail-safe: return empty

    meta_df = pl.DataFrame(meta_dicts)
    # horizontally concatenate 'file' with the new PII columns
    return pl.concat([df.select("file"), meta_df], how="horizontal")
