# chunk_worker.py
"""
Process a CSV slice in a separate process.

Behaviour
=========

* NO -e / --encrypt  →  fast-path
      · No model/NER/FPE work at all
      · Returns:  file, message   (raw body)

* WITH -e / --encrypt →  full pipeline
      · NER + Format-Preserving Encryption (FPE)
      · Returns:  file, message (encrypted) + PII columns
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import polars as pl

_DO_ENCRYPT = os.getenv("PII_DO_ENCRYPT") == "1"

# Import heavy e-mail-processing machinery only when needed
if _DO_ENCRYPT:  # conditional import saves memory & start-up time
    from email_processing import (
        ner_batch,
        encrypt_batch,
        _extract_body,
        BATCH_SIZE,
    )


# ─────────────────────────────────────────────────────────────────────────────
def _stream_batches(
    bodies: List[str], progress_q, step: int
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Mini-batch helper: runs NER + FPE (only called when _DO_ENCRYPT).
    """
    meta_out: List[Dict[str, str]] = []
    body_out: List[str] = []

    for i in range(0, len(bodies), step):
        batch = bodies[i : i + step]

        meta_rows, enc_bodies = encrypt_batch(batch)
        meta_out.extend(meta_rows)
        body_out.extend(enc_bodies)

        if progress_q is not None:
            progress_q.put(len(batch))

    return meta_out, body_out


# ─────────────────────────────────────────────────────────────────────────────
def process_rows(
    path: str | Path,
    start: int,
    n: int,
    progress_q,
) -> pl.DataFrame:
    """
    Worker entry point called by ProcessPoolExecutor.

    Fast-path (no –encrypt) skips all ML/FPE work.
    """
    path = Path(path)

    # read slice; skip_rows accounts for header (+1)
    df = pl.read_csv(
        path,
        has_header=False,
        skip_rows=start + 1,
        n_rows=n,
    )
    df.columns = ["file", "message"]

    # ── FAST-PATH ────────────────────────────────────────────────────────
    if not _DO_ENCRYPT:
        return df.select("file", "message")

    # ── ENCRYPTION PATH ─────────────────────────────────────────────────
    bodies = [_extract_body(m) for m in df["message"]]

    try:
        meta_dicts, new_bodies = _stream_batches(bodies, progress_q, BATCH_SIZE)
    except Exception:
        traceback.print_exc()
        return pl.DataFrame()  # fail-safe

    df = df.with_columns(pl.Series("message", new_bodies))
    left = df.select("file", "message")
    meta_df = pl.DataFrame(meta_dicts)

    return pl.concat([left, meta_df], how="horizontal")
