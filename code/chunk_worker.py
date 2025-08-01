# chunk_worker.py
"""
Process a CSV slice in a separate process.

Behaviour
=========

* NO -e / --encrypt  →  fast-path
      · No model/NER/FPE work at all
      · Returns:  file, message   (original)

* WITH -e / --encrypt →  full pipeline
      · NER + Format-Preserving Encryption (FPE)
      · Returns:  file, message (original),
                  enc_message (body-encrypted) + PII columns
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
    partial_dir: str | None = None,
) -> pl.DataFrame:
    """
    Worker entry point called by ProcessPoolExecutor.

    Fast-path (no –encrypt) skips all ML/FPE work.
    If *partial_dir* is given, the slice is also written to
    DIR/part_<start>_<end>.csv before returning.
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

    # ── FAST-PATH ───────────────────────────────────────────────────────
    if not _DO_ENCRYPT:
        out_df = df.select("file", "message")

    # ── ENCRYPTION PATH ────────────────────────────────────────────────
    else:
        original_msgs = df["message"].to_list()
        bodies = [_extract_body(m) for m in original_msgs]

        try:
            meta_dicts, enc_bodies = _stream_batches(bodies, progress_q, BATCH_SIZE)
        except Exception:
            traceback.print_exc()
            return pl.DataFrame()  # fail-safe

        # reconstruct full encrypted messages (header untouched, body encrypted)
        enc_msgs_full: List[str] = []
        for raw, body, enc_body in zip(original_msgs, bodies, enc_bodies):
            if body in raw:
                enc_msgs_full.append(raw.replace(body, enc_body, 1))
            else:
                # fallback: put encrypted body alone
                enc_msgs_full.append(enc_body)

        out_df = (
            df.with_columns(pl.Series("enc_message", enc_msgs_full))
              .select("file", "message", "enc_message")
              .hstack(pl.DataFrame(meta_dicts))
        )

    # ──   Incremental write for live inspection   ──────────────────────
    if partial_dir:
        part_dir = Path(partial_dir)
        part_dir.mkdir(parents=True, exist_ok=True)
        part_file = part_dir / f"part_{start:09d}_{start+n-1:09d}.csv"
        out_df.write_csv(part_file, include_header=not part_file.exists())

    return out_df
