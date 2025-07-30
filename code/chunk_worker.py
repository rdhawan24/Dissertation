"""
chunk_worker.py – worker for ProcessPoolExecutor.
Loads one ROW slice from the Enron CSV, then extracts e-mail metadata + PII.
Polars-version-agnostic: no Series.apply / Expr.apply required.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
from email_processing import parse_email_raw

def _safe_extract_entities(text_for_ner: str, sender_email: str) -> dict:
    """Import lazily; always return stable keys."""
    try:
        from pii_utils import extract_entities
        return extract_entities(text_for_ner, sender_email)
    except Exception:
        return {"first_name": "", "last_name": "", "money": "", "card_number": ""}

def process_rows(path: str | Path, start_row: int, n_rows: int) -> pl.DataFrame:
    """
    Parameters
    ----------
    start_row : data-row offset (0-based, header excluded)
    n_rows    : number of rows assigned to this slice
    """
    if n_rows <= 0:
        return pl.DataFrame()

    # 1) Load slice
    df = pl.read_csv(
        path,
        has_header=True,
        skip_rows=start_row,        # header already parsed
        n_rows=n_rows,
        new_columns=["file", "message"],
        infer_schema_length=0,
        low_memory=True,
    )

    # 2) Parse + PII extraction (use subject + body to improve recall)
    messages = df["message"].to_list()
    meta_dicts = []
    for raw in messages:
        meta = parse_email_raw(raw)
        text_for_ner = f"{meta.get('subject','')}\n{meta.get('body','')}"
        ner = _safe_extract_entities(text_for_ner, meta.get("sender", ""))
        meta.update(ner)
        meta_dicts.append(meta)

    # 3) Build DF & combine with "file"
    meta_df = pl.from_dicts(meta_dicts)
    return pl.concat([df.drop("message"), meta_df], how="horizontal")

