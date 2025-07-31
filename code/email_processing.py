"""
email_processing.py
───────────────────
Fast PII extraction with a *local* copy of ab-ai/pii_model.
Five pipe-separated columns are returned for each e-mail:

    first_names | last_names | currencies | credit_cards | cc_issuers
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

import warnings
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

# ── silence noisy warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module=r"transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch")

# ── model location & config ─────────────────────────────────────────────────
MODEL_DIR     = Path.home() / "hf_models" / "pii_model"  # adjust if needed
BATCH_SIZE    = 32        # needs ~1.8 GB RAM/worker; lower to 16 on 8 GB boxes
TORCH_THREADS = 1         # intra-op threads per worker (keeps CPU fair)

LABEL_MAP = {
    "FIRSTNAME": "first_names",
    "LASTNAME":  "last_names",
    "AMOUNT":    "currencies",
    "CREDITCARDNUMBER": "credit_cards",
    "CREDITCARDISSUER": "cc_issuers",
}
_EMPTY = {v: "" for v in LABEL_MAP.values()}

_PIPE = None  # lazy singleton per process


def _get_pipe():
    """Load weights/tokenizer once per worker."""
    global _PIPE
    if _PIPE is None:
        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        mdl = AutoModelForTokenClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        _PIPE = pipeline(
            "ner",
            model=mdl,
            tokenizer=tok,
            device=-1,                 # CPU
            batch_size=BATCH_SIZE,
            aggregation_strategy="simple",   # future-proof
        )
    return _PIPE


@lru_cache(maxsize=4096)
def _extract_body(raw: str, limit: int = 2000) -> str:
    """
    Ultra-fast plain-text body extraction:
    take text after the first blank line (RFC-822 separator).
    """
    parts = raw.split("\n\n", 1)
    return (parts[1] if len(parts) > 1 else raw)[:limit]


# ── public batched helper ───────────────────────────────────────────────────
def ner_batch(bodies: List[str]) -> List[Dict[str, str]]:
    """
    Run PII NER on a batch of plain-text snippets.

    Parameters
    ----------
    bodies : list[str]  (len > 0)

    Returns
    -------
    list[dict]  same length, each dict has the 5 pipe-sep columns
    """
    pipe = _get_pipe()
    raw = pipe(bodies)

    # transformers returns dict instead of list when len==1
    if isinstance(raw, dict):
        raw = [raw]

    parsed: List[Dict[str, str]] = []
    for ents in raw:
        buckets = {k: set() for k in LABEL_MAP.values()}
        for e in ents:
            lab = e.get("entity_group") or e.get("entity")
            if lab in LABEL_MAP:
                buckets[LABEL_MAP[lab]].add(e["word"].strip())
        parsed.append({k: "|".join(sorted(v)) for k, v in buckets.items()})

    while len(parsed) < len(bodies):      # rare pipeline failure padding
        parsed.append(_EMPTY.copy())

    return parsed


# ── optional: preload in parent so forked workers share RAM ────────────────
def preload_pipe() -> None:
    _get_pipe()     # just trigger lazy loader
