# email_processing.py
"""
email_processing.py
───────────────────
Fast PII extraction with a *local* copy of ab-ai/pii_model.
Returns five pipe-separated columns per e-mail:
    first_names | last_names | currencies | credit_cards | cc_issuers
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import List, Dict
import os
import warnings

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Map the model’s entity labels to our five output columns
LABEL_MAP: Dict[str, str] = {
    "PER_FIRST": "first_names",
    "PER_LAST": "last_names",
    "CURRENCY": "currencies",
    "CREDIT_CARD": "credit_cards",
    "CC_ISSUER": "cc_issuers",
}

# Empty template for padding
_EMPTY = {col: "" for col in LABEL_MAP.values()}

# Model/pipeline configuration
MODEL_DIR   = Path.home() / "hf_models" / "pii_model"
BATCH_SIZE  = int(os.getenv("PII_BATCH_SIZE", "32"))
TORCH_THREADS = 1
FORCE_CPU   = os.getenv("PII_FORCE_CPU", "0") == "1"
DEVICE      = 0 if (torch.cuda.is_available() and not FORCE_CPU) else -1

_PIPE = None

def _get_pipe():
    global _PIPE
    if _PIPE is None:
        if not MODEL_DIR.exists():
            warnings.warn(f"Model dir {MODEL_DIR!r} not found; make sure you’ve downloaded ab-ai/pii_model there.")
        # load tokenizer & model
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        mdl = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        # restrict PyTorch CPU threads
        torch.set_num_threads(TORCH_THREADS)
        _PIPE = pipeline(
            "ner",
            model=mdl,
            tokenizer=tok,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            aggregation_strategy="simple",
        )
    return _PIPE

@lru_cache(maxsize=4096)
def _extract_body(raw: str, limit: int | None = None) -> str:
    """
    Split off headers (first blank line), then optionally truncate.
    If limit is None, return full body.
    """
    parts = raw.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else raw
    return body if limit is None else body[:limit]

def ner_batch(bodies: List[str]) -> List[Dict[str, str]]:
    """
    Run NER on each snippet and bucket into our five columns.
    Returns one dict per input snippet.
    """
    pipe = _get_pipe()
    all_entities = pipe(bodies)

    parsed: List[Dict[str, str]] = []
    for snippet_entities in all_entities:
        # fresh buckets per snippet
        buckets: Dict[str, set[str]] = {col: set() for col in LABEL_MAP.values()}
        for e in snippet_entities:
            lab = e.get("entity_group") or e.get("entity")
            col = LABEL_MAP.get(lab)
            if col:
                buckets[col].add(e["word"].strip())
        parsed.append({col: "|".join(sorted(vals)) for col, vals in buckets.items()})

    # pad if pipeline returned fewer results (rare)
    while len(parsed) < len(bodies):
        parsed.append(_EMPTY.copy())

    return parsed

# Optional helper to warm up the pipeline in the parent process
def preload_pipe() -> None:
    _get_pipe()
