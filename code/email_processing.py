# code/email_processing.py
"""
Strict local PII extractor for ab-ai/pii_model.
Loads model ONLY from the directory given in the env-var PII_MODEL_DIR.
If that env-var is unset or the directory is incomplete → hard error.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Resolve and validate model dir ───────────────────────────────────────
try:
    MODEL_DIR = Path(os.environ["PII_MODEL_DIR"]).expanduser().resolve()
except KeyError:
    sys.exit("ERROR: PII_MODEL_DIR not set. Use -m/--model-dir or export PII_MODEL_DIR.")

required_file = MODEL_DIR / "config.json"
if not required_file.exists():
    sys.exit(f"ERROR: {MODEL_DIR} does not look like a valid snapshot "
             "(missing config.json).")

# ── Config knobs ─────────────────────────────────────────────────────────
BATCH_SIZE    = int(os.getenv("PII_BATCH_SIZE", "32"))
TORCH_THREADS = 1
FORCE_CPU     = os.getenv("PII_FORCE_CPU", "0") == "1"
DEVICE        = 0 if torch.cuda.is_available() and not FORCE_CPU else -1

LABEL_MAP: Dict[str, str] = {
    "PER_FIRST":   "first_names",
    "PER_LAST":    "last_names",
    "CURRENCY":    "currencies",
    "CREDIT_CARD": "credit_cards",
    "CC_ISSUER":   "cc_issuers",
}
_EMPTY_ROW = {col: "" for col in LABEL_MAP.values()}

_PIPE = None  # singleton


def _get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    warnings.filterwarnings("ignore", category=FutureWarning)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

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


# ── Helpers ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=4096)
def _extract_body(raw: str) -> str:
    """Strip headers, cut at signature marker, remove rudimentary HTML."""
    parts = raw.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else raw
    body = body.split("\n-- \n", 1)[0]
    body = re.sub(r"<[^>]+>", " ", body)
    return body.strip()


def ner_batch(bodies: List[str]):
    pipe = _get_pipe()
    with torch.inference_mode():
        entities_per_row = pipe(bodies)

    rows = []
    for ents in entities_per_row:
        buckets = {col: set() for col in LABEL_MAP.values()}
        for ent in ents:
            col = LABEL_MAP.get(ent.get("entity_group") or ent.get("entity"))
            if col:
                buckets[col].add(ent["word"].strip())
        rows.append({c: "|".join(sorted(v)) for c, v in buckets.items()})

    # pad if len mismatch (rare but safe)
    while len(rows) < len(bodies):
        rows.append(_EMPTY_ROW.copy())

    return rows


def preload_pipe():
    """Load model once in parent so forkserver children share memory."""
    _get_pipe()
