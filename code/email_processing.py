# email_processing.py
"""
Strict local PII extractor + Format-Preserving Encryption helpers.

Only loaded when PII_DO_ENCRYPT=1, so normal runs incur zero model cost.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pyffx                      # pip install pyffx
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Model snapshot dir ──────────────────────────────────────────────────────
try:
    MODEL_DIR = Path(os.environ["PII_MODEL_DIR"]).expanduser().resolve()
except KeyError:
    sys.exit("ERROR: PII_MODEL_DIR not set (use -m/--model-dir).")

if not (MODEL_DIR / "config.json").exists():
    sys.exit(f"ERROR: {MODEL_DIR} is not a valid model snapshot (missing config.json).")

# ── Config knobs ────────────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("PII_BATCH_SIZE", "32"))
DEVICE = 0 if torch.cuda.is_available() and not os.getenv("PII_FORCE_CPU") else -1
TORCH_THREADS = 1

LABEL_MAP: Dict[str, str] = {
    "PER_FIRST": "first_names",
    "PER_LAST": "last_names",
    "CURRENCY": "currencies",
    "CREDIT_CARD": "credit_cards",
    "CC_ISSUER": "cc_issuers",
}
_EMPTY_ROW = {col: "" for col in LABEL_MAP.values()}

_PIPE = None

# ── FPE constants ───────────────────────────────────────────────────────────
_FPE_KEY = os.getenv("FPE_KEY", "mysupersecretkey").encode()
_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .'-"


# ── Pipeline loader ─────────────────────────────────────────────────────────
def _get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

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


# ── Helpers ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4096)
def _extract_body(raw: str) -> str:
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

    while len(rows) < len(bodies):
        rows.append(_EMPTY_ROW.copy())

    return rows


# ── FPE primitives ──────────────────────────────────────────────────────────
def _enc_name(token: str) -> str:
    cipher = pyffx.String(_FPE_KEY, alphabet=_ALPHABET, length=len(token))
    return cipher.encrypt(token)


def _enc_int_str(num: str) -> str:
    cipher = pyffx.Integer(_FPE_KEY, length=len(num))
    return str(cipher.encrypt(int(num))).zfill(len(num))


def encrypt_money(expr: str) -> str:
    return re.sub(r"\d+", lambda m: _enc_int_str(m.group()), expr)


def encrypt_card(card: str) -> str:
    digits = re.sub(r"\D", "", card)
    enc = _enc_int_str(digits)
    return " ".join(enc[i : i + 4] for i in range(0, len(enc), 4))


# ── Batch variant (NER + FPE) ───────────────────────────────────────────────
def encrypt_batch(bodies: List[str]):
    meta_rows = ner_batch(bodies)
    enc_bodies: List[str] = []

    for raw_body, meta in zip(bodies, meta_rows):
        text = raw_body

        # names
        for bucket in ("first_names", "last_names"):
            for name in filter(None, meta[bucket].split("|")):
                text = re.sub(rf"\b{re.escape(name)}\b", _enc_name(name), text)

        # card numbers
        for cc in filter(None, meta["credit_cards"].split("|")):
            digits = re.sub(r"\D", "", cc)
            text = text.replace(digits, encrypt_card(digits))

        # money amounts like "$12,345.67"
        for expr in re.findall(r"[$€£]\s?\d[\d,.,]*", text):
            text = text.replace(expr, encrypt_money(expr))

        enc_bodies.append(text)

    return meta_rows, enc_bodies


def preload_pipe():
    """Optionally called by parent so workers share model memory."""
    _get_pipe()
