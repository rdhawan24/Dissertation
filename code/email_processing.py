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
    "FIRSTNAME": "first_names",
    "LASTNAME": "last_names",
    "CREDIT_CARD": "credit_cards",
    "CC_ISSUER": "cc_issuers",
    "CURRENCY": "currencies",
    "CURRENCYSYMBOL": "currencies",
    "CURRENCYCODE": "currencies",
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


# Assuming LABEL_MAP, _EMPTY_ROW, and _get_pipe() are already defined elsewhere

# Utility functions from your snippet
def fix_apostrophes(entities):
    for ent in entities:
        ent["word"] = re.sub(r"\s*'\s*", "'", ent["word"])
    return entities

def firstname_validate(name: str) -> bool:
    clean = re.sub(r"[.,;:!?]+$", "", name)
    return bool(re.fullmatch(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$", clean))

def lastname_validate(name: str) -> bool:
    clean = re.sub(r"[.,;:!?]+$", "", name)
    return bool(re.fullmatch(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$", clean))

def ner_batch(bodies: List[str]):
    pipe = _get_pipe()
    with torch.inference_mode():
        entities_per_row = pipe(bodies)

    rows = []
    for ents in entities_per_row:
        # Preprocess and validate
        ents = fix_apostrophes(ents)

        buckets = {col: set() for col in LABEL_MAP.values()}

        for ent in ents:
            grp = ent.get("entity_group") or ent.get("entity")
            word = ent.get("word", "").strip()

            if grp == "FIRSTNAME" and not firstname_validate(word):
                continue
            elif grp == "LASTNAME" and not lastname_validate(word):
                continue

            col = LABEL_MAP.get(grp)
            if col:
                buckets[col].add(word)

        # Final structured row
        rows.append({col: "|".join(sorted(words)) for col, words in buckets.items()})

    # Pad with empty rows if needed
    rows.extend([_EMPTY_ROW.copy() for _ in range(len(bodies) - len(rows))])

    return rows

# ── FPE primitives ──────────────────────────────────────────────────────────
#def _enc_name(token: str) -> str:
#    cipher = pyffx.String(_FPE_KEY, alphabet=_ALPHABET, length=len(token))
#    return cipher.encrypt(token)
 
def _enc_name(token: str) -> str:
    core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
    if not core:
        return token

    cipher = pyffx.String(_FPE_KEY, alphabet=_ALPHABET, length=len(core))
    return cipher.encrypt(core)

   
#def _enc_name(token: str) -> str:
#    core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
#    if not core:
#        return token
#    if len(core) not in _cipher_cache:
#        _cipher_cache[len(core)] = pyffx.String(_FPE_KEY, alphabet=_ALPHABET, length=len(core))
#    cipher = _cipher_cache[len(core)]
#    return cipher.encrypt(core)
    


#def _enc_int_str(num: str) -> str:
#    cipher = pyffx.Integer(_FPE_KEY, length=len(num))
#    return str(cipher.encrypt(int(num))).zfill(len(num))


#def encrypt_money(expr: str) -> str:
#    return re.sub(r"\d+", lambda m: _enc_int_str(m.group()), expr)

def _enc_int_str(num: str) -> str:
    """
    Encrypt digits using format-preserving integer encryption and reinsert into original format.
    Preserves currency formatting like "$1,234.56" or "USD 123,456".
    """
    digits = re.sub(r"\D", "", num)
    if not digits:
        return num

    try:
        cipher = pyffx.Integer(_FPE_KEY, length=len(digits))
        enc_digits = str(cipher.encrypt(int(digits))).zfill(len(digits))
    except Exception as e:
        print(f" Encryption failed for {num!r}: {e}")
        return num

    # Reinsert encrypted digits into original structure
    result = []
    idx = 0
    for ch in num:
        if ch.isdigit():
            result.append(enc_digits[idx])
            idx += 1
        else:
            result.append(ch)

    return "".join(result)


def encrypt_money(expr: str) -> str:
    """
    Replace each numeric portion of a money expression (with symbols) using FPE.
    Example: "$1,234.56" → "$8,392.14"
    """
    return re.sub(r"\d[\d,.\s]*\d|\d", lambda m: _enc_int_str(m.group()), expr)


#def encrypt_card(card: str) -> str:
#    digits = re.sub(r"\D", "", card)
#    enc = _enc_int_str(digits)
#    return " ".join(enc[i : i + 4] for i in range(0, len(enc), 4))

def encrypt_card(card: str) -> str:
    """
    Encrypt a credit card number preserving original separators (spaces, hyphens).
    """
    digits = re.sub(r"\D", "", card)
    if not digits:
        return card

    # Encrypt only the digits
    num_cipher = pyffx.String(_FPE_KEY, "0123456789", len(digits))
    enc_digits = num_cipher.encrypt(digits)

    # Reconstruct card with separators preserved
    result = []
    idx = 0
    for ch in card:
        if ch.isdigit():
            result.append(enc_digits[idx])
            idx += 1
        else:
            result.append(ch)
    return "".join(result)


# ── Batch variant (NER + FPE) ───────────────────────────────────────────────
def encrypt_batch(bodies: List[str]):
    meta_rows = ner_batch(bodies)
    enc_bodies: List[str] = []

    for raw_body, meta in zip(bodies, meta_rows):
        text = raw_body

        # 1) Encrypt first and last names if valid
        for bucket, validator in [("first_names", firstname_validate), ("last_names", lastname_validate)]:
            for name in filter(None, meta[bucket].split("|")):
                clean = re.sub(r"[.,;:!?]+$", "", name)
                if validator(clean) and all(c in _ALPHABET for c in clean):
                    enc = _enc_name(clean)
                    # Replace using core part only (removes surrounding punctuations)
                    core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", name)
                    text = re.sub(rf"\b{re.escape(core)}\b", enc, text, flags=re.IGNORECASE)

        # 2) Encrypt credit card numbers from meta
        for cc in filter(None, meta["credit_cards"].split("|")):
            digits = re.sub(r"\D", "", cc)
            if digits:
                enc = encrypt_card(digits)
                text = text.replace(digits, enc)

        # 3) Encrypt monetary values from body directly
        for expr in re.findall(r"[$€£]\s?\d[\d,.,]*", text):
            enc = encrypt_money(expr)
            text = text.replace(expr, enc)

        enc_bodies.append(text)

    return meta_rows, enc_bodies



def preload_pipe():
    """Optionally called by parent so workers share model memory."""
    _get_pipe()
