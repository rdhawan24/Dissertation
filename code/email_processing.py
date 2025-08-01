# email_processing.py
"""
Strict local PII extractor + Format-Preserving Encryption helpers.

Loaded only when the ``-e`` flag is passed (sets ``PII_DO_ENCRYPT=1``), so
plaintext runs pay **zero** Transformer cost.

Hardening extras beyond basic NER
---------------------------------
• confidence floor on entities (default 0.80)  
• apostrophe fix & hyphenated-name merge  
• lexicon validation for first/last names (small built-ins or user-supplied)  
• ISO-4217 whitelist for currency codes  
• Luhn + issuer-prefix checks for credit-card numbers  
• rich money regex (negatives, parens, commas, dots)
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

import pyffx            # pip install pyffx
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────── Model snapshot dir ────────────────────────────
try:
    MODEL_DIR = Path(os.environ["PII_MODEL_DIR"]).expanduser().resolve()
except KeyError:
    sys.exit("ERROR: PII_MODEL_DIR not set (use -m/--model-dir).")

if not (MODEL_DIR / "config.json").exists():
    sys.exit(f"ERROR: {MODEL_DIR} is not a valid model snapshot (missing config.json).")

# ───────────────────────────────── Config knobs ─────────────────────────────
BATCH_SIZE       = int(os.getenv("PII_BATCH_SIZE", "32"))
DEVICE           = 0 if torch.cuda.is_available() and not os.getenv("PII_FORCE_CPU") else -1
TORCH_THREADS    = 1
NAME_SCORE_FLOOR = float(os.getenv("PII_NAME_MIN_SCORE", "0.80"))

# ──────────────────────────────── Lexicons ─────────────────────────────────
def _load_name_set(fname: str) -> Set[str]:
    try:
        with open(fname, "r", encoding="utf-8") as fp:
            return {ln.strip().lower() for ln in fp if ln.strip()}
    except FileNotFoundError:
        return set()

FIRSTNAME_SET: Set[str] = (
    _load_name_set(os.getenv("PII_FIRSTNAME_FILE", "")) or
    {"john", "mary", "michael", "linda", "james", "robert", "patricia", "peter"}
)
SURNAME_SET: Set[str] = (
    _load_name_set(os.getenv("PII_SURNAME_FILE", "")) or
    {"smith", "jones", "brown", "williams", "johnson", "patel", "lee", "singh"}
)

ISO_CURRENCY_CODES = {
    "USD", "EUR", "INR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF", "SGD", "SEK",
    "NOK", "RUB", "BRL", "ZAR", "HKD", "NZD", "KRW", "MXN", "AED", "SAR", "IDR",
}

# ───────────────────────────────── Label map ───────────────────────────────
LABEL_MAP: Dict[str, str] = {
    # personal names
    "PER_FIRST": "first_names", "FIRST_NAME": "first_names", "FIRSTNAME": "first_names",
    "PER_LAST":  "last_names",  "LAST_NAME":  "last_names",  "LASTNAME":  "last_names",
    # money
    "CURRENCY": "currencies", "CURRENCYSYMBOL": "currencies", "CURRENCYCODE": "currencies",
    # payments
    "CREDIT_CARD": "credit_cards", "CC_ISSUER": "cc_issuers",
}

_EMPTY_ROW = {col: "" for col in LABEL_MAP.values()}
_PIPE      = None               # lazy-loaded NER pipeline

# ────────────────────────────── FPE constants ──────────────────────────────
_FPE_KEY  = os.getenv("FPE_KEY", "mysupersecretkey").encode()
_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .'-"

# ───────────────────────────── Regex helpers ───────────────────────────────
CARD_RE  = re.compile(r"\b(?:\d[ \-]*?){13,19}\b")                # 13–19 digits
MONEY_RE = re.compile(r"[-(]?\s?[€£$¥₹₱₽₩₦฿₫₪₭₲₴₡₵]?\s?\d[\d,\.\s]*[)]?")  # extended symbols]?")

# ──────────────────────────── Pipeline loader ──────────────────────────────
def _get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    tok  = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl  = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    torch.set_num_threads(TORCH_THREADS)

    _PIPE = pipeline(
        "ner", model=mdl, tokenizer=tok,
        device=DEVICE, batch_size=BATCH_SIZE,
        aggregation_strategy="simple",
    )
    return _PIPE

# ────────────────────────────────── Helpers ────────────────────────────────
@lru_cache(maxsize=4096)
def _extract_body(raw: str) -> str:
    """Very simple header-strip + signature trim + HTML tag removal."""
    body = raw.split("\n\n", 1)[-1]
    body = body.split("\n-- \n", 1)[0]
    return re.sub(r"<[^>]+>", " ", body).strip()

def fix_apostrophes(ents):
    for ent in ents:
        ent["word"] = re.sub(r"\s*'\s*", "'", ent["word"])
    return ents

NAME_RE = re.compile(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$")

def _is_all_caps(word: str, body: str) -> bool:
    return word.isupper() and any(c.islower() for c in body)

def firstname_validate(name: str) -> bool:
    n = re.sub(r"[.,;:!?]+$", "", name)
    return bool(NAME_RE.fullmatch(n)) and n.lower() in FIRSTNAME_SET

def lastname_validate(name: str) -> bool:
    n = re.sub(r"[.,;:!?]+$", "", name)
    return bool(NAME_RE.fullmatch(n)) and n.lower() in SURNAME_SET

# ─────────────────────────────── NER batch ─────────────────────────────────
def ner_batch(bodies: List[str]):
    pipe = _get_pipe()
    with torch.inference_mode():
        entities_per_row = pipe(bodies)

    rows = []
    for ents, body in zip(map(fix_apostrophes, entities_per_row), bodies):
        buckets = {c: set() for c in LABEL_MAP.values()}
        i, merged = 0, []
        # merge FIRSTNAME-hyphen-FIRSTNAME tokens (e.g. Mary-Ann)
        while i < len(ents):
            cur = ents[i]
            if (
                i + 2 < len(ents)
                and cur["entity_group"].startswith("FIRST")
                and ents[i + 1]["word"] == "-"
                and ents[i + 2]["entity_group"].startswith("FIRST")
            ):
                merged.append({**cur, "word": f"{cur['word']}-{ents[i+2]['word']}"})
                i += 3
            else:
                merged.append(cur); i += 1

        for ent in merged:
            grp   = ent.get("entity_group") or ent.get("entity")
            word  = ent.get("word", "").strip()
            score = float(ent.get("score", 1.0))

            # confidence gate for names
            if score < NAME_SCORE_FLOOR and LABEL_MAP.get(grp) in {"first_names", "last_names"}:
                continue

            # name sanitisation
            if grp in ("PER_FIRST", "FIRST_NAME", "FIRSTNAME"):
                if _is_all_caps(word, body) or not firstname_validate(word):
                    continue
            elif grp in ("PER_LAST", "LAST_NAME", "LASTNAME"):
                if _is_all_caps(word, body) or not lastname_validate(word):
                    continue
            elif grp == "CURRENCYCODE" and word.upper() not in ISO_CURRENCY_CODES:
                continue

            col = LABEL_MAP.get(grp)
            if col:
                buckets[col].add(word)

        rows.append({c: "|".join(sorted(v)) for c, v in buckets.items()})

    while len(rows) < len(bodies):
        rows.append(_EMPTY_ROW.copy())

    return rows

# ──────────────────────────── FPE primitives ───────────────────────────────
def _enc_name(token: str) -> str:
    core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
    if not core:
        return token
    cipher = pyffx.String(_FPE_KEY, alphabet=_ALPHABET, length=len(core))
    return cipher.encrypt(core)

@lru_cache(maxsize=128)
def _cipher_int(length: int):
    return pyffx.Integer(_FPE_KEY, length=length)

def _enc_int_str(num: str) -> str:
    return str(_cipher_int(len(num)).encrypt(int(num))).zfill(len(num))

def encrypt_money(expr: str) -> str:
    """Encrypt every digit **character** in *expr*, keeping punctuation."""
    digits = re.sub(r"\D", "", expr)
    if not digits:
        return expr
    enc_digits = _enc_int_str(digits)
    out, idx = [], 0
    for ch in expr:
        if ch.isdigit():
            out.append(enc_digits[idx]); idx += 1
        else:
            out.append(ch)
    return "".join(out)

# ───────────────────── Credit-card validation helpers ──────────────────────
def _luhn_ok(num: str) -> bool:
    s = 0
    for i, d in enumerate(map(int, num[::-1])):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        s += d
    return s % 10 == 0

def _valid_cc_prefix(num: str) -> bool:
    two  = int(num[:2])
    four = int(num[:4]) if len(num) >= 4 else 0
    if num[0] == "4":                         # Visa
        return True
    if 51 <= two <= 55 or 2221 <= four <= 2720:  # Mastercard
        return True
    if two in (34, 37):                       # AmEx
        return True
    return False

def encrypt_card(card: str) -> str:
    digits = re.sub(r"\D", "", card)
    if (
        not digits or len(set(digits)) == 1 or
        not (13 <= len(digits) <= 19 and _luhn_ok(digits) and _valid_cc_prefix(digits))
    ):
        return card

    enc_digits = _enc_int_str(digits)
    out, idx = [], 0
    for ch in card:
        if ch.isdigit():
            out.append(enc_digits[idx]); idx += 1
        else:
            out.append(ch)
    return "".join(out)

# ─────────────────────── Batch variant (NER + FPE) ─────────────────────────
def encrypt_batch(bodies: List[str]):
    meta_rows  = ner_batch(bodies)
    enc_bodies = []

    for raw_body, meta in zip(bodies, meta_rows):
        text = raw_body

        # — First & last names —
        for bucket, validator in (
            ("first_names", firstname_validate),
            ("last_names",  lastname_validate),
        ):
            for name in filter(None, meta[bucket].split("|")):
                clean = re.sub(r"[.,;:!?]+$", "", name)
                if not validator(clean):
                    continue
                pattern = re.compile(rf"\b{re.escape(clean)}\b", flags=re.IGNORECASE)
                text = pattern.sub(lambda m: _enc_name(m.group(0)), text)

        # — Credit cards —
        for match in CARD_RE.finditer(text):
            original = match.group(0)
            text = text.replace(original, encrypt_card(original))

        # — Money amounts —
        for expr in MONEY_RE.findall(text):
            text = text.replace(expr, encrypt_money(expr))

        enc_bodies.append(text)

    return meta_rows, enc_bodies

# ───────────────────────────── Warm-up helper ──────────────────────────────
def preload_pipe():
    """Call in parent before spawning workers so they share model RAM."""
    _get_pipe()
