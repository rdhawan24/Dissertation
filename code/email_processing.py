# email_processing.py
"""
PII extraction + Format-Preserving Encryption (FPE).

• Works with either a HuggingFace token-classification checkpoint
  or a GLiNER ONNX snapshot (auto-detected).
• Has per-backend default label lists, overridable via
      PII_GLINER_LABELS   (comma-sep)
      PII_HF_LABELS       (comma-sep; empty ⇒ accept all)
"""

from __future__ import annotations
import os, re, sys, warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────── model directory & type detection ──────────────────────────────
try:
    MODEL_DIR = Path(os.environ["PII_MODEL_DIR"]).expanduser().resolve()
except KeyError:
    sys.exit("ERROR: PII_MODEL_DIR not set (use -m / --model-dir).")

try:
    from gliner import GLiNER
    _HAS_GLINER = True
except ImportError:
    _HAS_GLINER = False

try:
    import torch
except ImportError:
    torch = None

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

_IS_GLINER = (MODEL_DIR / "onnx" / "model.onnx").exists() and _HAS_GLINER
if not _IS_GLINER and not (MODEL_DIR / "config.json").exists():
    sys.exit(
        "ERROR: model dir is neither a HF checkpoint nor a GLiNER export "
        "(missing both config.json and onnx/model.onnx)."
    )

# ────────── runtime config ────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("PII_BATCH_SIZE", "32"))
TORCH_THREADS = 1
_HAS_CUDA = bool(torch and torch.cuda.is_available())
DEVICE = 0 if (not _IS_GLINER and not os.getenv("PII_FORCE_CPU") and _HAS_CUDA) else -1

# ────────── per-backend label defaults ────────────────────────────────────
_GLINER_LABELS_DEFAULT = [
    "person", "organization", "phone number", "address", "passport number",
    "email", "credit card number", "social security number",
    "health insurance id number", "date of birth", "mobile phone number",
    "bank account number", "medication", "cpf", "driver's license number",
    "tax identification number", "medical condition", "identity card number",
    "national id number", "ip address", "email address", "iban",
    "credit card expiration date", "username", "health insurance number",
    "registration number", "student id number", "insurance number",
    "flight number", "landline phone number", "blood type", "cvv",
    "reservation number", "digital signature", "social media handle",
    "license plate number", "cnpj", "postal code", "passport_number",
    "serial number", "vehicle registration number", "credit card brand",
    "fax number", "visa number", "insurance company",
    "identity document number", "transaction number",
    "national health insurance number", "cvc", "birth certificate number",
    "train ticket number", "passport expiration date", "social_security_number"
]
_HF_LABELS_DEFAULT: list[str] = []   # empty ⇒ accept every HF tag

_GLINER_LABELS = [
    l.strip() for l in os.getenv("PII_GLINER_LABELS", ",".join(_GLINER_LABELS_DEFAULT)).split(",")
    if l.strip()
] if _HAS_GLINER else []

_HF_LABELS = {
    l.strip().upper() for l in os.getenv("PII_HF_LABELS", ",".join(_HF_LABELS_DEFAULT)).split(",")
    if l.strip()
}

# ────────── static lexicons ───────────────────────────────────────────────
def _load_set(path: str) -> Set[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return {ln.strip().lower() for ln in f if ln.strip()}
    except FileNotFoundError:
        return set()

FIRSTNAME_SET = _load_set(os.getenv("PII_FIRSTNAME_FILE", "")) or {
    "john", "mary", "michael", "linda", "james", "robert", "patricia", "peter"}
SURNAME_SET = _load_set(os.getenv("PII_SURNAME_FILE", "")) or {
    "smith", "jones", "brown", "williams", "johnson", "patel", "lee", "singh"}

ISO_CURRENCY_CODES = {
    "USD","EUR","INR","GBP","JPY","CNY","AUD","CAD","CHF","SGD","SEK",
    "NOK","RUB","BRL","ZAR","HKD","NZD","KRW","MXN","AED","SAR","IDR",
}

# ────────── unified label-to-column map ───────────────────────────────────
LABEL_MAP: Dict[str, str] = {
    # names (HF)
    "PER_FIRST": "first_names", "FIRST_NAME": "first_names", "FIRSTNAME": "first_names",
    "PER_LAST": "last_names",   "LAST_NAME": "last_names",   "LASTNAME": "last_names",
    # GLiNER person tag → will be split later
    "PERSON": "first_names",

    # currency
    "CURRENCY": "currencies", "CURRENCYCODE": "currencies",

    # credit card number
    "CREDIT_CARD": "credit_cards", "CREDIT_CARD_NUMBER": "credit_cards",
    "CREDIT CARD NUMBER": "credit_cards", "VISA NUMBER": "credit_cards",

    # credit-card issuer / brand
    "CC_ISSUER": "cc_issuers",
    "CREDIT_CARD_BRAND": "cc_issuers", "CREDIT CARD BRAND": "cc_issuers",

    # optional: CVV/CVC
    "CVV": "credit_cards", "CVC": "credit_cards",
}
_EMPTY_ROW = {c: "" for c in LABEL_MAP.values()}

# ────────── regexes & FPE helpers ─────────────────────────────────────────
CARD_RE  = re.compile(r"\b(?:\d[\s\u00A0\-]*?){13,19}\b", re.UNICODE)
MONEY_RE = re.compile(r"[-(]?\s?[€£$¥₹₱₽₩₦฿₫₪₭₲₴₡₵]?\s?\d[\d,\.\s]*[)]?")

import pyffx
_FPE_KEY  = os.getenv("FPE_KEY", "mysupersecretkey").encode()
_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .'-"

@lru_cache(maxsize=128)
def _cipher_int(l: int): return pyffx.Integer(_FPE_KEY, length=l)
def _enc_int_str(num: str) -> str: return str(_cipher_int(len(num)).encrypt(int(num))).zfill(len(num))
def _enc_name(tok: str) -> str:
    core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", tok)
    return tok if not core else pyffx.String(_FPE_KEY, _ALPHABET, length=len(core)).encrypt(core)
def encrypt_money(expr: str) -> str:
    digits = re.sub(r"\D", "", expr); enc = _enc_int_str(digits)
    out, idx = [], 0
    for ch in expr:
        out.append(enc[idx] if ch.isdigit() else ch)
        if ch.isdigit(): idx += 1
    return "".join(out)
def _luhn_ok(num: str) -> bool:
    s = 0
    for i, d in enumerate(map(int, num[::-1])):
        if i % 2: d = d*2 - 9 if d > 4 else d*2
        s += d
    return s % 10 == 0
def _valid_cc_prefix(num: str) -> bool:
    two = int(num[:2]); four = int(num[:4]) if len(num) >= 4 else 0
    return num[0] == "4" or 51 <= two <= 55 or 2221 <= four <= 2720 or two in (34, 37)
def encrypt_card(card: str) -> str:
    digits = re.sub(r"\D", "", card)
    if not (13 <= len(digits) <= 19 and _luhn_ok(digits) and _valid_cc_prefix(digits)):
        return card
    enc = _enc_int_str(digits); out, idx = [], 0
    for ch in card:
        out.append(enc[idx] if ch.isdigit() else ch)
        if ch.isdigit(): idx += 1
    return "".join(out)

# ────────── pipeline builders ─────────────────────────────────────────────
def _build_gliner_pipe():
    model = GLiNER.from_pretrained(
        MODEL_DIR.as_posix(),
        load_onnx_model=True,
        load_tokenizer=True,
        onnx_model_file="onnx/model.onnx",
        local_files_only=True,
    )
    labels = _GLINER_LABELS
    th = float(os.getenv("PII_GLINER_THRESHOLD", "0.3"))
    class _Pipe:
        def __call__(self, texts):
            rows = []
            for txt in texts:
                ents = model.predict_entities(txt, labels, threshold=th)
                print(ents)
                rows.append([
                    {
                        "entity_group": e["label"].upper(),
                        "word": e["text"],
                        "score": float(e.get("score", 1.0)),
                    } for e in ents
                ])
            return rows
    return _Pipe()

def _build_hf_pipe():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    if torch: torch.set_num_threads(TORCH_THREADS)
    ner = pipeline(
        "ner", model=mdl, tokenizer=tok,
        device=DEVICE, batch_size=BATCH_SIZE,
        aggregation_strategy="simple",
    )
    if not _HF_LABELS:
        return ner
    want = _HF_LABELS
    class _Filter:
        def __call__(self, txts):
            raw = ner(txts)
            return [[e for e in r if e["entity_group"].upper() in want] for r in raw]
    return _Filter()

_PIPE = None
def _get_pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = _build_gliner_pipe() if _IS_GLINER else _build_hf_pipe()
    return _PIPE

# ────────── body-extraction (needed by chunk_worker) ──────────────────────
@lru_cache(maxsize=4096)
def get_credit_card_issuer(card_number: str) -> str:
    """
    Identifies the credit card issuer from its number.

    This function checks the card number against known IIN patterns
    to determine the issuing network. It does not validate the card
    number using the Luhn algorithm.

    Args:
        card_number: The credit card number as a string.
                     Can contain spaces or dashes.

    Returns:
        The name of the issuer (e.g., 'Visa', 'Mastercard') or
        'Unknown' if it cannot be identified.
    """
    # Clean the input by removing spaces and dashes
    cleaned_number = re.sub(r'[\s-]', '', str(card_number))

    # Define IIN patterns for major card networks
    # The order is important to handle overlapping prefixes correctly
    issuers = {
        # Mastercard: Starts with 51-55 or 2221-2720
        'Mastercard': r'^(5[1-5]|222[1-9]|22[3-9]|2[3-6]\d|27[0-1]|2720)',
        # Visa: Starts with 4
        'Visa': r'^4',
        # American Express: Starts with 34 or 37
        'American Express': r'^3[47]',
        # Discover: Starts with 6011, 65, or 644-649
        'Discover': r'^(6011|65|64[4-9])',
        # RuPay (India): Starts with 60, 6521, 6522, 508
        'RuPay': r'^(60|6521|6522|508)',
        # Diners Club: Starts with 300-305, 36, or 38-39
        'Diners Club': r'^3(0[0-5]|[689])',
        # JCB: Starts with 3528-3589
        'JCB': r'^35(2[89]|[3-8]\d)',
    }

    for issuer, pattern in issuers.items():
        if re.match(pattern, cleaned_number):
            return issuer

    return 'Unknown'

def _extract_body(raw_email: str) -> str:
    import email
    from email.message import EmailMessage
    """
    Extracts the body from a raw email string.

    This function parses the email and finds the most suitable
    text part for the body, prioritizing plain text over HTML.
    It handles multipart messages and decodes the content.

    Args:
        raw_email: A string containing the full raw email source.

    Returns:
        A string with the extracted email body, or an empty
        string if no suitable body is found.
    """
    # Parse the raw email into a message object
    msg = email.message_from_string(raw_email)
    
    body = ""

    # Check if the email is multipart (contains multiple parts)
    if msg.is_multipart():
        # Walk through all parts of the email
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Look for a 'text/plain' part that is not an attachment
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    # Decode the payload and set it as the body
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='replace')
                    # Prioritize plain text, so we can stop here
                    return body.strip()
                except (AttributeError, TypeError, UnicodeDecodeError):
                    # Handle cases where payload is not a string or has decoding issues
                    continue

        # If no plain text found, fall back to the first HTML part
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/html" and "attachment" not in content_disposition:
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='replace')
                    # Take the first HTML part found
                    return body.strip()
                except (AttributeError, TypeError, UnicodeDecodeError):
                    continue

    # If not multipart, the payload is the body
    else:
        try:
            charset = msg.get_content_charset() or 'utf-8'
            body = msg.get_payload(decode=True).decode(charset, errors='replace')
        except (AttributeError, TypeError, UnicodeDecodeError):
            body = "" # Unable to decode or get payload

    return body.strip()

# ────────── validation helpers ────────────────────────────────────────────
NAME_RE = re.compile(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$")
def _caps_only(w: str, b: str) -> bool: return w.isupper() and any(c.islower() for c in b)
def _fn_ok(name: str) -> bool: return NAME_RE.fullmatch(name) and name.lower() in FIRSTNAME_SET
def _ln_ok(name: str) -> bool: return NAME_RE.fullmatch(name) and name.lower() in SURNAME_SET

# ────────── NER-batch → bucket dicts ──────────────────────────────────────
def ner_batch(bodies: List[str]):
    pipe = _get_pipe()
    ents_rows = pipe(bodies)
    print(ents_rows)
    rows: List[Dict[str, str]] = []
    for ents, body in zip(ents_rows, bodies):
        buckets: Dict[str, set] = {c: set() for c in LABEL_MAP.values()}

        # merge hyphenated first names
        merged, i = [], 0
        while i < len(ents):
            cur = ents[i]
            if (i+2 < len(ents) and cur["entity_group"].startswith("FIRST")
                    and ents[i+1]["word"] == "-" and ents[i+2]["entity_group"].startswith("FIRST")):
                merged.append({**cur, "word": f"{cur['word']}-{ents[i+2]['word']}"})
                i += 3
            else:
                merged.append(cur); i += 1

        for ent in merged:
            grp = ent["entity_group"]
            word = ent["word"].strip()
            score = float(ent.get("score", 1.0))

            # special handling: GLiNER PERSON → split
            if grp == "PERSON":
                parts = word.split()
                if len(parts) == 1:
                    if _fn_ok(parts[0]):
                        buckets["first_names"].add(parts[0])
                elif len(parts) >= 2:
                    if _fn_ok(parts[0]):
                        buckets["first_names"].add(parts[0])
                    if _ln_ok(parts[-1]):
                        buckets["last_names"].add(parts[-1])
                continue

            if grp.startswith(("FIRST", "LAST")) and score < 0.80:
                continue
            if grp.startswith("FIRST") and (_caps_only(word, body) or not _fn_ok(word)):
                continue
            if grp.startswith("LAST") and (_caps_only(word, body) or not _ln_ok(word)):
                continue
            if grp.startswith("CURRENCY") and word.upper() not in ISO_CURRENCY_CODES:
                continue

            col = LABEL_MAP.get(grp)
            if col:
                print(col, word)
                buckets[col].add(word)
                if col == "credit_cards":
                    buckets["cc_issuers"].add(get_credit_card_issuer(word))

        rows.append({c: ",".join(sorted(v)) for c, v in buckets.items()})

    while len(rows) < len(bodies):
        rows.append(_EMPTY_ROW.copy())
    return rows

# ────────── NER + FPE batch processor ─────────────────────────────────────
def encrypt_batch(bodies: List[str]):
    meta_rows = ner_batch(bodies)
    enc_bodies: List[str] = []
    for raw_body, meta in zip(bodies, meta_rows):
        text = raw_body
        for bucket, validator in (("first_names", _fn_ok), ("last_names", _ln_ok)):
            for name in filter(None, meta[bucket].split("|")):
                clean = re.sub(r"[.,;:!?]+$", "", name)
                if not validator(clean):
                    continue
                text = re.sub(rf"\b{re.escape(clean)}\b",
                              lambda m: _enc_name(m.group(0)),
                              text, flags=re.IGNORECASE)

        for m in CARD_RE.finditer(text):
            text = text.replace(m.group(0), encrypt_card(m.group(0)))
        for expr in MONEY_RE.findall(text):
            text = text.replace(expr, encrypt_money(expr))

        enc_bodies.append(text)
    return meta_rows, enc_bodies

# ────────── preload helper (parent before fork) ───────────────────────────
def preload_pipe(): _get_pipe()
