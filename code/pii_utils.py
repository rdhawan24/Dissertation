# pii_utils.py
"""
PII utilities:
- Names: Hugging Face `ab-ai/pii_model` (token-classification) → first_name, last_name
          Falls back to greetings/sender-email heuristics if HF unavailable.
- Cards: Microsoft Presidio CreditCardRecognizer (with Luhn); regex+Luhn fallback.
- Money: Presidio PatternRecognizer for currency; regex fallback.

Auth:
- Set HF token via env:  HF_TOKEN or HUGGINGFACE_HUB_TOKEN
"""

from __future__ import annotations
import os
import re
from typing import Optional, Tuple

# ── Lazy singletons for optional deps ─────────────────────────────────────────
_hf_pipe = None
def _get_hf_pipe():
    """Load HF token-classification pipeline once per process (CPU)."""
    global _hf_pipe
    if _hf_pipe is not None:
        return _hf_pipe
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        auth_kw = {}
        if token:
            # new param name (Transformers >=4.40)
            auth_kw["token"] = token
            # backward compat
            auth_kw["use_auth_token"] = token

        model_id = "ab-ai/pii_model"
        tok = AutoTokenizer.from_pretrained(model_id, **auth_kw)
        mdl = AutoModelForTokenClassification.from_pretrained(model_id, **auth_kw)
        _hf_pipe = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=-1,   # CPU
        )
    except Exception:
        # Optional open fallback if needed (leave None to skip if unavailable)
        try:
            from transformers import pipeline
            _hf_pipe = pipeline(
                "token-classification",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1,
            )
        except Exception:
            _hf_pipe = None
    return _hf_pipe

_currency_recognizer = None
_cc_recognizer = None
def _get_presidio_recognizers():
    """Create Presidio recognizers; returns (currency_recognizer, creditcard_recognizer)."""
    global _currency_recognizer, _cc_recognizer
    if _currency_recognizer is not None and _cc_recognizer is not None:
        return _currency_recognizer, _cc_recognizer
    try:
        from presidio_analyzer import Pattern, PatternRecognizer
        from presidio_analyzer.predefined_recognizers import CreditCardRecognizer

        currency_patterns = [
            # $1,234.56 / £200 / € 3,000.00
            Pattern(name="CURRENCY_SYMBOL",
                    regex=r"(?:[$£€]\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)(?!\S)", score=0.5),
            # USD 1,000.00 / EUR 250 / GBP 75.25 / INR 10,000
            Pattern(name="CURRENCY_ISO",
                    regex=r"(?:USD|EUR|GBP|INR|CAD|AUD)\s?-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?", score=0.5),
        ]
        _currency_recognizer = PatternRecognizer(
            supported_entity="CURRENCY", patterns=currency_patterns
        )
        _cc_recognizer = CreditCardRecognizer()  # includes Luhn check
    except Exception:
        _currency_recognizer = None
        _cc_recognizer = None
    return _currency_recognizer, _cc_recognizer

# ── Heuristic fallbacks (no external deps) ────────────────────────────────────
_GREETING_RE = re.compile(r"(?im)^\s*(hi|hello|dear)\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?")
_STOP_TOKENS = {
    "info","sales","support","noreply","no-reply","admin","team",
    "contact","service","services","help","alerts","billing","accounts"
}

def _name_from_greeting(text: str) -> Tuple[str, str]:
    m = _GREETING_RE.search(text or "")
    if not m:
        return "", ""
    return m.group(2) or "", (m.group(3) or "")

def _name_from_sender(sender_email: str) -> Tuple[str, str]:
    if not sender_email or "@" not in sender_email:
        return "", ""
    local = sender_email.split("@", 1)[0]
    toks = [t for t in re.split(r"[._+\-]+", local)
            if t and t.isalpha() and t.lower() not in _STOP_TOKENS]
    if not toks:
        return "", ""
    caps = [t.capitalize() for t in toks]
    if len(caps) == 1:
        return caps[0], ""
    # smith_john → John Smith
    if len(caps[1]) <= len(caps[0]):
        return caps[1], caps[0]
    return caps[0], caps[-1]

# Regex fallbacks
_MONEY_FALLBACK_RE = re.compile(
    r"(?:[$£€]\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|"
    r"(?:USD|EUR|GBP|INR|CAD|AUD)\s?-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)",
    re.I,
)
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

def _luhn_ok(digits: str) -> bool:
    try:
        s = 0; alt = False
        for ch in reversed(digits):
            d = ord(ch) - 48
            if d < 0 or d > 9:
                return False
            if alt:
                d *= 2
                if d > 9:
                    d -= 9
            s += d
            alt = not alt
        return s % 10 == 0
    except Exception:
        return False

# ── Public API ────────────────────────────────────────────────────────────────
def extract_person_names(text: Optional[str], sender_email: str = "") -> Tuple[str, str]:
    """
    Prefer HF model; otherwise use greeting/sender heuristics.
    Splits PERSON spans into first/last as needed.
    """
    s = (text or "").strip()
    pipe = _get_hf_pipe()
    if pipe is not None and s:
        try:
            # avoid massive messages
            snippet = s if len(s) <= 4000 else s[:4000]
            ents = pipe(snippet)
            first = last = ""

            # ab-ai/pii_model label sets
            FIRST = {"FIRST_NAME","B-FIRST_NAME","I-FIRST_NAME","FIRSTNAME","GIVEN_NAME","B-GIVEN_NAME","I-GIVEN_NAME"}
            LAST  = {"LAST_NAME","B-LAST_NAME","I-LAST_NAME","SURNAME","FAMILY_NAME","B-SURNAME","I-SURNAME"}
            PERSON= {"PERSON","PER","NAME","B-PER","I-PER","B-NAME","I-NAME"}

            # Priority 1: explicit first/last
            for e in ents:
                lab = (e.get("entity_group") or e.get("entity") or "").upper()
                if not first and lab in FIRST:
                    first = e["word"].strip()
                if not last and lab in LAST:
                    last = e["word"].strip()
                if first and last:
                    return first, last

            # Priority 2: generic PERSON/NAME → split ends
            for e in ents:
                lab = (e.get("entity_group") or e.get("entity") or "").upper()
                if lab in PERSON:
                    parts = e["word"].strip().split()
                    if parts:
                        first = parts[0]
                        last = parts[-1] if len(parts) > 1 else ""
                        return first, last
        except Exception:
            pass

    # Heuristics
    f, l = _name_from_greeting(s)
    if not f:
        f2, l2 = _name_from_sender(sender_email)
        f, l = f or f2, l or l2
    return f or "", l or ""

def find_credit_card(text: Optional[str]) -> str:
    s = text or ""
    currency_rec, cc_rec = _get_presidio_recognizers()
    if cc_rec is not None:
        try:
            results = cc_rec.analyze(s, entities=["CREDIT_CARD"], nlp_artifacts=None)
            if results:
                r = max(results, key=lambda x: x.score)
                return s[r.start:r.end]
        except Exception:
            pass
    # Fallback: regex + Luhn
    for m in _CARD_RE.finditer(s):
        pan = re.sub(r"\D", "", m.group())
        if 13 <= len(pan) <= 19 and _luhn_ok(pan):
            return pan
    return ""

def find_currency(text: Optional[str]) -> str:
    s = text or ""
    currency_rec, _ = _get_presidio_recognizers()
    if currency_rec is not None:
        try:
            results = currency_rec.analyze(s, entities=["CURRENCY"], nlp_artifacts=None)
            if results:
                r = max(results, key=lambda x: x.score)
                return s[r.start:r.end]
        except Exception:
            pass
    # Fallback: regex
    m = _MONEY_FALLBACK_RE.search(s)
    return m.group(0) if m else ""

def extract_entities(text: Optional[str], sender_email: str = "") -> dict:
    """Unified helper used by workers."""
    first, last = extract_person_names(text, sender_email)
    money = find_currency(text)
    card = find_credit_card(text)
    return {
        "first_name": first,
        "last_name": last,
        "money": money,
        "card_number": card,
    }

