"""
email_processing.py – extract metadata from raw RFC‑822 messages.
Now uses policy.compat32 so broken headers don’t abort parsing.
"""
from __future__ import annotations
import re
import email
from email import policy
from email.utils import getaddresses
from typing import Dict, Any, Tuple

# ─── helpers ─────────────────────────────────────────────────────────────────
_SUBJ_RE = re.compile(r"^(?:(?:re|fw|fwd)\s*[:\]]\s*)+", re.I)
_ADDR_RE = re.compile(r"[^@,\s]+@[^@,\s]+")


def normalise_subject(subj: str | None) -> str:
    if subj is None:
        return ""
    subj = _SUBJ_RE.sub("", subj).strip()
    return re.sub(r"\s+", " ", subj)


def addr_set(field_val: str | None) -> Tuple[str, ...]:
    if not field_val:
        return tuple()
    return tuple(
        sorted(
            {
                a.lower()
                for _, a in getaddresses([field_val])
                if _ADDR_RE.fullmatch(a)
            }
        )
    )


# ─── main extractor ─────────────────────────────────────────────────────────
def parse_email_raw(raw: str) -> Dict[str, Any]:
    """
    Robustly parse *raw* message → dict with sender, recipients, etc.
    Any exception returns blank fields rather than raising.
    """
    try:
        # compat32 ⇒ headers come back as plain text; no structured parsing
        msg = email.message_from_string(raw, policy=policy.compat32)

        sender = addr_set(msg.get("From"))
        to_addrs = addr_set(msg.get("To"))
        cc_addrs = addr_set(msg.get("Cc"))

        # Plain‑text body (first text/plain part, else whole payload)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", "replace"
                    ).strip()
                    break
        else:
            body = msg.get_payload(decode=True).decode(
                msg.get_content_charset() or "utf-8", "replace"
            ).strip()

        subj_raw = (msg.get("Subject") or "").strip()

        return {
            "sender": sender[0] if sender else "",
            "to": ", ".join(to_addrs),
            "cc": ", ".join(cc_addrs),
            "subject": subj_raw,
            "subj_norm": normalise_subject(subj_raw),
            "participants_sig": "|".join(sorted({*sender, *to_addrs, *cc_addrs})),
            "date": msg.get("Date"),
            "body": body,
        }

    except Exception:
        # Fallback for completely broken messages
        return {
            "sender": "",
            "to": "",
            "cc": "",
            "subject": "",
            "subj_norm": "",
            "participants_sig": "",
            "date": "",
            "body": "",
        }
