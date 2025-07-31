import os
import sys
import re
import logging
from pathlib import Path

import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────────
FPE_DIR      = Path(os.environ.get("FPE_DIR", "/cs/student/projects2/sec/2024/rdhawan/FPEProject"))
INPUT_CSV    = FPE_DIR / "datasets" / "emails.csv"
OUTPUT_CSV   = FPE_DIR / "datasets" / "emails_financial_1.csv"
LOG_FILE     = FPE_DIR / "filter_financial_emails.log"

# ─── Card regexes ───────────────────────────────────────────────────────────────
CARD_REGEXES = {
    'Visa': re.compile(r"\b4\d{3}([ \-]?)\d{4}\1\d{4}\1\d{4}\b"),
    'MasterCard': re.compile(r"\b5[1-5]\d{2}([ \-]?)\d{4}\1\d{4}\1\d{4}\b"),
    'AmEx': re.compile(r"\b3[47]\d{2}([ \-]?)\d{6}\1\d{5}\b"),
    'Discover': re.compile(r"\b6(?:011|5\d{2})([ \-]?)\d{4}\1\d{4}\1\d{4}\b"),
    'Maestro': re.compile(r"\b(?:5018|5020|5038|5893|6304|6759|6761|6763)([ \-]?)\d{4}\1\d{4}\1\d{4,7}\b")
}

def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("=== Starting financial‑email filter ===")


def readdataset(path: str) -> pd.DataFrame:
    """
    Load the Enron emails CSV into a DataFrame.
    Expects at least columns: 'file' and 'message'.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        logging.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(
            csv_path,
            dtype=str,
            usecols=['file', 'message'],
            low_memory=False,
            encoding='utf-8'
        )
        logging.info(f"Loaded dataset from {csv_path} ({len(df):,} rows)")
        return df
    except Exception as e:
        logging.exception(f"Failed to read CSV: {e}")
        sys.exit(1)


def parse_email(row: pd.Series) -> pd.Series:
    """
    Split the raw 'message' text into:
      - 'file'    : original file identifier
      - 'message' : full raw message text
      - 'Headers' : all header lines up to the first blank line
      - 'Body'    : text after the first blank line, whitespace‑normalized
    """
    raw = row['message'] or ''
    text = raw.replace("\r\n", "\n")
    parts = text.split("\n\n", 1)
    headers = parts[0]
    body = parts[1] if len(parts) > 1 else ''
    body_clean = re.sub(r"[\t ]+", " ", body).strip()
    return pd.Series({
        'file':    row['file'],
        'message': raw,
        'Headers': headers,
        'Body':    body_clean
    })


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Further normalize the 'Body' field: collapse spaces, periods, newlines.
    """
    if 'Body' in df.columns:
        df = df.copy()
        df['Body'] = (
            df['Body'].fillna("")
                     .astype(str)
                     .str.replace(r"[ \t]+", " ", regex=True)
                     .str.replace(r"\.{2,}", ".", regex=True)
                     .str.replace(r"\n{3,}", "\n\n", regex=True)
                     .str.strip()
        )
    return df


def luhn_checksum(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return (checksum % 10) == 0


def find_cards_in_body(body: str) -> bool:
    """
    Return True if any Luhn-valid card number is found in 'body'.
    """
    for regex in CARD_REGEXES.values():
        for m in regex.finditer(body):
            if luhn_checksum(m.group()):
                return True
    return False


def main():
    setup_logging()

    # 1) Read dataset
    raw_df = readdataset(INPUT_CSV)

    # 2) Parse headers/body
    parsed = raw_df.apply(parse_email, axis=1)

    # 3) Clean up Body
    cleaned = clean_dataframe(parsed)

    # 4) Filter emails containing any valid card number
    results = []
    for _, row in cleaned.iterrows():
        if find_cards_in_body(row['Body']):
            results.append({
                'file':    row['file'],
                'message': row['message']
            })

    logging.info(f"Identified {len(results):,} emails with valid card occurrences")

    # 5) Write out file and message only
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Wrote {len(out_df):,} rows to {OUTPUT_CSV}")
    else:
        logging.info("No valid card numbers found; no output CSV generated")

    logging.info("=== Finished financial‑email filter ===")


if __name__ == '__main__':
    main()

