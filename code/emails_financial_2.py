import os
import sys
import re
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
FPE_DIR        = Path(os.environ.get("FPE_DIR", "/cs/student/projects2/sec/2024/rdhawan/FPEProject"))
INPUT_CSV      = FPE_DIR / "datasets" / "emails.csv"
KEYWORDS_TXT   = FPE_DIR / "keywords.txt"
OUTPUT_CSV     = FPE_DIR / "datasets" / "emails_financial_2.csv"
LOG_FILE       = FPE_DIR / "filter_financial_emails.log"
NUM_WORKERS    = os.cpu_count() or 4
MIN_WORDS      = 50   # minimum words in body
MIN_MONEY_HITS = 2    # minimum currency matches
CTX_CHARS      = 10   # context chars around money

# ─── Card regexes ───────────────────────────────────────────────────────────────
CARD_REGEXES = {
    'Visa':       re.compile(r"\b4\d{3}([ \-]?)\d{4}\1\d{4}\1\d{4}\b"),
    'MasterCard': re.compile(r"\b5[1-5]\d{2}([ \-]?)\d{4}\1\d{4}\1\d{4}\b"),
    'AmEx':       re.compile(r"\b3[47]\d{2}([ \-]?)\d{6}\1\d{5}\b"),
    'Discover':   re.compile(r"\b6(?:011|5\d{2})([ \-]?)\d{4}\1\d{4}\1\d{4}\b"),
    'Maestro':    re.compile(r"\b(?:5018|5020|5038|5893|6304|6759|6761|6763)([ \-]?)\d{4}\1\d{4}\1\d{4,7}\b")
}


def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("=== Starting email filtering (keywords + money + cards) ===")


def readdataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        logging.error(f"CSV file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, dtype=str, usecols=['file', 'message'], low_memory=False, encoding='utf-8')
    logging.info(f"Loaded {len(df):,} emails from {path}")
    return df


def load_keywords(path: Path) -> list[str]:
    if not path.exists():
        logging.error(f"Keywords file not found: {path}")
        sys.exit(1)
    kws = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    kws = sorted(set(kws))
    logging.info(f"Loaded {len(kws)} keywords for filtering")
    return kws


def build_regex(keywords: list[str]) -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    # keyword regex
    escaped = [re.escape(kw) for kw in keywords]
    kw_re = re.compile(r"\b(?:(?:" + "|".join(escaped) + r"))\b", flags=re.IGNORECASE)
    # money regex
    money_pattern = r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
    money_re = re.compile(money_pattern)
    # money with context
    money_ctx_re = re.compile(rf".{{{CTX_CHARS},}}{money_pattern}.{{{CTX_CHARS},}}", flags=re.DOTALL)
    logging.info("Compiled kw_re, money_re, money_ctx_re with CTX_CHARS=%d", CTX_CHARS)
    return kw_re, money_re, money_ctx_re


def parse_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    def _extract_body(row):
        raw = row['message'] or ''
        text = raw.replace("\r\n", "\n")
        parts = text.split("\n\n", 1)
        body = parts[1] if len(parts) > 1 else ''
        return re.sub(r"[\t ]+", " ", body).strip()

    df = df.copy()
    df['Body'] = df.apply(_extract_body, axis=1)
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


def filter_partition(part_df: pd.DataFrame,
                     kw_re: re.Pattern,
                     money_re: re.Pattern,
                     money_ctx_re: re.Pattern) -> list[dict]:
    results = []
    for _, row in part_df.iterrows():
        body = row['Body']
        # enforce minimum body length
        if len(body.split()) < MIN_WORDS:
            continue
        keep = False
        # 1) keyword match
        if kw_re.search(body):
            keep = True
        # 2) money matches
        elif len(money_re.findall(body)) >= MIN_MONEY_HITS and money_ctx_re.search(body):
            keep = True
        # 3) credit-card match
        else:
            for regex in CARD_REGEXES.values():
                for m in regex.finditer(body):
                    if luhn_checksum(m.group()):
                        keep = True
                        break
                if keep:
                    break
        if keep:
            results.append({'file': row['file'], 'message': row['message']})
    return results


def main():
    setup_logging()

    df_raw = readdataset(INPUT_CSV)
    df_clean = parse_and_clean(df_raw)
    kws = load_keywords(KEYWORDS_TXT)
    kw_re, money_re, money_ctx_re = build_regex(kws)

    chunks = np.array_split(df_clean, NUM_WORKERS)
    logging.info(f"Split data into {len(chunks)} chunks for parallel filtering")

    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(filter_partition, chunk, kw_re, money_re, money_ctx_re)
                   for chunk in chunks]
        for fut in as_completed(futures):
            res = fut.result()
            logging.info(f"Chunk returned {len(res)} matches")
            all_results.extend(res)

    if all_results:
        out_df = pd.DataFrame(all_results)
        out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        logging.info(f"Wrote {len(out_df):,} rows to {OUTPUT_CSV}")
    else:
        logging.info("No matching emails found; no output generated")

    logging.info("=== Finished processing ===")


if __name__ == '__main__':
    main()

