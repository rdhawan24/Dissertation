# classify_emails.py
"""
Bulk‑classify corporate emails with a local DeepSeek‑LLM 7B Chat model.

The script expects **processed** e‑mails produced by your existing pipeline
(`main.py`).  It can ingest either a single consolidated CSV (default) **or** a
directory containing many `part_*.csv` slices (`--csv-dir`).  Each record must
contain at least two columns:

    file     – original source filename (String)
    message  – full raw e‑mail text (String)

Key features
============
* Batching that respects the 16 384‑token context window of DeepSeek 7B.
* Prompts ("probes") stored in a separate text file and supplied via
  `--probe-file`.
* Optional **dummy mode** (`--dummy`) for fast, offline testing without the LLM.
* Robust JSON parsing and fallback to "unknown" labels on malformed output.
* Flexible CLI allowing token‑budget tweaks, device selection, and progress
  monitoring.
* Prints a classification summary table at the end for quick sanity‑check.

Example
-------
```bash
python classify_emails.py \
  --csv processed_emails.csv \
  --probe-file probes.txt \
  --model-path ~/models/deepseek-llm-7b-chat \
  --output classified_emails.csv
```

or, when processing live slices:

```bash
python classify_emails.py \
  --csv-dir partial_output/ \
  --probe-file probes.txt \
  --model-path ~/models/deepseek-llm-7b-chat
```
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import polars as pl
from tqdm import tqdm

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
    )
except ImportError as exc:
    sys.exit("ERROR: transformers not found – pip install transformers")

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def load_probes(path: str | Path) -> str:
    """Read the probe template used as the common prefix for every prompt."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def yield_records(df: pl.DataFrame) -> Iterable[Tuple[str, str]]:
    """Generator over (file, body) pairs.

    The DataFrame *may* contain additional columns; only the required two are
    accessed.  Using a generator keeps memory usage ~O(batch_size).
    """
    files = df["file"].to_list()
    msgs = df["message"].to_list()
    return zip(files, msgs, strict=True)


def group_batches(
    pairs: Iterable[Tuple[str, str]],
    tokenizer,
    max_tokens: int,
    reserve_tokens: int,
) -> Iterable[List[Tuple[str, str]]]:
    """Greedy batching subject to a total‑token budget.

    *pairs*      – (file, message) iterator
    *tokenizer*  – HF tokenizer to count tokens
    *max_tokens* – hard context limit (16 384 for DeepSeek‑7B)
    *reserve_tokens* – kept free for the probe + JSON overhead
    """
    current: List[Tuple[str, str]] = []
    used_tokens = 0

    for fname, body in pairs:
        # token budget for one sample: "Filename: … Body: …"
        sample_text = f"Filename: {fname}\nBody:\n{body}"
        sample_tokens = len(tokenizer.encode(sample_text))

        if used_tokens + sample_tokens + reserve_tokens > max_tokens:
            if current:  # flush batch
                yield current
                current, used_tokens = [], 0
            # If single sample already exceeds budget, still yield singly
            if sample_tokens + reserve_tokens > max_tokens:
                yield [(fname, body)]
                continue
        current.append((fname, body))
        used_tokens += sample_tokens

    if current:
        yield current


def call_llm(pipe, prompt: str, max_new_tokens: int) -> str:
    """Wrapper around HF pipeline for consistent parameters."""
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )[0]["generated_text"]
    return out


def parse_json_array(text: str, expected_len: int) -> List[Dict[str, str]]:
    """Extract and parse the first JSON array appearing in *text*."""
    try:
        start = text.index("[")
        parsed = json.loads(text[start:])
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass  # fall through to fallback
    return [{"category": "unknown", "topic": "unknown"} for _ in range(expected_len)]


def dummy_labels(n: int) -> List[Dict[str, str]]:
    cats = ["approval", "update", "request"]
    topics = ["HR", "finance", "travel", "others"]
    rnd = random.Random(42)
    return [{"category": rnd.choice(cats), "topic": rnd.choice(topics)} for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Classify e‑mails with DeepSeek‑LLM 7B")

    gsrc = ap.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--csv", help="CSV file produced by main.py")
    gsrc.add_argument("--csv-dir", help="Directory containing many part_*.csv files")

    ap.add_argument("--probe-file", required=True, help="Prompt template text file")
    ap.add_argument("--model-path", default="deepseek-ai/deepseek-llm-7b-chat",
                    help="HF model dir or name (default: deepseek 7B chat)")
    ap.add_argument("--output", default="classified_emails.csv", help="Output CSV path")

    ap.add_argument("--max-rows", type=int,
                    help="Process only first N rows (debug)")
    ap.add_argument("--batch-tokens", type=int, default=16_000, help="Token budget per LLM call")
    ap.add_argument("--reserve", type=int, default=2_000, help="Tokens reserved for probe + JSON")
    ap.add_argument("--max-new", type=int, default=1024, help="max_new_tokens for generation")

    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device map override for HF (default: auto)")
    ap.add_argument("--dummy", action="store_true", help="Skip LLM; generate random labels")

    args = ap.parse_args()

    # ── Load source data ────────────────────────────────────────────────
    if args.csv_dir:
        part_files = sorted(Path(args.csv_dir).glob("part_*.csv"))
        if not part_files:
            sys.exit("ERROR: --csv-dir provided but no part_*.csv found")
        df = pl.concat([pl.read_csv(p) for p in part_files], how="vertical")
    else:
        df = pl.read_csv(args.csv)

    if args.max_rows:
        df = df.head(args.max_rows)

    # ── Load probe template ─────────────────────────────────────────────
    probe = load_probes(args.probe_file)

    # ── Prepare tokenizer / model unless dummy ─────────────────────────
    if not args.dummy:
        print("[*] Loading tokenizer/model…", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=args.device
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        tokenizer = None  # type: ignore
        pipe = None       # type: ignore

    pairs_iter = yield_records(df)

    # ── Classification loop ────────────────────────────────────────────
    results: List[Dict[str, str]] = []

    batches = (
        group_batches(pairs_iter, tokenizer, args.batch_tokens, args.reserve)
        if not args.dummy else [list(pairs_iter)]  # all at once in dummy mode
    )

    for batch in tqdm(list(batches), desc="Classifying", unit="batch"):
        if args.dummy:
            res = dummy_labels(len(batch))
        else:
            joined = "\n\n".join(
                f"Email {i+1}:\nFilename: {fname}\nBody:\n{body}"
                for i, (fname, body) in enumerate(batch)
            )
            prompt = f"{probe}\n\n{joined}\n\nRespond in JSON:"
            raw_out = call_llm(pipe, prompt, args.max_new)
            res = parse_json_array(raw_out, len(batch))
        results.extend(res)

    # ── Sanity check ───────────────────────────────────────────────────
    if len(results) != len(df):
        print(
            f"[!] Warning: Expected {len(df)} results, got {len(results)} – padding",
            file=sys.stderr,
        )
        while len(results) < len(df):
            results.append({"category": "unknown", "topic": "unknown"})

    # ── Write output ────────────────────────────────────────────────────
    out_df = df.with_columns([
        pl.Series("classification", [r["category"] for r in results]),
        pl.Series("topic",          [r["topic"]     for r in results]),
    ])
    out_df.write_csv(args.output)

    # ── Print summary ──────────────────────────────────────────────────
    summary = (
        out_df.groupby(["classification", "topic"]).len().sort(by="len", descending=True)
    )
    print("\nClassification summary:\n", summary, file=sys.stderr)
    print(f"\n[✓] Done – output saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

