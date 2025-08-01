# llm_pipeline.py
"""
Two‑phase pipeline
==================
1. **PII extraction / optional encryption** — spawns an existing ``main.py``
   (path via ``--main``).  Each worker writes its slice to ``part_*.csv`` in a
   temporary directory.
2. **LLM probing** — concatenates those slice files, then asks DeepSeek‑LLM a
   set of user‑defined *probes* for **every** e‑mail body.

Outputs
-------
* A full **Parquet** file of merged results (path via ``--out``).
* A **sample CSV** (first 100 rows) named ``<out>.sample.csv`` for quick human
  inspection.

Assumptions
-----------
* ``openai`` Python client ≥ 1.0.
* Local **vLLM** server exposing an OpenAI‑compatible endpoint at
  ``http://localhost:8000`` running the model
  ``deepseek-ai/deepseek-llm-7b-chat``.
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# ── Third‑party ───────────────────────────────────────────────────────────
import polars as pl
from openai import OpenAI
from tqdm import tqdm

# ── LLM client setup ──────────────────────────────────────────────────────
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "deepseek-ai/deepseek-llm-7b-chat"

# ───────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────

def _run_main(
    main_py: Path,
    csv_path: str,
    model_dir: str,
    encrypt: bool,
    slice_rows: int | None,
    jobs: int,
    num_rows: int | None,
    partial_dir: Path,
) -> None:
    """Invoke *main.py* with the desired parameters."""

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(main_py),
        csv_path,
        "-m",
        model_dir,
        f"-j{jobs}",
        f"--partial-dir={partial_dir}",
    ]
    if slice_rows is not None:
        cmd.append(f"--slice-rows={slice_rows}")
    if num_rows is not None:
        cmd.append(f"-n{num_rows}")
    if encrypt:
        cmd.append("-e")

    print("[llm] ⇢  Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode:
        sys.exit(f"[llm] main.py exited with status {proc.returncode}")


def _load_processed_dataframe(partial_dir: Path) -> pl.DataFrame:
    """Read every ``part_*.csv`` into one Polars DataFrame."""

    parts = sorted(partial_dir.glob("part_*.csv"))
    if not parts:
        sys.exit("[llm] Expected slice files in --partial-dir, found none.")

    dfs = [pl.read_csv(p) for p in parts]
    return pl.concat(dfs, how="vertical")


def _build_messages(template: str, body: str) -> List[Dict[str, str]]:
    """Wrap the rendered template in OpenAI chat format."""

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer strictly according to the user instructions.",
        },
        {"role": "user", "content": template.format(body=body)},
    ]


def _ask_llm(template: str, body: str) -> str:
    """Fire one prompt → return assistant text."""

    messages = _build_messages(template, body)
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content.strip()


def _probe_row(file: str, body: str, probes: List[Dict[str, str]]) -> Dict[str, str]:
    """Run all probes for a single e‑mail body."""

    row: Dict[str, str] = {"file": file}
    for probe in probes:
        try:
            row[probe["name"]] = _ask_llm(probe["prompt"], body)
        except Exception as exc:
            row[probe["name"]] = f"[error] {exc}"
    return row


def _apply_probes(df: pl.DataFrame, probes: List[Dict[str, str]], workers: int) -> pl.DataFrame:
    """Parallel‑run _probe_row across the DataFrame."""

    rows = [(r["file"], r["message"]) for r in df.iter_rows(named=True)]
    results: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_probe_row, f, m, probes): f for f, m in rows}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LLM probes"):
            results.append(fut.result())

    return pl.DataFrame(results)

# ───────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PII + LLM pipeline")
    parser.add_argument("csv", help="Input CSV (file,message)")
    parser.add_argument("-m", "--model-dir", required=True)
    parser.add_argument("--probes", required=True, help="JSON list of probe objects")
    parser.add_argument("-o", "--out", required=True, help="Output Parquet path")
    parser.add_argument("--main", default="main.py", help="Path to main.py to execute")
    parser.add_argument("-e", "--encrypt", action="store_true", help="Enable -e for main.py")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--slice-rows", type=int)
    parser.add_argument("-n", "--num-rows", type=int)
    parser.add_argument("--llm-workers", type=int, default=4)
    args = parser.parse_args()

    # ── Parse probes JSON ────────────────────────────────────────────────
    probes: List[Dict[str, str]] = json.loads(Path(args.probes).read_text())
    if not probes:
        sys.exit("[llm] --probes is empty; nothing to ask the model.")

    # ── Validate main.py path ────────────────────────────────────────────
    main_path = Path(args.main).expanduser()
    if not main_path.exists():
        sys.exit(f"[llm] main.py not found at {main_path}")

    # ── Phase 1: run PII extraction ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        part_dir = Path(tmpdir) / "parts"
        _run_main(
            main_path,
            args.csv,
            args.model_dir,
            args.encrypt,
            args.slice_rows,
            args.jobs,
            args.num_rows,
            part_dir,
        )

        df_processed = _load_processed_dataframe(part_dir)
        # ── Phase 2: LLM probes ─────────────────────────────────────────
        df_llm = _apply_probes(df_processed, probes, args.llm_workers)

    # ── Merge and write outputs ──────────────────────────────────────────
    df_final = df_processed.join(df_llm, on="file", how="left")

    parquet_path = Path(args.out)
    df_final.write_parquet(parquet_path)

    sample_path = parquet_path.with_suffix(".sample.csv")
    df_final.head(100).write_csv(sample_path)

    print(f"[llm] Parquet written to: {parquet_path}")
    print(f"[llm] Sample CSV (100 rows): {sample_path}")


if __name__ == "__main__":
    main()

