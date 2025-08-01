#!/usr/bin/env python3
"""
build_namesets.py  –  merge two open datasets into big name lists

Outputs
-------
datasets/firstnames.txt   (≈780 k lines)
datasets/surnames.txt     (≈470 k lines)

Usage
-----
python build_namesets.py
"""

import csv
import io
import pathlib
import unicodedata
import zipfile
from typing import Set

import requests
from names_dataset import NameDataset

##############################################################################
# config paths
##############################################################################
DST = pathlib.Path("datasets").resolve()
DST.mkdir(exist_ok=True)

FIRST_OUT = DST / "firstnames.txt"
LAST_OUT  = DST / "surnames.txt"

##############################################################################
# helpers
##############################################################################
def norm(name: str) -> str:
    """
    NFC normalise → lower → strip all chars except A-Z, apostrophe, hyphen.
    """
    name = unicodedata.normalize("NFC", name).lower()
    return "".join(c for c in name if c.isalpha() or c in {"-", "'"})

##############################################################################
# 1) load philipperemy/name-dataset via PyPI
##############################################################################
print("• loading name-dataset … (this may take ~15 s)")
nd = NameDataset()                       # first call unpickles ~50 MB dictionary

first: Set[str] = {norm(n) for n in nd.first_names.keys()}
last:  Set[str] = {norm(n) for n in nd.last_names.keys()}

##############################################################################
# 2) fetch & extract 2010 US-Census surnames
##############################################################################
CENSUS_URL = (
    "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
)

CSV_NAME = "Names_2010Census.csv"

csv_path = DST / CSV_NAME
if not csv_path.exists():
    print("• downloading US-Census surnames …")
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    r = requests.get(CENSUS_URL, timeout=30, verify=False)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extract(CSV_NAME, DST)

print("• ingesting Census CSV …")
with open(csv_path, newline="", encoding="utf-8") as fp:
    for row in csv.DictReader(fp):
        n = norm(row["name"])
        if 2 <= len(n) <= 40:            # drop initials & oddities
            last.add(n)

##############################################################################
# 3) write results
##############################################################################
print(f"→ {len(first):,} unique first names")
print(f"→ {len(last):,} unique surnames")

FIRST_OUT.write_text("\n".join(sorted(first)))
LAST_OUT.write_text("\n".join(sorted(last)))

print(f"✓ wrote {FIRST_OUT.relative_to(pathlib.Path.cwd())}")
print(f"✓ wrote {LAST_OUT.relative_to(pathlib.Path.cwd())}")
