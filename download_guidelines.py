"""
download_guidelines.py
----------------------
Downloads free NCCN Guidelines for Patients PDFs directly from nccn.org
(no login required — these are publicly available patient resources).

Run once before starting the FastAPI server:
    python download_guidelines.py

PDFs are saved to the auto_index/ folder, which FastAPI indexes on startup.
"""

import os
import time
import urllib.request
from pathlib import Path

# ── Output folder (same folder FastAPI auto-indexes on startup) ──────────────
AUTO_INDEX_DIR = Path(__file__).resolve().parent / "auto_index"
AUTO_INDEX_DIR.mkdir(exist_ok=True)

# ── Free NCCN Guidelines for Patients — direct PDF URLs (no login needed) ───
# Pattern: https://www.nccn.org/patients/guidelines/content/PDF/{slug}-patient.pdf
GUIDELINES = [
    # Cancer type                        slug
    ("Breast Cancer - Invasive",        "breast"),
    ("Breast Cancer - Metastatic",      "breast-metastatic"),
    ("Lung Cancer - Non-Small Cell",    "nsclc"),
    ("Lung Cancer - Small Cell",        "sclc"),
    ("Colon Cancer",                    "colon"),
    ("Prostate Cancer",                 "prostate"),
    ("Ovarian Cancer",                  "ovarian"),
    ("Cervical Cancer",                 "cervical"),
    ("Pancreatic Cancer",               "pancreatic"),
    ("Stomach Cancer",                  "stomach"),
    ("Liver Cancer",                    "liver-hp"),
    ("Melanoma - Skin",                 "melanoma"),
    ("Chronic Lymphocytic Leukemia",    "cll"),
    ("Chronic Myeloid Leukemia",        "cml"),
    ("Hodgkin Lymphoma",                "hodgkin"),
    ("Diffuse Large B-Cell Lymphoma",   "dlbcl"),
    ("Multiple Myeloma",                "myeloma"),
    ("Acute Lymphoblastic Leukemia",    "all"),
    ("Thyroid Cancer",                  "thyroid"),
    ("Bladder Cancer",                  "bladder"),
]

BASE_URL = "https://www.nccn.org/patients/guidelines/content/PDF/{slug}-patient.pdf"

# ── Download helper ───────────────────────────────────────────────────────────

def download(name: str, slug: str) -> bool:
    url      = BASE_URL.format(slug=slug)
    filename = f"NCCN_{slug}_patient.pdf"
    dest     = AUTO_INDEX_DIR / filename

    if dest.exists():
        print(f"  [skip]  {name} — already downloaded")
        return True

    print(f"  [get]   {name}")
    print(f"          {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OncologyAI-Demo/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()

        # Sanity check: must be a real PDF (starts with %PDF)
        if not content.startswith(b"%PDF"):
            print(f"  [warn]  Response is not a PDF — skipping {name}")
            return False

        dest.write_bytes(content)
        size_kb = len(content) // 1024
        print(f"  [ok]    Saved {filename} ({size_kb} KB)")
        return True

    except Exception as exc:
        print(f"  [fail]  {name}: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nOncology AI — NCCN Guideline Downloader")
    print(f"Output folder: {AUTO_INDEX_DIR}\n")

    ok_count   = 0
    fail_count = 0

    for name, slug in GUIDELINES:
        success = download(name, slug)
        if success:
            ok_count += 1
        else:
            fail_count += 1
        time.sleep(1)          # be polite — 1 second between requests

    print(f"\nDone. {ok_count} downloaded, {fail_count} failed.")
    print(f"\nFiles in auto_index/:")
    for f in sorted(AUTO_INDEX_DIR.glob("*.pdf")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    print("\nNow start the FastAPI server — it will auto-index all PDFs on startup.")
    print("  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
