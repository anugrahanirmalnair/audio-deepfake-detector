#!/usr/bin/env python3
"""
End-to-end API test: send one bonafide and one spoof ASVspoof clip to POST /analyze.

Prerequisites:
  1. API running: uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
  2. pip install requests  (if not already installed)

Usage:
  python test_api.py
  python test_api.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE_CSV = REPO_ROOT / "asvspoof_sample_2k.csv"
FALLBACK_SAMPLE_CSV = REPO_ROOT / "asvspoof_sample.csv"


def resolve_audio_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_file():
        return p
    candidate = REPO_ROOT / raw_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Audio not found: {raw_path}")


def load_sample_paths(sample_csv: Path) -> tuple[Path, Path]:
    import pandas as pd

    df = pd.read_csv(sample_csv)
    if "label" not in df.columns or "path" not in df.columns:
        raise ValueError(f"CSV must have path and label columns: {sample_csv}")

    bonafide_row = df[df["label"] == "bonafide"].iloc[0]
    spoof_row = df[df["label"] == "spoof"].iloc[0]
    return (
        resolve_audio_path(str(bonafide_row["path"])),
        resolve_audio_path(str(spoof_row["path"])),
    )


def check_health(base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/health"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Unexpected health response: {data}")
    print(f"GET /health -> {data}\n")


def analyze_file(base_url: str, audio_path: Path, label: str) -> dict:
    url = f"{base_url.rstrip('/')}/analyze"
    mime = "audio/flac" if audio_path.suffix.lower() == ".flac" else "application/octet-stream"

    print("=" * 60)
    print(f"{label.upper()} — {audio_path.name}")
    print(f"POST {url}")
    print("=" * 60)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"audio": (audio_path.name, f, mime)},
            timeout=600,
        )

    if not resp.ok:
        print(f"HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        resp.raise_for_status()

    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test /analyze with real vs spoof ASVspoof audio")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--sample-csv",
        default="",
        help="CSV with path,label columns (default: asvspoof_sample_2k.csv or asvspoof_sample.csv)",
    )
    parser.add_argument(
        "--real",
        default="",
        help="Override bonafide audio path",
    )
    parser.add_argument(
        "--spoof",
        default="",
        help="Override spoof audio path",
    )
    args = parser.parse_args()

    if args.real and args.spoof:
        real_path = resolve_audio_path(args.real)
        spoof_path = resolve_audio_path(args.spoof)
    else:
        sample_csv = Path(args.sample_csv) if args.sample_csv else None
        if sample_csv is None or not sample_csv.is_file():
            sample_csv = (
                DEFAULT_SAMPLE_CSV
                if DEFAULT_SAMPLE_CSV.is_file()
                else FALLBACK_SAMPLE_CSV
            )
        if not sample_csv.is_file():
            print(
                "No sample CSV found. Pass --real and --spoof, or add asvspoof_sample_2k.csv",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using samples from: {sample_csv}\n")
        real_path, spoof_path = load_sample_paths(sample_csv)

    check_health(args.base_url)

    for path, label in ((real_path, "bonafide (REAL expected)"), (spoof_path, "spoof (FAKE expected)")):
        result = analyze_file(args.base_url, path, label)
        print(json.dumps(result, indent=2))
        print(
            f"\nSummary: verdict={result['verdict']}  "
            f"spoof_probability={result['spoof_probability']}  "
            f"authenticity_score={result['authenticity_score']}  "
            f"confidence={result['confidence']}  "
            f"windows={len(result['windows'])}\n"
        )


if __name__ == "__main__":
    main()
