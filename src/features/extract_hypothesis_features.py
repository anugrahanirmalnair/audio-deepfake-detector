"""Extract full dual-stream DAC features for hypothesis validation."""

import argparse
from pathlib import Path

import pandas as pd

from dac_pipeline import load_dac_model, run_dac_forward
from dac_recon_features import extract_latent_features, extract_reconstruction_features
from dac_token_features import extract_token_features
from dac_window_features import extract_window_timeline, extract_windowed_features

ROOT = Path(__file__).resolve().parents[2]


def _build_features_from_dac_output(out: dict) -> dict[str, float]:
    features = {}
    features.update(
        extract_reconstruction_features(out["error"], out["hop_length"])
    )
    features.update(
        extract_latent_features(out["z"], out["latents"])
    )
    features.update(
        extract_token_features(out["codes"], include_histogram=True)
    )
    features.update(
        extract_windowed_features(
            out["error"],
            out["codes"],
            out["hop_length"],
            out["sample_rate"],
        )
    )
    return features


def extract_all_features(model, audio_path: str) -> dict[str, float]:
    out = run_dac_forward(model, audio_path)
    return _build_features_from_dac_output(out)


def extract_analysis(
    model, audio_path: str
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Clip-level features plus per-window timeline for API responses."""
    out = run_dac_forward(model, audio_path)
    features = _build_features_from_dac_output(out)
    windows = extract_window_timeline(
        out["error"],
        out["codes"],
        out["hop_length"],
        out["sample_rate"],
    )
    return features, windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-csv",
        default=str(ROOT / "asvspoof_sample.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "asvspoof_features_full.csv"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip paths already present in output CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.sample_csv)
    done_paths: set[str] = set()
    existing_rows: list[dict] = []
    out_path = Path(args.output_csv)

    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        done_paths = set(prev["path"])
        existing_rows = prev.to_dict("records")
        print(f"Resuming: {len(done_paths)} clips already extracted")

    model = load_dac_model()
    print("DAC model loaded.\n")

    meta_cols = [c for c in df.columns if c not in ("path", "label")]
    rows = list(existing_rows)
    for idx, row in df.iterrows():
        if row["path"] in done_paths:
            continue
        if idx % 20 == 0:
            print(f"Processing {idx}/{len(df)}")
        feats = extract_all_features(model, row["path"])
        record = {"path": row["path"], "label": row["label"], **feats}
        for c in meta_cols:
            record[c] = row[c]
        rows.append(record)
        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({feature_df.shape[1] - 2} feature columns)")


if __name__ == "__main__":
    main()
