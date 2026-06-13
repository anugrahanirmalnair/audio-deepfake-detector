"""Extract per-codebook + transition DAC features for ASVspoof sample."""

import argparse
from pathlib import Path

import dac
import librosa
import pandas as pd
import torch

from dac_token_features import extract_token_features

ROOT = Path(__file__).resolve().parents[2]


def load_dac_model():
    print("Loading DAC model...")
    model = dac.utils.load_model(
        model_type="24khz",
        model_bitrate="8kbps",
    )
    print("Model loaded.\n")
    return model


def encode_tokens(model, audio_path: str) -> torch.Tensor:
    audio, _ = librosa.load(audio_path, sr=None)
    audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model.encode(audio_tensor)
    return output[1].squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-csv",
        default=str(ROOT / "asvspoof_sample.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "asvspoof_features_enriched.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.sample_csv)
    model = load_dac_model()

    rows = []
    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"Processing {idx}/{len(df)}")

        tokens = encode_tokens(model, row["path"]).cpu().numpy()
        feats = extract_token_features(tokens)
        rows.append({"path": row["path"], "label": row["label"], **feats})

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {args.output_csv}")
    print(feature_df.groupby("label")[["entropy", "repeat_rate"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
