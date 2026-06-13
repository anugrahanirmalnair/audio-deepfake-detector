import dac
import librosa
import torch
import numpy as np
import pandas as pd

from collections import Counter


print("Loading DAC model...")

model = dac.utils.load_model(
    model_type="24khz",
    model_bitrate="8kbps"
)

print("Model loaded.\n")


def entropy(tokens):

    counts = Counter(tokens.flatten())

    total = sum(counts.values())

    probs = np.array(
        [c / total for c in counts.values()]
    )

    return -np.sum(
        probs * np.log2(probs + 1e-12)
    )


def extract_features(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=None
    )

    audio_tensor = torch.tensor(audio)

    audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model.encode(audio_tensor)

    tokens = output[1].squeeze(0)

    repeat_rate = (
        (tokens[:, 1:] == tokens[:, :-1])
        .float()
        .mean()
        .item()
    )

    return {
        "entropy": entropy(
            tokens.cpu().numpy()
        ),
        "repeat_rate": repeat_rate
    }


df = pd.read_csv(
    "asvspoof_sample.csv"
)

rows = []

for idx, row in df.iterrows():

    if idx % 20 == 0:
        print(
            f"Processing {idx}/{len(df)}"
        )

    feats = extract_features(
        row["path"]
    )

    rows.append({
        "path": row["path"],
        "label": row["label"],
        **feats
    })


feature_df = pd.DataFrame(rows)

feature_df.to_csv(
    "asvspoof_features.csv",
    index=False
)

print("\nSaved asvspoof_features.csv")

print(
    feature_df.groupby("label")[
        ["entropy", "repeat_rate"]
    ].agg(
        ["mean", "std"]
    )
)