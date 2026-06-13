import dac
import librosa
import torch
import numpy as np
import pandas as pd

from pathlib import Path
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


root = Path(
    "data/raw/voice_dataset"
)

rows = []

for file in root.rglob("*"):

    if file.suffix.lower() not in [
        ".wav",
        ".mp3",
        ".m4a"
    ]:
        continue

    label = (
        "real"
        if "original" in file.name.lower()
        else "fake"
    )

    print(f"Processing {file}")

    feats = extract_features(file)

    rows.append({
        "path": str(file),
        "label": label,
        **feats
    })


df = pd.DataFrame(rows)

print("\n\n========== SUMMARY ==========\n")

print(
    df.groupby("label")[
        ["entropy", "repeat_rate"]
    ].agg(
        ["mean", "std"]
    )
)

df.to_csv(
    "dataset_features.csv",
    index=False
)

print(
    "\nSaved dataset_features.csv"
)