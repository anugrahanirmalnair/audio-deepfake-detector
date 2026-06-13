import dac
import librosa
import torch
import numpy as np
from collections import Counter
from pathlib import Path


# --------------------------
# Load DAC once
# --------------------------

print("Loading DAC model...")

model = dac.utils.load_model(
    model_type="24khz",
    model_bitrate="8kbps"
)

print("Model loaded.\n")


# --------------------------
# Entropy helper
# --------------------------

def calculate_entropy(token_array):
    """
    Shannon entropy of token distribution
    """

    counts = Counter(token_array.flatten())

    total = sum(counts.values())

    probs = np.array([
        count / total
        for count in counts.values()
    ])

    entropy = -np.sum(
        probs * np.log2(probs + 1e-12)
    )

    return entropy


# --------------------------
# Extract DAC features
# --------------------------

def extract_features(audio_path):

    print(f"\nProcessing: {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=None
    )

    audio_tensor = torch.tensor(audio)

    # shape:
    # [samples]
    # -> [1, samples]
    # -> [1, 1, samples]

    audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model.encode(audio_tensor)

    # Item 1 = token IDs
    tokens = output[1]

    tokens = tokens.squeeze(0)

    unique_tokens = len(
        torch.unique(tokens)
    )

    entropy = calculate_entropy(
        tokens.cpu().numpy()
    )

    repeat_rate = (
        (tokens[:, 1:] == tokens[:, :-1])
        .float()
        .mean()
        .item()
    )

    return {
        "shape": tuple(tokens.shape),
        "unique_tokens": unique_tokens,
        "entropy": entropy,
        "repeat_rate": repeat_rate
    }


# --------------------------
# One real
# --------------------------

real_path = Path(
    "data/raw/voice_dataset/USA/male/1/original.m4a"
)

real_stats = extract_features(
    real_path
)


# --------------------------
# One fake
# --------------------------

fake_path = Path(
    "data/raw/voice_dataset/USA/male/1/synthetic_1.mp3"
)

fake_stats = extract_features(
    fake_path
)


# --------------------------
# Results
# --------------------------

print("\n" + "=" * 50)
print("REAL AUDIO")
print("=" * 50)

for k, v in real_stats.items():
    print(f"{k}: {v}")

print("\n" + "=" * 50)
print("FAKE AUDIO")
print("=" * 50)

for k, v in fake_stats.items():
    print(f"{k}: {v}")