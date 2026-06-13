import dac
import librosa
import torch
from pathlib import Path

print("Loading model...")

model = dac.utils.load_model(
    model_type="24khz",
    model_bitrate="8kbps"
)

audio_path = Path(
    "data/raw/voice_dataset/USA/male/1/original.m4a"
)

audio, sr = librosa.load(audio_path, sr=None)

print("Sample Rate:", sr)
print("Audio Shape:", audio.shape)

audio_tensor = torch.tensor(audio)

audio_tensor = audio_tensor.unsqueeze(0)
audio_tensor = audio_tensor.unsqueeze(0)

print("Tensor Shape:", audio_tensor.shape)

with torch.no_grad():
    output = model.encode(audio_tensor)

print(type(output))

for i, item in enumerate(output):
    print(f"\nItem {i}")
    print(type(item))

    if hasattr(item, "shape"):
        print(item.shape)

tokens = output[1]
print("Shape:", tokens.shape)

print("Min Token:", tokens.min().item())
print("Max Token:", tokens.max().item())

print("Unique Tokens:",
      len(torch.unique(tokens)))