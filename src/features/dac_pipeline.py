"""Load audio and run DAC encode / decode (project-brief backbone)."""

from __future__ import annotations

import librosa
import numpy as np
import torch
from pathlib import Path

from pathlib import Path

def load_dac_model():
    import dac

    weights_path = Path("/app/weights/weights_24khz_8kbps_0.0.4.pth")

    return dac.utils.load_model(load_path=str(weights_path))

def load_audio_for_dac(audio_path: str, sample_rate: int) -> torch.Tensor:
    """Return float tensor [1, 1, T] at DAC sample rate."""
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    waveform = torch.tensor(audio, dtype=torch.float32)
    return waveform.unsqueeze(0).unsqueeze(0)


def run_dac_forward(model, audio_path: str) -> dict:
    """
    Single encode+decode pass. Returns tensors needed for dual-stream features.
    """
    sample_rate = int(model.sample_rate)
    audio = load_audio_for_dac(audio_path, sample_rate)
    length = audio.shape[-1]

    with torch.no_grad():
        audio_prep = model.preprocess(audio, sample_rate)
        z, codes, latents, _, _ = model.encode(audio_prep)
        recon = model.decode(z)

    min_len = min(audio.shape[-1], recon.shape[-1])
    error = (
        (audio[..., :min_len] - recon[..., :min_len])
        .abs()
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )
    audio = audio[..., :min_len]
    recon = recon[..., :min_len]
    z_np = z.squeeze(0).cpu().numpy()
    codes_np = codes.squeeze(0).cpu().numpy().astype(np.int64)
    latents_np = latents.squeeze(0).cpu().numpy()
    recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()

    return {
        "audio": audio.squeeze().cpu().numpy(),
        "recon": recon_np,
        "error": error,
        "z": z_np,
        "codes": codes_np,
        "latents": latents_np,
        "sample_rate": sample_rate,
        "hop_length": int(model.hop_length),
    }
