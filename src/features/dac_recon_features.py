"""Stream 1: reconstruction-error and continuous latent features."""

import numpy as np


def _distribution_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "std", "max", "p50", "p90", "p99")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def frame_pool_error(error: np.ndarray, hop_length: int) -> np.ndarray:
    """Mean absolute error per codec frame."""
    n_frames = int(np.ceil(len(error) / hop_length))
    pooled = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        end = min((i + 1) * hop_length, len(error))
        if start < end:
            pooled[i] = error[start:end].mean()
    return pooled


def extract_reconstruction_features(error: np.ndarray, hop_length: int) -> dict[str, float]:
    features = _distribution_stats(error, "err")
    frame_err = frame_pool_error(error, hop_length)
    features.update(_distribution_stats(frame_err, "frame_err"))

    if frame_err.size > 0:
        threshold = frame_err.mean() + frame_err.std()
        features["frame_err_high_ratio"] = float(
            (frame_err > threshold).mean()
        )
        features["frame_err_temporal_std"] = float(frame_err.std())
    else:
        features["frame_err_high_ratio"] = 0.0
        features["frame_err_temporal_std"] = 0.0

    return features


def extract_latent_features(z: np.ndarray, latents: np.ndarray) -> dict[str, float]:
    """Summary stats on quantized z and pre-quantization latents."""
    features: dict[str, float] = {}

    for name, tensor in (("z", z), ("latent", latents)):
        if tensor.size == 0:
            continue
        channel_means = tensor.mean(axis=-1)
        channel_stds = tensor.std(axis=-1)
        temporal_std = tensor.std(axis=-1).mean()

        features[f"{name}_global_mean"] = float(tensor.mean())
        features[f"{name}_global_std"] = float(tensor.std())
        features[f"{name}_channel_mean_avg"] = float(channel_means.mean())
        features[f"{name}_channel_mean_std"] = float(channel_means.std())
        features[f"{name}_channel_std_avg"] = float(channel_stds.mean())
        features[f"{name}_temporal_std_avg"] = float(temporal_std)
        features[f"{name}_l2_norm"] = float(np.linalg.norm(tensor) / np.sqrt(tensor.size))

    return features
