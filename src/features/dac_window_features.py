"""Stream 4: 500 ms windowed aggregates (local artifact detection)."""

import numpy as np

from dac_recon_features import frame_pool_error
from dac_token_features import shannon_entropy


def window_frame_indices(n_frames: int, frames_per_window: int) -> list[tuple[int, int]]:
    if n_frames == 0 or frames_per_window <= 0:
        return []
    windows = []
    start = 0
    while start < n_frames:
        end = min(start + frames_per_window, n_frames)
        if end > start:
            windows.append((start, end))
        start += frames_per_window
    return windows


def _compute_window_series(
    error: np.ndarray,
    codes: np.ndarray,
    hop_length: int,
    sample_rate: int,
    window_ms: float = 500.0,
) -> tuple[list[tuple[int, int]], list[float], list[float], list[float]]:
    """Return window frame ranges and per-window recon / token signals."""
    frames_per_window = max(1, int((window_ms / 1000.0) * sample_rate / hop_length))
    frame_err = frame_pool_error(error, hop_length)
    n_frames = frame_err.shape[0]
    windows = window_frame_indices(n_frames, frames_per_window)

    win_frame_err: list[float] = []
    win_entropy: list[float] = []
    win_repeat: list[float] = []

    for start, end in windows:
        win_frame_err.append(float(frame_err[start:end].mean()))
        slice_codes = codes[:, start:end]
        if slice_codes.shape[1] == 0:
            win_entropy.append(0.0)
            win_repeat.append(0.0)
            continue
        flat = slice_codes.reshape(-1)
        counts = np.bincount(flat.astype(np.int64), minlength=1024)
        win_entropy.append(shannon_entropy(counts))
        repeats = []
        for cb in range(slice_codes.shape[0]):
            seq = slice_codes[cb]
            if seq.size >= 2:
                repeats.append((seq[1:] == seq[:-1]).mean())
        win_repeat.append(float(np.mean(repeats)) if repeats else 0.0)

    return windows, win_frame_err, win_entropy, win_repeat


def extract_window_timeline(
    error: np.ndarray,
    codes: np.ndarray,
    hop_length: int,
    sample_rate: int,
    window_ms: float = 500.0,
) -> list[dict[str, float]]:
    """Per-window scores for API heatmaps (500 ms windows by default)."""
    windows, win_frame_err, win_entropy, win_repeat = _compute_window_series(
        error, codes, hop_length, sample_rate, window_ms
    )
    ms_per_frame = hop_length / sample_rate * 1000.0
    timeline = []
    for (start, end), recon, entropy, repeat in zip(
        windows, win_frame_err, win_entropy, win_repeat
    ):
        # token_score: higher repeat + lower entropy variability → more synthetic-like
        token_score = float(repeat * 100.0 + entropy * 0.01)
        timeline.append(
            {
                "start_ms": round(start * ms_per_frame, 2),
                "end_ms": round(end * ms_per_frame, 2),
                "recon_score": round(recon, 6),
                "token_score": round(token_score, 6),
            }
        )
    return timeline


def extract_windowed_features(
    error: np.ndarray,
    codes: np.ndarray,
    hop_length: int,
    sample_rate: int,
    window_ms: float = 500.0,
) -> dict[str, float]:
    """
    Per-window recon + token stats, then mean/std across windows.
    Surfaces local spoof artifacts averaged out in clip-level features.
    """
    _windows, win_frame_err, win_entropy, win_repeat = _compute_window_series(
        error, codes, hop_length, sample_rate, window_ms
    )
    if not _windows:
        return {}

    features: dict[str, float] = {}
    for name, values in (
        ("win_frame_err", np.array(win_frame_err, dtype=np.float64)),
        ("win_token_entropy", np.array(win_entropy, dtype=np.float64)),
        ("win_token_repeat", np.array(win_repeat, dtype=np.float64)),
    ):
        if values.size == 0:
            continue
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_std"] = float(values.std())
        features[f"{name}_max"] = float(values.max())

    return features
