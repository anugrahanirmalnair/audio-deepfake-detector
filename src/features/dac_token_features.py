"""Handcrafted features from DAC discrete token streams."""

import re
from collections import Counter

import numpy as np

VOCAB_SIZE = 1024


def shannon_entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def codebook_entropy(tokens_1d: np.ndarray) -> float:
    counts = np.bincount(
        tokens_1d.astype(np.int64),
        minlength=VOCAB_SIZE,
    )
    return shannon_entropy(counts)


def codebook_repeat_rate(tokens_1d: np.ndarray) -> float:
    if tokens_1d.size < 2:
        return 0.0
    return float((tokens_1d[1:] == tokens_1d[:-1]).mean())


def codebook_histogram_features(tokens_1d: np.ndarray) -> tuple[float, float, float]:
    """Vocabulary usage, top-token mass, effective vocabulary size."""
    if tokens_1d.size == 0:
        return 0.0, 0.0, 0.0

    counts = np.bincount(
        tokens_1d.astype(np.int64),
        minlength=VOCAB_SIZE,
    )
    probs = counts[counts > 0] / counts.sum()
    entropy = shannon_entropy(counts)
    vocab_usage = float(len(probs) / tokens_1d.size)
    top1_mass = float(probs.max())
    eff_vocab = float(2**entropy)
    return vocab_usage, top1_mass, eff_vocab


def codebook_transition_features(
    tokens_1d: np.ndarray,
    vocab_size: int = VOCAB_SIZE,
) -> tuple[float, float, float]:
    """Bigram entropy, mean outgoing entropy, and peak transition probability."""
    if tokens_1d.size < 2:
        return 0.0, 0.0, 0.0

    from_t = tokens_1d[:-1].astype(np.int64)
    to_t = tokens_1d[1:].astype(np.int64)

    pair_ids = from_t * vocab_size + to_t
    _, counts = np.unique(pair_ids, return_counts=True)
    probs = counts / counts.sum()
    trans_entropy = shannon_entropy(counts)

    row_entropies = []
    for source in np.unique(from_t):
        outgoing = to_t[from_t == source]
        outgoing_counts = np.bincount(outgoing, minlength=vocab_size)
        row_entropies.append(shannon_entropy(outgoing_counts))

    mean_row_entropy = float(np.mean(row_entropies)) if row_entropies else 0.0
    max_trans_prob = float(probs.max())

    return trans_entropy, mean_row_entropy, max_trans_prob


def extract_token_features(
    tokens: np.ndarray,
    include_histogram: bool = True,
) -> dict[str, float]:
    """
    tokens: (n_codebooks, time) integer token IDs from DAC encode output.
    """
    tokens = np.asarray(tokens)
    n_codebooks, _ = tokens.shape
    features: dict[str, float] = {}

    flat_counts = Counter(tokens.flatten())
    total = sum(flat_counts.values())
    probs = np.array([c / total for c in flat_counts.values()])
    features["entropy"] = float(-np.sum(probs * np.log2(probs + 1e-12)))

    if tokens.shape[1] >= 2:
        features["repeat_rate"] = float(
            (tokens[:, 1:] == tokens[:, :-1]).mean()
        )
    else:
        features["repeat_rate"] = 0.0

    for cb in range(n_codebooks):
        cb_tokens = tokens[cb]
        prefix = f"cb{cb:02d}"
        features[f"{prefix}_entropy"] = codebook_entropy(cb_tokens)
        features[f"{prefix}_repeat_rate"] = codebook_repeat_rate(cb_tokens)

        trans_entropy, mean_row_entropy, max_trans_prob = (
            codebook_transition_features(cb_tokens)
        )
        features[f"{prefix}_trans_entropy"] = trans_entropy
        features[f"{prefix}_row_entropy"] = mean_row_entropy
        features[f"{prefix}_max_trans_prob"] = max_trans_prob

        if include_histogram:
            vocab_usage, top1_mass, eff_vocab = codebook_histogram_features(
                cb_tokens
            )
            features[f"{prefix}_vocab_usage"] = vocab_usage
            features[f"{prefix}_top1_mass"] = top1_mass
            features[f"{prefix}_eff_vocab"] = eff_vocab

    return features


def infer_n_codebooks(columns) -> int:
    indices = []
    for col in columns:
        match = re.match(r"cb(\d+)_entropy$", col)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 32


def _per_codebook_suffixes(include_trans: bool, include_histogram: bool) -> list[str]:
    suffixes = ["entropy", "repeat_rate"]
    if include_trans:
        suffixes.extend(
            ["trans_entropy", "row_entropy", "max_trans_prob"]
        )
    if include_histogram:
        suffixes.extend(["vocab_usage", "top1_mass", "eff_vocab"])
    return suffixes


def feature_columns_for_set(feature_set: str, n_codebooks: int = 32) -> list[str]:
    if feature_set == "baseline":
        return ["entropy", "repeat_rate"]
    if feature_set == "per_codebook":
        cols = ["entropy", "repeat_rate"]
        for cb in range(n_codebooks):
            prefix = f"cb{cb:02d}"
            cols.extend([f"{prefix}_entropy", f"{prefix}_repeat_rate"])
        return cols
    if feature_set in ("per_codebook_trans", "token_codec"):
        cols = ["entropy", "repeat_rate"]
        suffixes = _per_codebook_suffixes(True, False)
        for cb in range(n_codebooks):
            prefix = f"cb{cb:02d}"
            cols.extend(f"{prefix}_{s}" for s in suffixes)
        return cols
    if feature_set == "token_full":
        cols = ["entropy", "repeat_rate"]
        suffixes = _per_codebook_suffixes(True, True)
        for cb in range(n_codebooks):
            prefix = f"cb{cb:02d}"
            cols.extend(f"{prefix}_{s}" for s in suffixes)
        return cols
    raise ValueError(f"Unknown feature set: {feature_set}")


def token_feature_columns(columns) -> list[str]:
    """All token-stream columns present in a feature dataframe."""
    skip = {"path", "label"}
    return [c for c in columns if c not in skip and not c.startswith(("err_", "frame_err_", "z_", "latent_", "win_"))]


def recon_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(("err_", "frame_err_"))]


def latent_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(("z_", "latent_"))]


def window_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("win_")]


METADATA_COLUMNS = frozenset(
    {"path", "label", "speaker_id", "file_id", "attack_id"}
)
