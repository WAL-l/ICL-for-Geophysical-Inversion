from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class MisfitEvaluation:
    residual: np.ndarray
    normalized_residual: np.ndarray
    data_std: np.ndarray
    rms: float
    block_rms: list[float | None]

def evaluate_misfit(
    observed,
    predicted,
    *,
    relative_error: float = 0.05,
    noise_floor: float | None = None,
    data_std=None,
) -> MisfitEvaluation:

    if data_std is not None:
        std = data_std
    else:
        if not np.isfinite(relative_error) or relative_error < 0:
            raise ValueError("relative_error must be finite and nonnegative")
        floor = float(np.min(observed)) * 1e-3 if noise_floor is None else noise_floor
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("noise_floor must be finite and nonnegative")
        std = relative_error * np.abs(observed) + floor

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        residual = predicted - observed
        normalized = residual / std
        rms = float(np.sqrt(np.mean(normalized ** 2)))
    if not np.isfinite(rms):
        raise ValueError("nonfinite normalized residual/RMS; check response units and error scale")
    blocks = np.array_split(normalized, 3)
    block_rms = [float(np.sqrt(np.mean(block ** 2))) if block.size else None
                 for block in blocks]
    return MisfitEvaluation(residual, normalized, std.copy(), rms, block_rms)
