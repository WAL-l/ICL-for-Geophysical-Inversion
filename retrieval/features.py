"""Deterministic AEM response features computable during forward generation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ICL.config import InversionConfig

_INV_CONFIG = InversionConfig()

SUMMARY_NAMES = (
    "early_log_amplitude",
    "middle_log_amplitude",
    "late_log_amplitude",
    "early_log_slope",
    "middle_log_slope",
    "late_log_slope",
    "log_dynamic_range",
    "curvature_rms",
)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = x - np.mean(x)
    denominator = max(float(np.dot(centered_x, centered_x)), 1e-12)
    return ((y - np.mean(y, keepdims=True)) @ centered_x) / denominator


@dataclass(frozen=True, slots=True)
class DeterministicResponseTransform:
    """Fixed log/physics transform; it has no ``fit`` operation by design."""

    times: np.ndarray = _INV_CONFIG.times
    response_floor: float = 1e-30

    @property
    def n_channels(self) -> int:
        return int(self.times.size)

    def transform_curves(self, responses):
        return np.log10(np.maximum(responses, self.response_floor))

    def transform_features(self, responses):
        curves = self.transform_curves(responses)
        log_times = np.log10(self.times)
        blocks = np.array_split(np.arange(self.n_channels), 3)
        amplitudes = [np.mean(curves[block]) for block in blocks]
        slopes = [
            _linear_slope(log_times[block], curves[block]) for block in blocks
        ]
        dynamic_range = np.max(curves) - np.min(curves)
        curvature = np.sqrt(
            np.mean(np.diff(curves, n=2) ** 2)
        )
        return np.asarray([*amplitudes, *slopes, dynamic_range, curvature])

    @staticmethod
    def routing_codes(
        values: np.ndarray,
        amplitude_bin_decades: float,
        decay_bin_decades: float,
        slope_bin: float,
    ) -> np.ndarray:
        """Return deterministic ``(amplitude, decay, late_slope)`` cell codes."""

        amplitude = np.mean(values[:3])
        decay = values[0] - values[2]
        late_slope = values[5]
        return np.asarray(
            [
                np.floor(amplitude / amplitude_bin_decades),
                np.floor(decay / decay_bin_decades),
                np.floor(late_slope / slope_bin),
            ]
        )


# Backward-compatible name, now deliberately deterministic rather than fitted.
ResponseTransform = DeterministicResponseTransform
