from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class InversionConfig:

    rho_min: float = 1.0
    rho_max: float = 2000.0
    max_iterations: int = 30  # Refinement calls after the initial proposal.
    patience: int | None = 30  # None disables stagnation stopping.
    min_improvement: float = 1e-3  # Absolute reduction in normalized RMS.
    max_delta_code: int = 100
    relative_error: float = 0.05
    noise_floor: float | None = None
    target_rms: float = 0.05
    # ========== Fixed Acquisition/Model Constants ==========
    n_layer: int = 30
    model_code_max: int = 1000
    response_code_max: int = 10000
    times: np.ndarray = np.asarray(
        [
            3.76e-4,
            3.83e-4,
            3.92e-4,
            4.04e-4,
            4.18e-4,
            4.36e-4,
            4.59e-4,
            4.87e-4,
            5.22e-4,
            5.68e-4,
            6.25e-4,
            6.97e-4,
            7.88e-4,
            9.02e-4,
            1.046e-3,
            1.227e-3,
            1.456e-3,
            1.744e-3,
            2.108e-3,
            2.567e-3,
            3.145e-3,
            3.864e-3,
            4.744e-3,
            5.821e-3,
            7.139e-3,
            8.753e-3,
            1.073e-2,
        ],
        dtype=float,
    )
    layer_thicknesses: np.ndarray = np.asarray(
        [5, 5.4, 5.8, 6.3, 6.7, 7.3, 7.8, 8.4, 9.1, 9.8, 10.6, 11.4, 12.3, 13.2, 14.3, 15.4, 16.6, 17.9, 19.3, 20.7,
         22.4, 24.1, 26, 28, 30.2, 32.5, 35, 37.8, 40.7])
