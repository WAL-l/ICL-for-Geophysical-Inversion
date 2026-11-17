from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ICL.config import InversionConfig
from ICL.misfit import MisfitEvaluation
_INV_CONFIG = InversionConfig()
DEFAULT_HM_TIMES = _INV_CONFIG.times


@dataclass(slots=True)
class Sounding:
    response: np.ndarray
    height: float
    times: np.ndarray
    resistivity: np.ndarray | None = None
    data_std: np.ndarray | None = None
    sounding_id: str = "query"


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    model_code: np.ndarray
    resistivity: np.ndarray
    predicted_response: np.ndarray
    misfit: MisfitEvaluation

    @property
    def rms(self) -> float:
        return self.misfit.rms


@dataclass(frozen=True, slots=True)
class ConstraintModel:

    layer_group_ids: np.ndarray
    constraint_interface_depths: np.ndarray
    mapped_interface_depths: np.ndarray

    @classmethod
    def from_constraint_model(
            cls,
            thicknesses,
            constraint_interfaces,
    ):
        aem_interfaces = np.cumsum(thicknesses)

        boundary_indices = np.abs(aem_interfaces[:, None] - constraint_interfaces).argmin(axis=0)
        mapped_depths = aem_interfaces[boundary_indices].copy()

        group_ids = np.zeros(thicknesses.size + 1, dtype=int)
        group_ids[boundary_indices + 1] = 1
        group_ids = np.cumsum(group_ids)
        for values in (group_ids, constraint_interfaces, mapped_depths):
            values.setflags(write=False)
        return cls(group_ids, constraint_interfaces, mapped_depths)

    @property
    def n_groups(self) -> int:
        return int(self.layer_group_ids[-1]) + 1

    @property
    def interface_offsets(self) -> np.ndarray:
        return self.mapped_interface_depths - self.constraint_interface_depths

    def expand(self, group_codes) -> np.ndarray:
        try:
            codes = np.asarray(group_codes, dtype=float)
        except OverflowError as error:
            raise ValueError("rho_code contains a number outside floating-point range") from error
        if codes.ndim != 1 or codes.size != self.n_groups:
            raise ValueError(f"rho_code must contain exactly {self.n_groups} group values")
        if not np.all(np.isfinite(codes)):
            raise ValueError("rho_code must contain only finite numbers")
        # A small explicit contrast preserves each prescribed structural
        # boundary, including when the model is later shown on a log scale.
        contrast = np.abs(np.diff(codes))
        if np.any(contrast == 0) or np.any(contrast < self.min_log_contrast * (1 - 1e-12)):
            raise ValueError(f"adjacent group rho_code values must differ by at least {self.min_log_contrast}")

        return codes[self.layer_group_ids]


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    prompt: str | None = None
    candidate: EvaluatedCandidate | None = None
    improved: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class InversionResult:
    query_id: str
    height_m: float
    best: EvaluatedCandidate
    history: list[IterationRecord]
    stop_reason: str
    constraint_model: ConstraintModel | None = None


class InversionError(RuntimeError):
    def __init__(self, message: str, history: list[IterationRecord]):
        super().__init__(message)
        self.history = history


class AEMDataset:
    def __init__(
            self,
            responses,
            heights,
            resistivities,
            times,
    ) -> None:
        self.responses = responses
        self.heights = heights
        self.resistivities = resistivities
        self.times = times

    @classmethod
    def from_npy(
            cls,
            data_path: str | Path,
            height_path: str | Path,
            rho_path: str | Path,
            times: np.ndarray | None = None,
            mmap_mode: str | None = "r",
    ):
        responses = np.load(Path(data_path), mmap_mode=mmap_mode)
        heights = np.load(Path(height_path), mmap_mode=mmap_mode)
        resistivities = np.load(Path(rho_path), mmap_mode=mmap_mode)
        if times is None:
            times = DEFAULT_HM_TIMES
        return cls(responses, heights, resistivities, times)

    def __len__(self):
        return int(self.responses.shape[0])

    def sounding(self, index, include_model):
        index = index
        rho = self.resistivities[index] if include_model else None
        return Sounding(
            response=self.responses[index],
            height=self.heights[index],
            times=self.times,
            resistivity=None if rho is None else rho,
            sounding_id=f"sample_{index}",
        )
