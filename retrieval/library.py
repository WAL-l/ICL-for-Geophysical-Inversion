"""Generation-time organized, mmap-readable AEM demonstration library."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import DeterministicResponseTransform


@dataclass(frozen=True, slots=True)
class LibraryLayoutConfig:
    """Fixed routing layout known before the first forward simulation."""

    height_bin_width_m: float = 2.0
    height_origin_m: float = 0.0


class AEMLibraryWriter:
    """Stream forward results directly into their final retrieval layout.

    Payload arrays are preallocated and joint-cell posting lists are maintained
    while responses are generated. Finalization only serializes those lists;
    it does not fit features, cluster samples, or scan response space again.
    """

    def __init__(
        self,
        path: str | Path,
        capacity: int,
        n_model_parameters: int | None,
        config: LibraryLayoutConfig | None = None,
    ) -> None:

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.config = config or LibraryLayoutConfig()
        self.transform = DeterministicResponseTransform()
        self.n_model_parameters = n_model_parameters
        self.count = 0
        self._finalized = False
        self._cell_members: dict[int, list[int]] = {}
        self._create_arrays()

    def _create_arrays(self) -> None:
        channels = self.transform.n_channels
        self.responses = np.lib.format.open_memmap(
            self.path / "responses.npy",
            mode="w+",
            dtype=np.float32,
            shape=(self.capacity, channels),
        )
        self.heights = np.lib.format.open_memmap(
            self.path / "heights.npy",
            mode="w+",
            dtype=np.float32,
            shape=(self.capacity,),
        )
        self.response_curves = np.lib.format.open_memmap(
            self.path / "response_curves.npy",
            mode="w+",
            dtype=np.float32,
            shape=(self.capacity, channels),
        )

        self.models = np.lib.format.open_memmap(
                self.path / "models.npy",
                mode="w+",
                dtype=np.float32,
                shape=(self.capacity, self.n_model_parameters),
            )


    def _flush_payload(self) -> None:
        for array in (
            self.responses,
            self.heights,
            self.response_curves,
            self.models,
        ):
            if array is not None:
                array.flush()

    def add(self,response,height_m,model):
        """Route and persist one response immediately after ``forward`` returns."""

        index = self.count
        curve = self.transform.transform_curves(response)
        height_code = int(
            np.floor(
                (float(height_m) - self.config.height_origin_m)
                / self.config.height_bin_width_m
            )
        )

        self.responses[index] = response
        self.heights[index] = height_m
        self.response_curves[index] = curve
        self.models[index] = model

        self._cell_members.setdefault(height_code, []).append(index)

        self.count += 1
        return index

    def finalize(self, allow_partial: bool = False) -> Path:
        """Seal posting lists already accumulated during generation."""

        if self._finalized:
            return self.path
        if self.count != self.capacity and not allow_partial:
            raise ValueError(
                f"generated {self.count} of {self.capacity} samples; use "
                "allow_partial=True only when the shorter library is intentional"
            )
        keys = sorted(self._cell_members)
        cell_keys = np.asarray(keys)
        counts = np.asarray(
            [len(self._cell_members[key]) for key in keys], dtype=np.int64
        )
        cell_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        sorted_indices = np.concatenate(
            [np.asarray(self._cell_members[key], dtype=np.int64) for key in keys]
        )

        np.save(self.path / "cell_keys.npy", cell_keys, allow_pickle=False)
        np.save(self.path / "cell_offsets.npy", cell_offsets, allow_pickle=False)
        np.save(self.path / "sorted_indices.npy", sorted_indices, allow_pickle=False)
        self._flush_payload()
        self._finalized = True
        return self.path

    def close(self) -> None:
        self.release()

    def release(self) -> None:
        """Release mmap handles, primarily for clean shutdown on Windows."""

        for array in (
            self.responses,
            self.heights,
            self.response_curves,
            self.models,
        ):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "AEMLibraryWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


@dataclass(slots=True)
class AEMSampleLibrary:
    """Read-only view of a library already organized during generation."""

    responses: np.ndarray
    heights: np.ndarray
    response_curves: np.ndarray
    cell_keys: np.ndarray
    sorted_indices: np.ndarray
    cell_offsets: np.ndarray
    layout_config: LibraryLayoutConfig
    models: np.ndarray | None = None


    @property
    def n_cells(self) -> int:
        return int(self.cell_keys.shape[0])

    def cell_indices(self, cell_index: int) -> np.ndarray:
        start, stop = self.cell_offsets[cell_index : cell_index + 2]
        return self.sorted_indices[int(start) : int(stop)]

    def close(self) -> None:
        """Release read-only mmap handles during service shutdown."""

        for array in (
            self.responses,
            self.heights,
            self.response_curves,
            self.cell_keys,
            self.sorted_indices,
            self.cell_offsets,
            self.models,
        ):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = "r") -> "AEMSampleLibrary":
        root = Path(path)

        def load(name):
            return np.load(root / f"{name}.npy")

        config = LibraryLayoutConfig()
        return cls(
            responses=load("responses"),
            heights=load("heights"),
            response_curves=load("response_curves"),
            cell_keys=load("cell_keys"),
            sorted_indices=load("sorted_indices"),
            cell_offsets=load("cell_offsets"),
            models=load("models"),
            layout_config=config,
        )


LibraryBuildConfig = LibraryLayoutConfig
