"""Real-time search over an AEM library organized during forward generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .library import AEMSampleLibrary


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    top_k: int = 10
    candidate_pool_size: int = 8_000
    rerank_pool_size: int = 8_000
    max_height_delta_m: float | None = 20.0
    allow_height_fallback: bool = True
    height_scale_m: float = 10.0
    response_weight: float = 0.80
    height_weight: float = 0.20
    amplitude_weight: float = 0.25
    shape_weight: float = 0.55
    slope_weight: float = 0.20
    diversity_weight: float = 0.03
    exact_search_threshold: int = 5_000

@dataclass(frozen=True, slots=True)
class RetrievedExample:
    index: int
    height_m: float
    response: np.ndarray
    model: np.ndarray | None


class HybridAEMRetriever:
    """Load posting lists directly; never builds or fits an index at query time."""

    def __init__(
        self,
        library: AEMSampleLibrary,
        config: RetrieverConfig | None = None,
    ):

        self.library = library
        self.config = config or RetrieverConfig()

    def _cell_height_distance(self, query_height: float) -> np.ndarray:
        width = self.library.layout_config.height_bin_width_m
        origin = self.library.layout_config.height_origin_m
        lower = origin + self.library.cell_keys.astype(np.float64) * width
        upper = lower + width
        return np.maximum(
            np.maximum(lower - query_height, query_height - upper), 0.0
        )

    def _ordered_cells(
        self,
        query_height: float,
        apply_height_gate: bool,
    ) -> np.ndarray:
        height_distance = self._cell_height_distance(query_height)
        allowed = np.ones(self.library.n_cells, dtype=bool)
        if apply_height_gate and self.config.max_height_delta_m is not None:
            allowed &= height_distance <= self.config.max_height_delta_m
        cell_indices = np.flatnonzero(allowed)
        if cell_indices.size == 0:
            return cell_indices

        distance = (
            height_distance[cell_indices] / self.config.height_scale_m
        )
        return cell_indices[np.lexsort((cell_indices, distance))]

    def _gather_ordered_cells(
        self, ordered_cells: np.ndarray, target: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sizes = (
            self.library.cell_offsets[ordered_cells + 1]
            - self.library.cell_offsets[ordered_cells]
        )
        cumulative = np.cumsum(sizes)
        selected_count = min(
            ordered_cells.size,
            int(np.searchsorted(cumulative, target, side="left")) + 1,
        )
        selected_cells = ordered_cells[:selected_count]
        candidates = np.concatenate(
            [self.library.cell_indices(int(cell)) for cell in selected_cells]
        )
        return candidates, selected_cells


    def _indexed_candidates(
        self,
        query_height: float,
        required: int,
    ):
        ordered_cells = self._ordered_cells(query_height, apply_height_gate=True)
        target = max(required, self.config.candidate_pool_size)
        candidates, selected_cells = self._gather_ordered_cells(ordered_cells, target)

        return candidates, selected_cells

    def _exact_candidates(
        self,
        query_height: float,
    ):
        indices = np.arange(len(self.library.responses), dtype=np.int64)
        max_delta = self.config.max_height_delta_m
        if max_delta is not None:
            indices = indices[
                np.abs(self.library.heights - query_height) <= max_delta
            ]
        return indices

    def _component_distances(
        self,
        candidates: np.ndarray,
        query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        curves = self.library.response_curves[candidates]
        n_channels_target = query.size
        curves = curves[:, :n_channels_target]

        candidate_mean = np.mean(curves, axis=1)
        query_mean = float(np.mean(query))
        shape = np.sqrt(
            np.mean(
                (
                    curves
                    - candidate_mean[:, None]
                    - (query - query_mean)[None, :]
                )
                ** 2,
                axis=1,
            )
        )
        slope = (
            np.sqrt(
                np.mean(
                    (np.diff(curves, axis=1) - np.diff(query)[None, :]) ** 2,
                    axis=1,
                )
            )
            if query.size > 1
            else np.zeros(candidates.size)
        )
        amplitude = np.abs(candidate_mean - query_mean)
        component_sum = (
            self.config.amplitude_weight
            + self.config.shape_weight
            + self.config.slope_weight
        )
        response = (
            self.config.amplitude_weight * amplitude
            + self.config.shape_weight * shape
            + self.config.slope_weight * slope
        ) / component_sum
        return amplitude, shape, slope, response

    def search(
        self,
        query: np.ndarray | object,
        height_m: float | None = None,
        top_k: int | None = None,
        exact: bool = False,
    ):
        requested = self.config.top_k if top_k is None else int(top_k)

        use_exact = exact or len(self.library.responses) <= self.config.exact_search_threshold
        if use_exact:
            candidates = self._exact_candidates(height_m)
        else:
            candidates, selected_cells = self._indexed_candidates(height_m, requested)

        height_delta = np.abs(self.library.heights[candidates]- float(height_m))
        coarse_total = height_delta / self.config.height_scale_m

        rerank_count = (
            candidates.size
            if use_exact
            else min(candidates.size, max(requested, self.config.rerank_pool_size))
        )
        if rerank_count < candidates.size:
            local = np.argpartition(coarse_total, rerank_count - 1)[:rerank_count]
            candidates = candidates[local]

        amplitude, shape, slope, response_distance = self._component_distances(
            candidates, query
        )

        normalized_height = np.abs(self.library.heights[candidates]- float(height_m)) / self.config.height_scale_m

        total = (
            self.config.response_weight * response_distance
            + self.config.height_weight * normalized_height
        )

        selected = np.lexsort((candidates, total))[:top_k]

        examples: list[RetrievedExample] = []
        for local in selected:
            index = int(candidates[local])
            examples.append(
                RetrievedExample(
                    index=index,
                    height_m=float(self.library.heights[index]),
                    response=np.asarray(self.library.responses[index]),
                    model=self.library.models[index]
                    )
            )
        return examples

FastAEMRetriever = HybridAEMRetriever
