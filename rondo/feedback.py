"""Compilation-behavior feedback: keys, counts and the sampling score."""

from __future__ import annotations

from dataclasses import dataclass, field

from .domain import BehaviorKey, Trace

__all__ = ["BehaviorFeedback", "RewardConfig"]


@dataclass(frozen=True)
class RewardConfig:
    """Which score the sampler maximises."""

    mode: str = "rarity"  # "rarity" (paper S_t) | "gain" (density)
    lambda_: float = 1.0
    length_exponent: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("rarity", "gain"):
            raise ValueError("mode must be 'rarity' or 'gain'")


@dataclass
class BehaviorFeedback:
    """Running key counts plus the frozen snapshot used for one decision."""

    config: RewardConfig = field(default_factory=RewardConfig)
    counts: dict[str, int] = field(default_factory=dict)
    frozen_counts: dict[str, int] = field(default_factory=dict)
    seen_cells: set[tuple] = field(default_factory=set)
    frozen_cells: set[tuple] = field(default_factory=set)
    seen_divergences: set[str] = field(default_factory=set)
    frozen_divergences: set[str] = field(default_factory=set)
    observed_keys: set[str] = field(default_factory=set)

    # observation
    def observe(self, trace: Trace) -> None:
        """Fold one evaluated trace into the running counts and coverage."""
        for event in trace.events:
            if event.key is not None:
                self.counts[event.key.text] = self.counts.get(event.key.text, 0) + 1
                self.observed_keys.add(event.key.text)
                self.seen_cells.add(_cell(event.key))
        for observation in trace.observations:
            for key in observation.behavior_keys:
                if key not in self.observed_keys:
                    # Keys the adapter only reports as strings still count.
                    self.counts[key] = self.counts.get(key, 0) + 1
                    self.observed_keys.add(key)
        self.seen_divergences.update(trace.divergences)

    def freeze(self) -> None:
        """Freeze the counts for one acceptance decision."""
        self.frozen_counts = dict(self.counts)
        self.frozen_cells = set(self.seen_cells)
        self.frozen_divergences = set(self.seen_divergences)

    # scores
    def rarity(self, keys: set[str]) -> float:
        """``S_t(X)`` over the *distinct* keys, using the frozen counts."""
        return sum(1.0 / (1.0 + self.frozen_counts.get(key, 0)) for key in keys)

    def components(self, trace: Trace) -> tuple[float, float]:
        """Return ``(coverage_term, divergence_term)`` of the configured score."""
        if not self.config.enabled:
            return 0.0, 0.0
        keys = set(trace.keys)
        if self.config.mode == "rarity":
            return self.rarity(keys), 0.0
        length = max(1, len(trace.observations))
        cells = {_cell(event.key) for event in trace.events if event.key is not None}
        divergences = set(trace.divergences)
        coverage = len(cells) / (length ** self.config.length_exponent)
        divergence = len(divergences) / length
        return coverage, divergence

    def score(self, trace: Trace) -> float:
        if not self.config.enabled:
            return 0.0
        coverage, divergence = self.components(trace)
        if self.config.mode == "rarity":
            return coverage
        return coverage + self.config.lambda_ * divergence

    # diagnostics
    def novel_keys(self, trace: Trace) -> set[str]:
        return {key for key in trace.keys if key not in self.observed_keys}

    def describe(self) -> dict[str, int]:
        return {
            "distinct_keys": len(self.counts),
            "cells": len(self.seen_cells),
            "divergences": len(self.seen_divergences),
        }


def _cell(key: BehaviorKey) -> tuple:
    """The coarse coverage cell: everything except mode and value effect."""
    return (
        key.operator,
        key.lowering,
        key.specialization,
        key.dtype_class,
        key.decision.value,
    )
