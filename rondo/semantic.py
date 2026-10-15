"""The six independent semantic checks."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence as TypingSequence

from .numeric import (
    DEFAULT_NUMERIC_POLICY,
    MismatchKind,
    NumericPolicy,
    compare_float,
    float_bits,
)

__all__ = [
    "CHECKS",
    "SemanticProbe",
    "SemanticVerdict",
    "evaluate_probe",
    "check_precision",
    "check_repeat_stability",
    "check_interning",
    "check_identity",
    "check_consensus",
    "check_bounds",
]

CHECKS = (
    "precision",
    "repeat_stability",
    "interning",
    "identity",
    "consensus",
    "bounds",
)


@dataclass(frozen=True)
class SemanticProbe:
    """One declared property to check, with the numbers already collected."""

    kind: str
    step_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    label: str = "output"

    def __post_init__(self) -> None:
        if self.kind not in CHECKS:
            raise ValueError(f"unknown semantic check {self.kind!r}")


@dataclass(frozen=True)
class SemanticVerdict:
    kind: str
    step_id: str
    label: str
    consistent: bool
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def obligation(self) -> str:
        """Stable identity of the violated obligation (for symptom pinning)."""
        return f"{self.kind}:{self.label}"


def evaluate_probe(
    probe: SemanticProbe, policy: NumericPolicy = DEFAULT_NUMERIC_POLICY
) -> SemanticVerdict:
    dispatch = {
        "precision": check_precision,
        "repeat_stability": check_repeat_stability,
        "interning": check_interning,
        "identity": check_identity,
        "consensus": check_consensus,
        "bounds": check_bounds,
    }
    function = dispatch[probe.kind]
    kwargs = dict(probe.payload)
    kwargs.setdefault("label", probe.label)
    if probe.kind != "precision":
        kwargs.setdefault("policy", policy)
    return function(**kwargs)


# 1. Numerical precision
def check_precision(
    observed: Any,
    reference: Any,
    *,
    label: str = "output",
    min_relative_error: float = 1e-9,
) -> SemanticVerdict:
    """Assert a ``float64`` computation was not silently done in ``float32``."""
    observed_values = _as_floats(observed)
    reference_values = _as_floats(reference)
    if len(observed_values) != len(reference_values):
        return _not_applicable("precision", label, "operand shapes differ")
    for got, want in zip(observed_values, reference_values):
        if math.isnan(got) and math.isnan(want):
            continue
        if math.isnan(got) or math.isnan(want) or math.isinf(got) or math.isinf(want):
            continue
        if got == want:
            continue
        relative = _relative_error(got, want)
        if relative < min_relative_error:
            continue
        if _float32_representable(got) and not _float32_representable(want):
            return SemanticVerdict(
                kind="precision",
                step_id="",
                label=label,
                consistent=False,
                reason="float32-precision result for a float64 computation",
                detail={"observed": got, "reference": want, "relative_error": relative},
            )
    return SemanticVerdict(kind="precision", step_id="", label=label, consistent=True)


def _float32_representable(value: float) -> bool:
    if math.isnan(value) or math.isinf(value):
        return True
    return struct.unpack("<f", struct.pack("<f", value))[0] == value


def _relative_error(got: float, want: float) -> float:
    if want == 0.0:
        return math.inf if got != 0.0 else 0.0
    return abs(got - want) / abs(want)


# 2. Repeat stability
def check_repeat_stability(
    samples: TypingSequence[Any],
    *,
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
    label: str = "output",
) -> SemanticVerdict:
    """Assert the same invocation, repeated, gives identical results."""
    values = _as_floats(samples)
    if len(values) < 2:
        return SemanticVerdict(kind="repeat_stability", step_id="", label=label, consistent=True)
    baseline = values[0]
    divergences: list[dict[str, Any]] = []
    for index, value in enumerate(values[1:], start=1):
        reason = compare_float(baseline, value, "float64", policy)
        if reason is not None:
            divergences.append({"index": index, "reason": reason, "value": value})
    if divergences:
        return SemanticVerdict(
            kind="repeat_stability",
            step_id="",
            label=label,
            consistent=False,
            reason="deterministic invocation was not repeat-stable",
            detail={"baseline": baseline, "divergences": divergences},
        )
    return SemanticVerdict(kind="repeat_stability", step_id="", label=label, consistent=True)


# 3. Interning / identical scalars
def check_interning(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    baseline: str = "(none)",
    label: str = "output",
    policy: NumericPolicy | None = None,
) -> SemanticVerdict:
    """Assert a value built after an equal-but-distinct value is not aliased."""
    del policy  # bit-exactness, not tolerance
    mismatches: list[dict[str, Any]] = []
    for requested, lanes in observations.items():
        if baseline not in lanes:
            raise ValueError(f"baseline lane {baseline!r} missing for {requested!r}")
        expected = _as_float(lanes[baseline])
        for lane, value in lanes.items():
            if lane == baseline:
                continue
            actual = _as_float(value)
            if float_bits(expected, "float64") != float_bits(actual, "float64"):
                if math.isnan(expected) and math.isnan(actual):
                    reason = MismatchKind.NAN_PAYLOAD
                elif expected == actual and expected == 0.0:
                    reason = MismatchKind.SIGNED_ZERO
                else:
                    reason = MismatchKind.EXACT
                mismatches.append(
                    {"requested": requested, "lane": lane, "reason": reason, "value": actual}
                )
    if mismatches:
        return SemanticVerdict(
            kind="interning",
            step_id="",
            label=label,
            consistent=False,
            reason="an equal-but-distinct value changed the observed bit pattern",
            detail={"mismatches": mismatches},
        )
    return SemanticVerdict(kind="interning", step_id="", label=label, consistent=True)


# 4. Mathematical identity
def check_identity(
    left: Any,
    right: Any,
    *,
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
    label: str = "output",
) -> SemanticVerdict:
    """Assert a declared identity ``left == right`` holds for the observation."""
    left_values = _as_floats(left)
    right_values = _as_floats(right)
    if len(left_values) != len(right_values):
        return _not_applicable("identity", label, "operand shapes differ")
    for index, (a, b) in enumerate(zip(left_values, right_values)):
        reason = compare_float(a, b, "float64", policy)
        if reason is not None:
            return SemanticVerdict(
                kind="identity",
                step_id="",
                label=label,
                consistent=False,
                reason=f"declared identity violated ({reason})",
                detail={"index": index, "left": a, "right": b},
            )
    return SemanticVerdict(kind="identity", step_id="", label=label, consistent=True)


# 5. Consensus among implementations
def check_consensus(
    observations: Mapping[str, Any],
    *,
    reference: Any | None = None,
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
    label: str = "value",
) -> SemanticVerdict:
    """Assert at least three implementations agree; report an outlier."""
    lanes = {name: _as_float(value) for name, value in observations.items()}
    if len(lanes) < 2:
        raise ValueError("consensus needs at least two labelled lanes")

    if reference is None:
        groups = _policy_groups(list(lanes.items()), policy)
        largest = max(groups, key=len)
        if sum(1 for group in groups if len(group) == len(largest)) > 1:
            return SemanticVerdict(
                kind="consensus",
                step_id="",
                label=label,
                consistent=False,
                reason="no majority among implementations",
                detail={"groups": [list(group) for group in groups]},
            )
        outliers = [
            {"lane": name, "value": lanes[name]}
            for group in groups
            if group is not largest
            for name in group
        ]
        if outliers:
            return SemanticVerdict(
                kind="consensus",
                step_id="",
                label=label,
                consistent=False,
                reason="implementation outlier",
                detail={"outliers": outliers, "majority": largest[0]},
            )
        return SemanticVerdict(kind="consensus", step_id="", label=label, consistent=True)

    arbiter = _as_float(reference)
    outliers = []
    for name, value in lanes.items():
        if compare_float(arbiter, value, "float64", policy) is not None:
            outliers.append({"lane": name, "value": value})
    if outliers:
        return SemanticVerdict(
            kind="consensus",
            step_id="",
            label=label,
            consistent=False,
            reason="implementations disagree with the reference",
            detail={"outliers": outliers, "reference": arbiter},
        )
    return SemanticVerdict(kind="consensus", step_id="", label=label, consistent=True)


def _policy_groups(
    lanes: list[tuple[str, float]], policy: NumericPolicy
) -> list[list[str]]:
    groups: list[list[str]] = []
    representatives: list[float] = []
    for name, value in lanes:
        for index, representative in enumerate(representatives):
            if compare_float(representative, value, "float64", policy) is None:
                groups[index].append(name)
                break
        else:
            groups.append([name])
            representatives.append(value)
    return groups


# 6. Mathematical bounds
def check_bounds(
    observations: Any,
    *,
    lower: float | None = None,
    upper: float | None = None,
    tolerance: float = 0.0,
    policy: NumericPolicy | None = None,
    label: str = "output",
) -> SemanticVerdict:
    """Assert every observed value lies inside its declared mathematical range."""
    del policy
    if lower is None and upper is None:
        raise ValueError("check_bounds needs at least one bound")
    cases: Mapping[str, Any]
    if isinstance(observations, Mapping):
        cases = observations
    else:
        cases = {str(index): value for index, value in enumerate(observations)}

    violations: list[dict[str, Any]] = []
    skipped: list[str] = []
    for case, raw in cases.items():
        value = _as_float(raw)
        if math.isnan(value):
            skipped.append(case)
            continue
        below = lower is not None and value < lower - tolerance
        above = upper is not None and value > upper + tolerance
        if below or above:
            violations.append({"case": case, "value": value})
    if violations:
        return SemanticVerdict(
            kind="bounds",
            step_id="",
            label=label,
            consistent=False,
            reason="result outside its declared range",
            detail={"violations": violations, "skipped": skipped},
        )
    return SemanticVerdict(kind="bounds", step_id="", label=label, consistent=True)


# Helpers
def _as_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"cannot interpret {value!r} as a scalar")
        value = value[0]
    return float(value)


def _as_floats(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _not_applicable(kind: str, label: str, reason: str) -> SemanticVerdict:
    """The check's preconditions did not hold, so it makes no claim."""
    return SemanticVerdict(
        kind=kind,
        step_id="",
        label=label,
        consistent=True,
        reason=f"not applicable: {reason}",
    )
