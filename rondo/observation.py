"""Structured observation comparison and candidate classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .domain import (
    ExceptionInfo,
    Invocation,
    Mode,
    Observation,
    Sequence,
    StateOperation,
    TensorValue,
    Trace,
)
from .numeric import (
    DEFAULT_NUMERIC_POLICY,
    MismatchKind,
    NumericPolicy,
    compare_scalar,
)

__all__ = [
    "Classification",
    "Comparison",
    "Difference",
    "DivergenceClass",
    "Equivalence",
    "classify_traces",
    "compare_observations",
]

MAX_DIFFERENCES = 8


class Equivalence(str, Enum):
    EQUAL = "equal"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


class DivergenceClass(str, Enum):
    """Candidate classes produced by the three-mode oracle."""

    NONE = "none"
    POINT_WISE = "point-wise"
    HISTORY_EXPOSED = "history-exposed"
    HISTORY_DEPENDENT_CANDIDATE = "history-dependent-candidate"
    UNDETERMINED = "undetermined"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class Difference:
    path: str
    kind: str
    left: Any = None
    right: Any = None

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "left": self.left, "right": self.right}


@dataclass(frozen=True)
class Comparison:
    result: Equivalence
    differences: tuple[Difference, ...] = ()

    @property
    def different(self) -> bool:
        return self.result is Equivalence.DIFFERENT


@dataclass(frozen=True)
class Classification:
    """Where and how the compiled modes first disagree with the reference."""

    category: DivergenceClass
    first_step_index: int | None = None
    first_step_id: str | None = None
    failing_modes: tuple[Mode, ...] = ()
    differences: tuple[Difference, ...] = ()
    reason: str = ""
    target_step: Any = None

    @property
    def is_candidate(self) -> bool:
        return self.category in (
            DivergenceClass.POINT_WISE,
            DivergenceClass.HISTORY_EXPOSED,
            DivergenceClass.HISTORY_DEPENDENT_CANDIDATE,
        )


# Value-tree comparison
def compare_observations(
    left: Observation,
    right: Observation,
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
) -> Comparison:
    """Compare two observations of the same step under one numeric policy."""
    if left.step_id != right.step_id or left.step_index != right.step_index:
        return Comparison(
            Equivalence.DIFFERENT,
            (
                Difference(
                    "$",
                    MismatchKind.STRUCTURE,
                    f"{left.step_index}:{left.step_id}",
                    f"{right.step_index}:{right.step_id}",
                ),
            ),
        )

    differences: list[Difference] = []
    if not _exceptions_equal(left.exception, right.exception):
        differences.append(
            Difference(
                "$.exception",
                MismatchKind.EXCEPTION,
                left.exception.to_json() if left.exception else None,
                right.exception.to_json() if right.exception else None,
            )
        )
    # When either side raised, the returned values are not comparable: the
    if left.exception is None and right.exception is None:
        differences.extend(
            _compare_tree(left.output, right.output, "$.output", policy)
        )
        differences.extend(
            _compare_tree(dict(left.state), dict(right.state), "$.state", policy)
        )

    if differences:
        return Comparison(Equivalence.DIFFERENT, tuple(differences[:MAX_DIFFERENCES]))

    if left.unsupported_fields or right.unsupported_fields:
        missing = sorted(set(left.unsupported_fields) | set(right.unsupported_fields))
        return Comparison(
            Equivalence.UNKNOWN,
            (Difference("$", MismatchKind.STRUCTURE, "unsupported fields", missing),),
        )
    return Comparison(Equivalence.EQUAL)


#: Compiler wrappers aliased to the eager failure they wrap.
EXCEPTION_CLASS_ALIASES = {
    "TorchRuntimeError": "RuntimeError",
    "TorchDynamoError": "RuntimeError",
    "BackendCompilerFailed": "RuntimeError",
    "InternalTorchDynamoError": "RuntimeError",
    "UnsupportedAliasMutationException": "RuntimeError",
    "TorchRuntimeErrorWithTraceback": "RuntimeError",
}


def canonical_exception_class(name: str) -> str:
    return EXCEPTION_CLASS_ALIASES.get(name, name)


def _exceptions_equal(left: ExceptionInfo | None, right: ExceptionInfo | None) -> bool:
    if left is None or right is None:
        return left is right
    # Messages are intentionally excluded: they are unstable across compilers.
    return (
        canonical_exception_class(left.class_name)
        == canonical_exception_class(right.class_name)
        and left.phase == right.phase
    )


def _compare_tree(
    left: Any, right: Any, path: str, policy: NumericPolicy
) -> list[Difference]:
    differences: list[Difference] = []

    if isinstance(left, TensorValue) or isinstance(right, TensorValue):
        if not (isinstance(left, TensorValue) and isinstance(right, TensorValue)):
            return [Difference(path, MismatchKind.STRUCTURE, type(left).__name__, type(right).__name__)]
        return _compare_tensor(left, right, path, policy)

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return [Difference(path, MismatchKind.STRUCTURE, type(left).__name__, type(right).__name__)]
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return [
                Difference(
                    path,
                    MismatchKind.STRUCTURE,
                    sorted(map(str, left_keys)),
                    sorted(map(str, right_keys)),
                )
            ]
        for key in sorted(left_keys, key=str):
            differences.extend(
                _compare_tree(left[key], right[key], f"{path}.{key}", policy)
            )
            if len(differences) >= MAX_DIFFERENCES:
                break
        return differences

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
            return [Difference(path, MismatchKind.STRUCTURE, type(left).__name__, type(right).__name__)]
        if len(left) != len(right):
            return [Difference(path, MismatchKind.STRUCTURE, len(left), len(right))]
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(_compare_tree(a, b, f"{path}[{index}]", policy))
            if len(differences) >= MAX_DIFFERENCES:
                break
        return differences

    if type(left) is not type(right):
        return [Difference(path, MismatchKind.SCALAR_TYPE, type(left).__name__, type(right).__name__)]
    if left != left and right != right:  # both NaN
        return []

    dtype = _infer_scalar_dtype(left)
    reason = compare_scalar(left, right, dtype, policy)
    if reason is not None:
        return [Difference(path, reason, left, right)]
    return differences


def _infer_scalar_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, complex):
        return "complex128"
    return "float64"


def _compare_tensor(
    left: TensorValue, right: TensorValue, path: str, policy: NumericPolicy
) -> list[Difference]:
    metadata: list[Difference] = []
    for field_name, kind in (
        ("shape", MismatchKind.SHAPE),
        ("dtype", MismatchKind.DTYPE),
        ("device", MismatchKind.DEVICE),
        ("strides", MismatchKind.STRIDES),
        ("requires_grad", MismatchKind.STRUCTURE),
    ):
        a = getattr(left, field_name)
        b = getattr(right, field_name)
        if a != b:
            metadata.append(Difference(f"{path}.{field_name}", kind, a, b))
    if metadata:
        # Metadata mismatch short-circuits: element comparison would be noise.
        return metadata

    if len(left.data) != len(right.data):
        return [Difference(f"{path}.values", MismatchKind.ELEMENT_COUNT, len(left.data), len(right.data))]

    differences: list[Difference] = []
    floating = left.is_floating
    component = left.dtype
    if left.dtype == "complex64":
        component = "complex64"
    for index, (a, b) in enumerate(zip(left.data, right.data)):
        if floating:
            reason = compare_scalar(a, b, component, policy)
        else:
            reason = None if a == b else MismatchKind.EXACT
        if reason is not None:
            differences.append(Difference(f"{path}.values[{index}]", reason, a, b))
            if len(differences) >= MAX_DIFFERENCES:
                break
    return differences


# Three-mode classification
def classify_traces(
    sequence: Sequence,
    traces: Mapping[Mode, Trace],
    policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
) -> Classification:
    """Classify one sequence from its three mode traces."""
    infrastructure = _infrastructure_reason(sequence, traces)
    if infrastructure is not None:
        return Classification(DivergenceClass.INFRASTRUCTURE, reason=infrastructure)

    reference = traces[Mode.WITHOUT_COMPILATION].observations
    fresh = traces[Mode.COMPILED_WITHOUT_STATE].observations
    persistent = traces[Mode.COMPILED_WITH_STATE].observations

    prior_history = 0
    unknown: tuple[int, Observation, list[Difference]] | None = None

    for index, step in enumerate(sequence.steps):
        ref_obs = reference[index]
        ref_fresh = compare_observations(ref_obs, fresh[index], policy)
        ref_persistent = compare_observations(ref_obs, persistent[index], policy)

        failing: list[Mode] = []
        differences: list[Difference] = []
        if ref_fresh.result is Equivalence.DIFFERENT:
            failing.append(Mode.COMPILED_WITHOUT_STATE)
            differences.extend(ref_fresh.differences)
        if ref_persistent.result is Equivalence.DIFFERENT:
            failing.append(Mode.COMPILED_WITH_STATE)
            differences.extend(ref_persistent.differences)

        if failing:
            if Mode.COMPILED_WITHOUT_STATE in failing:
                category = (
                    DivergenceClass.POINT_WISE
                    if prior_history == 0
                    else DivergenceClass.HISTORY_EXPOSED
                )
                reason = "compilation without retained state disagrees with the reference"
            elif prior_history > 0:
                category = DivergenceClass.HISTORY_DEPENDENT_CANDIDATE
                reason = "only retained compiler state disagrees with the reference"
            else:
                category = DivergenceClass.UNDETERMINED
                reason = "compiled modes disagree before any compiler history exists"
            return Classification(
                category=category,
                first_step_index=index,
                first_step_id=ref_obs.step_id,
                failing_modes=tuple(failing),
                differences=tuple(differences[:MAX_DIFFERENCES]),
                reason=reason,
                target_step=step,
            )

        if unknown is None and (
            ref_fresh.result is Equivalence.UNKNOWN
            or ref_persistent.result is Equivalence.UNKNOWN
        ):
            unknown = (index, ref_obs, list(ref_fresh.differences + ref_persistent.differences))

        if _leaves_compiler_history(step):
            prior_history += 1

    if unknown is not None:
        index, observation, differences = unknown
        return Classification(
            category=DivergenceClass.UNDETERMINED,
            first_step_index=index,
            first_step_id=observation.step_id,
            differences=tuple(differences[:MAX_DIFFERENCES]),
            reason="a required observation field is unavailable",
        )
    return Classification(
        category=DivergenceClass.NONE, reason="all three modes agree"
    )


def _leaves_compiler_history(step: Any) -> bool:
    if isinstance(step, Invocation):
        return True
    return isinstance(step, StateOperation) and step.introduces_compiler_history


def _infrastructure_reason(
    sequence: Sequence, traces: Mapping[Mode, Trace]
) -> str | None:
    missing = [mode for mode in Mode if mode not in traces]
    if missing:
        return "missing mode traces: " + ", ".join(mode.value for mode in missing)
    errors = [
        f"{mode.value}: {traces[mode].infrastructure_error}"
        for mode in Mode
        if traces[mode].infrastructure_error
    ]
    if errors:
        return "; ".join(errors)
    expected = [step.step_id for step in sequence.steps]
    for mode in Mode:
        actual = [observation.step_id for observation in traces[mode].observations]
        if actual != expected:
            return f"{mode.value} returned unaligned observations: {actual!r} != {expected!r}"
    return None
