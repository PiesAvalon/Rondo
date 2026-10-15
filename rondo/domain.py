"""Core data model of Rondo."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, TypeAlias

__all__ = [
    "Argument",
    "BehaviorKey",
    "Candidate",
    "CompilerEvent",
    "Decision",
    "ExecMode",
    "ExceptionInfo",
    "Guard",
    "GuardGoal",
    "Invocation",
    "JsonValue",
    "LegalityError",
    "Mode",
    "Observation",
    "ProgramState",
    "ResetBoundary",
    "RunStatus",
    "Scenario",
    "Sequence",
    "StateOperation",
    "Step",
    "StepSpec",
    "SymptomSignature",
    "TensorValue",
    "Test",
    "Trace",
    "WitnessClass",
    "Witness",
]


# Errors
class LegalityError(ValueError):
    """A generated test case is not legal for its scenario."""


# Values
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
_COMPLEX_DTYPES = ("complex64", "complex128")
_INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8")
_DTYPES = _FLOAT_DTYPES + _COMPLEX_DTYPES + _INT_DTYPES + ("bool",)


@dataclass(frozen=True)
class TensorValue:
    """A tensor leaf of an observation."""

    dtype: str
    shape: tuple[int, ...]
    data: tuple[Any, ...]
    device: str = "cpu"
    requires_grad: bool = False
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r}")
        size = 1
        for dim in self.shape:
            if dim < 0:
                raise ValueError(f"negative dimension in shape {self.shape!r}")
            size *= dim
        if size != len(self.data):
            raise ValueError(
                f"shape {self.shape!r} needs {size} elements, got {len(self.data)}"
            )

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def is_floating(self) -> bool:
        return self.dtype in _FLOAT_DTYPES or self.dtype in _COMPLEX_DTYPES

    def to_json(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "kind": "tensor",
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
            "requires_grad": self.requires_grad,
            "values": [_encode_scalar(value) for value in self.data],
        }
        if self.strides is not None:
            payload["strides"] = list(self.strides)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TensorValue":
        if payload.get("kind") != "tensor":
            raise ValueError(f"not a tensor payload: {payload!r}")
        strides = payload.get("strides")
        return cls(
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            data=tuple(_decode_scalar(value) for value in payload["values"]),
            device=str(payload.get("device", "cpu")),
            requires_grad=bool(payload.get("requires_grad", False)),
            strides=tuple(int(s) for s in strides) if strides is not None else None,
        )


def _encode_scalar(value: Any) -> JsonValue:
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    return float(value)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return complex(float(value[0]), float(value[1]))
    return value


Argument: TypeAlias = TensorValue | None | bool | int | float | str


def encode_value(value: Any) -> JsonValue:
    """Encode an observation tree (tensors, containers, scalars) to JSON."""
    if isinstance(value, TensorValue):
        return value.to_json()
    if isinstance(value, Mapping):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise TypeError(f"cannot encode {type(value).__name__} as an observation value")


def decode_value(payload: JsonValue) -> Any:
    """Inverse of :func:`encode_value`."""
    if isinstance(payload, dict):
        if payload.get("kind") == "tensor":
            return TensorValue.from_json(payload)
        return {key: decode_value(item) for key, item in payload.items()}
    if isinstance(payload, list):
        return [decode_value(item) for item in payload]
    return payload


# Program state
@dataclass
class ProgramState:
    """Program state ``s``."""

    parameters: dict[str, TensorValue] = field(default_factory=dict)
    buffers: dict[str, TensorValue] = field(default_factory=dict)
    inputs: dict[str, TensorValue] = field(default_factory=dict)
    mode: str = "train"
    requires_grad: dict[str, bool] = field(default_factory=dict)
    rng_state: int = 0
    aliases: tuple[tuple[str, str], ...] = ()
    extra: dict[str, JsonValue] = field(default_factory=dict)

    def clone(self) -> "ProgramState":
        """Deep-enough copy: tensors are immutable, containers are copied."""
        return ProgramState(
            parameters=dict(self.parameters),
            buffers=dict(self.buffers),
            inputs=dict(self.inputs),
            mode=self.mode,
            requires_grad=dict(self.requires_grad),
            rng_state=self.rng_state,
            aliases=tuple(self.aliases),
            extra=dict(self.extra),
        )

    def observable(self, fields: Iterable[str]) -> dict[str, Any]:
        """The part of the state declared observable by the contract."""
        view: dict[str, Any] = {}
        for name in fields:
            if name in self.parameters:
                view[f"param.{name}"] = self.parameters[name]
            elif name in self.buffers:
                view[f"buffer.{name}"] = self.buffers[name]
            elif name in self.inputs:
                view[f"input.{name}"] = self.inputs[name]
            elif name in self.extra:
                view[f"extra.{name}"] = self.extra[name]
        return view

    def digest(self) -> str:
        payload = encode_value(
            {
                "parameters": self.parameters,
                "buffers": self.buffers,
                "inputs": self.inputs,
                "mode": self.mode,
                "requires_grad": self.requires_grad,
                "rng_state": self.rng_state,
                "aliases": [list(pair) for pair in self.aliases],
                "extra": self.extra,
            }
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


# Steps
@dataclass(frozen=True)
class StepSpec:
    """The enumerable language of one model invocation."""

    operator: str
    shape: tuple[int, ...] = (4,)
    dtype: str = "float32"
    value_class: str = "ordinary"
    layout: str = "contiguous"
    device: str = "cpu"
    value_seed: int = 0
    scale: float = 1.0
    requires_grad: bool = False

    def with_fields(self, **changes: Any) -> "StepSpec":
        return replace(self, **changes)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "operator": self.operator,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "value_class": self.value_class,
            "layout": self.layout,
            "device": self.device,
            "value_seed": self.value_seed,
            "scale": self.scale,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True)
class Invocation:
    """One model invocation ``a`` with concrete inputs and call attributes."""

    step_id: str
    args: tuple[Argument, ...] = ()
    kwargs: Mapping[str, Argument] = field(default_factory=dict)
    operator: str = "call"
    attributes: Mapping[str, JsonValue] = field(default_factory=dict)
    repeat_of: str | None = None
    spec: StepSpec | None = None
    kind: str = field(default="invoke", init=False)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("Invocation.step_id must be non-empty")

    def signature(self) -> str:
        payload = encode_value(
            {
                "operator": self.operator,
                "args": list(self.args),
                "kwargs": dict(self.kwargs),
                "attributes": dict(self.attributes),
            }
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": "invoke",
            "step_id": self.step_id,
            "operator": self.operator,
            "args": [encode_value(arg) for arg in self.args],
            "kwargs": {key: encode_value(val) for key, val in self.kwargs.items()},
            "attributes": dict(self.attributes),
            "repeat_of": self.repeat_of,
            "spec": self.spec.to_json() if self.spec is not None else None,
        }


#: State operations that leave *compiler* history behind (not update/mode/restore).
COMPILER_HISTORY_OPERATIONS = frozenset({"warmup", "trace", "compile", "capture"})
STATE_OPERATIONS = frozenset({"update", "mode", "restore", "backward"}) | set(
    COMPILER_HISTORY_OPERATIONS
)


@dataclass(frozen=True)
class StateOperation:
    """A declared change to program state (or a compiler-history event)."""

    step_id: str
    operation: str
    parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    kind: str = field(default="state", init=False)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("StateOperation.step_id must be non-empty")
        if self.operation not in STATE_OPERATIONS:
            raise ValueError(f"unknown state operation {self.operation!r}")

    @property
    def introduces_compiler_history(self) -> bool:
        return self.operation in COMPILER_HISTORY_OPERATIONS

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "kind": "state",
            "step_id": self.step_id,
            "operation": self.operation,
            "parameters": dict(self.parameters),
        }


Step: TypeAlias = Invocation | StateOperation


# Scenario, behavior, traces
@dataclass(frozen=True)
class ResetBoundary:
    """The boundary ``b`` at which compiler state is reset."""

    scope: str = "framework"
    isolation: str = "in-process"  # "in-process" | "process"


@dataclass(frozen=True)
class Guard:
    """A condition a compiled artifact matched on."""

    artifact_id: str
    condition: str
    outcome: str
    fields: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardGoal:
    """A target guard outcome expressed over the step language."""

    op: str | None = None
    rank: int | None = None
    shape_eq: int | None = None
    shape_dim_1: int | None = None
    shape_divisible_by: int | None = None
    dtype: str | None = None
    value_class: str | None = None
    pending_state: str | None = None

    def describe(self) -> str:
        parts = [
            f"{name}={value!r}"
            for name, value in (
                ("op", self.op),
                ("rank", self.rank),
                ("shape_eq", self.shape_eq),
                ("shape_dim_1", self.shape_dim_1),
                ("shape_divisible_by", self.shape_divisible_by),
                ("dtype", self.dtype),
                ("value_class", self.value_class),
                ("pending_state", self.pending_state),
            )
            if value is not None
        ]
        return "GuardGoal(" + ", ".join(parts) + ")"


@dataclass(frozen=True)
class Scenario:
    """A fixed testing scenario."""

    name: str
    framework: str
    backend: str = "default"
    device: str = "cpu"
    options: Mapping[str, JsonValue] = field(default_factory=dict)
    reset_boundary: ResetBoundary = field(default_factory=ResetBoundary)
    observable_state_fields: tuple[str, ...] = ()
    max_sequence_length: int = 16
    seed: int = 0

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "framework": self.framework,
            "backend": self.backend,
            "device": self.device,
            "options": dict(self.options),
            "reset_boundary": {
                "scope": self.reset_boundary.scope,
                "isolation": self.reset_boundary.isolation,
            },
            "observable_state_fields": list(self.observable_state_fields),
            "max_sequence_length": self.max_sequence_length,
            "seed": self.seed,
        }


class Decision(str, Enum):
    """What the compiler did with an invocation."""

    COMPILED = "compiled"
    REUSED = "reused"


@dataclass(frozen=True)
class BehaviorKey:
    """Lossy projection of the compiler behavior observed for one invocation."""

    operator: str
    lowering: str
    specialization: str
    dtype_class: str
    decision: Decision

    @property
    def text(self) -> str:
        return "|".join(
            (
                self.operator,
                self.lowering,
                self.specialization,
                self.dtype_class,
                self.decision.value,
            )
        )

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "operator": self.operator,
            "lowering": self.lowering,
            "specialization": self.specialization,
            "dtype_class": self.dtype_class,
            "decision": self.decision.value,
        }


@dataclass(frozen=True)
class CompilerEvent:
    """A single compile/reuse decision, tied to the step that caused it."""

    step_index: int
    step_id: str
    artifact_id: str
    decision: Decision
    guard: Guard | None = None
    key: BehaviorKey | None = None


@dataclass(frozen=True)
class ExceptionInfo:
    """A framework failure recorded as data (never raised across the oracle)."""

    class_name: str
    phase: str
    message: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {"class_name": self.class_name, "phase": self.phase, "message": self.message}


@dataclass(frozen=True)
class Observation:
    """Structured observation ``o = ⟨y, π(s')⟩`` for one step."""

    step_index: int
    step_id: str
    output: Any = None
    state: Mapping[str, Any] = field(default_factory=dict)
    exception: ExceptionInfo | None = None
    behavior_keys: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.exception is None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "step_index": self.step_index,
            "step_id": self.step_id,
            "output": encode_value(self.output),
            "state": encode_value(dict(self.state)),
            "exception": self.exception.to_json() if self.exception else None,
            "behavior_keys": list(self.behavior_keys),
            "unsupported_fields": list(self.unsupported_fields),
        }


class Mode(str, Enum):
    """The three invocation modes of the test oracle."""

    WITHOUT_COMPILATION = "without-compilation"
    COMPILED_WITHOUT_STATE = "compiled-without-state"
    COMPILED_WITH_STATE = "compiled-with-state"

    # Convenience aliases used throughout the code.
    REFERENCE = "without-compilation"
    FRESH = "compiled-without-state"
    PERSISTENT = "compiled-with-state"

    @property
    def uses_compilation(self) -> bool:
        return self is not Mode.WITHOUT_COMPILATION

    @property
    def retains_compiler_state(self) -> bool:
        return self is Mode.COMPILED_WITH_STATE


#: ExecMode is kept as a readable alias for the mode a *single* run uses.
ExecMode = Mode


@dataclass(frozen=True)
class RunStatus(str, Enum):
    OK = "ok"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class Trace:
    """The result of running one sequence under one mode."""

    mode: Mode
    observations: tuple[Observation, ...]
    events: tuple[CompilerEvent, ...] = ()
    divergences: tuple[str, ...] = ()
    infrastructure_error: str | None = None
    environment: Mapping[str, JsonValue] = field(default_factory=dict)

    def observation(self, step_id: str) -> Observation:
        for observation in self.observations:
            if observation.step_id == step_id:
                return observation
        raise KeyError(step_id)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for obs in self.observations for key in obs.behavior_keys)


# Sequences and tests
@dataclass(frozen=True)
class Sequence:
    """An invocation sequence ``X = ⟨a_1, …, a_n⟩`` within a scenario."""

    scenario: str
    steps: tuple[Step, ...] = ()
    seed: int = 0
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise LegalityError(f"duplicate step id {step.step_id!r}")
            if isinstance(step, Invocation) and step.repeat_of is not None:
                if step.repeat_of not in seen:
                    raise LegalityError(
                        f"{step.step_id!r} repeats {step.repeat_of!r}, which is not earlier"
                    )
            seen.add(step.step_id)

    # construction helpers
    def append(self, step: Step) -> "Sequence":
        return replace(self, steps=self.steps + (step,))

    def replace_step(self, index: int, step: Step) -> "Sequence":
        steps = list(self.steps)
        steps[index] = step
        return replace(self, steps=tuple(steps))

    def truncate(self, length: int) -> "Sequence":
        if length < 0:
            raise ValueError("length must be non-negative")
        return replace(self, steps=self.steps[:length])

    def drop(self, indices: Iterable[int]) -> "Sequence":
        drop = set(indices)
        return replace(
            self, steps=tuple(s for i, s in enumerate(self.steps) if i not in drop)
        )

    def prefix(self, length: int) -> "Sequence":
        return replace(self, steps=self.steps[:length])

    def target(self) -> Step:
        if not self.steps:
            raise ValueError("empty sequence has no target step")
        return self.steps[-1]

    # inspection
    @property
    def invocations(self) -> tuple[Invocation, ...]:
        return tuple(s for s in self.steps if isinstance(s, Invocation))

    @property
    def invocation_count(self) -> int:
        return len(self.invocations)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def digest(self) -> str:
        payload = encode_value(
            {
                "scenario": self.scenario,
                "steps": [step.to_json() for step in self.steps],
            }
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def prefix_history_count(self, index: int) -> int:
        """Number of compiler-history steps strictly before ``index``."""
        count = 0
        for step in self.steps[:index]:
            if isinstance(step, Invocation) or (
                isinstance(step, StateOperation) and step.introduces_compiler_history
            ):
                count += 1
        return count

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "steps": [step.to_json() for step in self.steps],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Test:
    """A test case ``⟨M, X⟩``: a scenario plus an invocation sequence."""

    scenario: Scenario
    sequence: Sequence

    def to_json(self) -> dict[str, JsonValue]:
        return {"scenario": self.scenario.to_json(), "sequence": self.sequence.to_json()}


# Candidates and witnesses
class WitnessClass(str, Enum):
    """Reproduction class of a candidate (see :mod:`rondo.refinement`)."""

    POINT_WISE = "point-wise"
    HISTORY_EXPOSED = "history-exposed"
    STATEFUL = "stateful"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class SymptomSignature:
    """A symptom to be preserved during minimization."""

    step_id: str
    path: str
    kind: str
    modes: tuple[str, ...]

    def text(self) -> str:
        return f"{self.step_id}:{self.path}:{self.kind}:{','.join(self.modes)}"

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "step_id": self.step_id,
            "path": self.path,
            "kind": self.kind,
            "modes": list(self.modes),
        }


@dataclass(frozen=True)
class Candidate:
    """A suspected bug produced by the oracle, before classification."""

    sequence: Sequence
    target_step_id: str
    failing_modes: tuple[Mode, ...]
    symptoms: tuple[SymptomSignature, ...]
    reason: str = ""
    semantic_obligation: str | None = None

    @property
    def prefix(self) -> Sequence:
        steps = self.sequence.steps
        for index, step in enumerate(steps):
            if step.step_id == self.target_step_id:
                return self.sequence.prefix(index)
        raise KeyError(self.target_step_id)

    @property
    def target(self) -> Step:
        for step in self.sequence.steps:
            if step.step_id == self.target_step_id:
                return step
        raise KeyError(self.target_step_id)

    @property
    def signature(self) -> str:
        payload = {
            "target": self.target_step_id,
            "failing_modes": sorted(mode.value for mode in self.failing_modes),
            "symptoms": sorted(symptom.text() for symptom in self.symptoms),
            "semantic_obligation": self.semantic_obligation,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class Witness:
    """A classified, minimized reproduction of one bug."""

    candidate: Candidate
    witness_class: WitnessClass
    kept_correct: bool
    reset_correct: bool
    standalone_reproduces: bool | None
    minimized_sequence: Sequence
    preserved: tuple[SymptomSignature, ...]
    evaluations: int = 0
    removed_step_ids: tuple[str, ...] = ()
    reason: str = ""

    @property
    def is_confirmed(self) -> bool:
        return self.witness_class is not WitnessClass.UNDETERMINED

    def summarize(self) -> str:
        return (
            f"[{self.witness_class.value}] {self.candidate.target_step_id} "
            f"in {self.minimized_sequence.length} step(s) "
            f"(kept_correct={self.kept_correct}, reset_correct={self.reset_correct}, "
            f"standalone={self.standalone_reproduces})"
        )
