"""A self-contained reference backend with a *modeled* stateful compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..domain import (
    BehaviorKey,
    CompilerEvent,
    Decision,
    Guard,
    GuardGoal,
    Invocation,
    Mode,
    Observation,
    ProgramState,
    Scenario,
    Sequence,
    StateOperation,
    StepSpec,
    TensorValue,
    Trace,
)
from ..generation import OperatorSpace
from ..semantic import SemanticProbe
from .base import FrameworkAdapter, materialize_values

__all__ = ["CompilerDefectProfile", "ReferenceAdapter", "spec_tag", "dtype_class"]


@dataclass(frozen=True)
class CompilerDefectProfile:
    """Which modeled defects the reference compiler exhibits."""

    stale_parameter_folding: bool = True
    mode_leak: bool = True
    nan_unsound_relu: bool = True


def dtype_class(dtype: str) -> str:
    if dtype in ("float16", "bfloat16"):
        return "low"
    return "wide"


def spec_tag(spec: StepSpec) -> str:
    """The specialization a guard records for a shape."""
    if any(dim == 1 for dim in spec.shape):
        return "s1"
    if all(dim % 16 == 0 for dim in spec.shape):
        return "tile16"
    if all(dim % 4 == 0 for dim in spec.shape):
        return "aligned"
    return "sparse"


def lowering_path(spec: StepSpec) -> str:
    if spec.operator == "matmul":
        return f"gemm-{spec_tag(spec)}"
    if spec.operator in ("mul", "add", "relu"):
        return "pointwise-vec" if len(spec.shape) > 1 else "pointwise-scalar"
    return f"reduce-{spec_tag(spec)}"


_OP_CLASS = {
    "matmul": "gemm",
    "mul": "pointwise",
    "add": "pointwise",
    "relu": "pointwise",
    "sum": "reduce",
    "softmax": "reduce",
}

#: Operators whose compiled artifacts fold the parameter into the code.
_FOLDING_OPERATORS = frozenset({"mul", "sum", "softmax", "matmul"})


@dataclass(frozen=True)
class _Artifact:
    operator: str
    tag: str
    dtype_cls: str
    mode: str
    path: str
    folds_parameter: bool
    parameter_snapshot: float
    parameter_version: int
    mode_scale: float

    @property
    def uid(self) -> str:
        return f"{self.operator}|{self.tag}|{self.dtype_cls}|{self.mode}"


class _Compiler:
    """A tiny compiler cache with guards, decisions and divergences."""

    def __init__(self, profile: CompilerDefectProfile) -> None:
        self.profile = profile
        self.artifacts: dict[tuple, _Artifact] = {}

    def reset(self) -> None:
        self.artifacts.clear()

    def _key(self, spec: StepSpec, state: ProgramState) -> tuple:
        base = (spec.operator, spec_tag(spec), dtype_class(spec.dtype))
        if self.profile.mode_leak:
            # The guard forgets the mode, so a train artifact can be reused in
            return base
        return base + (state.mode,)

    def guard_holds(self, artifact: _Artifact, spec: StepSpec, state: ProgramState) -> bool:
        if artifact.operator != spec.operator:
            return False
        if artifact.tag != spec_tag(spec):
            return False
        if artifact.dtype_cls != dtype_class(spec.dtype):
            return False
        if not self.profile.mode_leak and artifact.mode != state.mode:
            return False
        return True

    def resolve(
        self, spec: StepSpec, state: ProgramState
    ) -> tuple[_Artifact, Decision]:
        key = self._key(spec, state)
        artifact = self.artifacts.get(key)
        if artifact is not None and self.guard_holds(artifact, spec, state):
            return artifact, Decision.REUSED
        artifact = self._compile(spec, state)
        self.artifacts[key] = artifact
        return artifact, Decision.COMPILED

    def _compile(self, spec: StepSpec, state: ProgramState) -> _Artifact:
        folds = (
            self.profile.stale_parameter_folding
            and spec.operator in _FOLDING_OPERATORS
            and spec_tag(spec) in ("s1", "tile16")
        )
        return _Artifact(
            operator=spec.operator,
            tag=spec_tag(spec),
            dtype_cls=dtype_class(spec.dtype),
            mode=state.mode,
            path=lowering_path(spec),
            folds_parameter=folds,
            parameter_snapshot=_weight(state),
            parameter_version=int(state.extra.get("param_version", 0)),
            mode_scale=_mode_scale(state),
        )

    def divergence(
        self, artifact: _Artifact, spec: StepSpec, state: ProgramState, decision: Decision
    ) -> str | None:
        if decision is not Decision.REUSED:
            return None
        if artifact.folds_parameter and artifact.parameter_version != int(
            state.extra.get("param_version", 0)
        ):
            return f"stale-parameter:{spec.operator}:{artifact.tag}"
        if self.profile.mode_leak and artifact.mode != state.mode:
            return f"mode-leak:{spec.operator}:{artifact.tag}"
        return None


def _weight(state: ProgramState) -> float:
    return float(state.parameters["weight"].data[0])


def _set_weight(state: ProgramState, value: float) -> None:
    state.parameters["weight"] = TensorValue("float64", (1,), (float(value),))


def _mode_scale(state: ProgramState) -> float:
    return 2.0 if state.mode == "train" else 1.0


class ReferenceAdapter(FrameworkAdapter):
    """The offline reference backend."""

    framework = "reference"

    def __init__(
        self,
        scenario: Scenario | None = None,
        space: OperatorSpace | None = None,
        profile: CompilerDefectProfile | None = None,
    ) -> None:
        scenario = scenario or Scenario(
            name="reference",
            framework="reference",
            observable_state_fields=("weight",),
            max_sequence_length=8,
        )
        super().__init__(scenario, space or OperatorSpace())
        self.profile = profile or CompilerDefectProfile()

    # state
    def initial_state(self) -> ProgramState:
        state = ProgramState(mode="eval")
        _set_weight(state, 2.0)
        state.extra["param_version"] = 0
        state.extra["saved_weight"] = 2.0
        state.extra["requires_grad"] = False
        return state

    # whole-sequence runs
    def run(self, sequence: Sequence, mode: Mode) -> Trace:
        if not self.legal(sequence):
            return Trace(
                mode=mode,
                observations=(),
                infrastructure_error="illegal sequence",
            )
        state = self.initial_state()
        compiler = _Compiler(self.profile)
        observations: list[Observation] = []
        events: list[CompilerEvent] = []
        divergences: list[str] = []

        for index, step in enumerate(sequence.steps):
            if isinstance(step, Invocation):
                if mode is Mode.COMPILED_WITHOUT_STATE:
                    compiler.reset()
                if mode is Mode.WITHOUT_COMPILATION:
                    observation = self._observe_eager(index, step, state)
                else:
                    observation, event, divergence = self._observe_compiled(
                        index, step, state, compiler
                    )
                    events.append(event)
                    if divergence:
                        divergences.append(divergence)
            else:
                observation = self._observe_state(index, step, state)
            observations.append(observation)

        return Trace(
            mode=mode,
            observations=tuple(observations),
            events=tuple(events),
            divergences=tuple(divergences),
        )

    # controlled replay
    def observe_aligned(
        self, prefix: Sequence, target, *, compiled: bool, reset: bool
    ) -> Observation:
        state = self.initial_state()
        compiler = _Compiler(self.profile)

        compile_prefix = compiled and not reset
        for index, step in enumerate(prefix.steps):
            if isinstance(step, Invocation):
                if compile_prefix:
                    self._observe_compiled(index, step, state, compiler)
                else:
                    self._observe_eager(index, step, state)
            else:
                self._observe_state(index, step, state)

        if reset:
            compiler.reset()

        if not compiled:
            return self._observe_eager(len(prefix.steps), target, state)
        observation, _, _ = self._observe_compiled(len(prefix.steps), target, state, compiler)
        return observation

    # step execution
    def _observe_eager(self, index: int, step: Invocation, state: ProgramState) -> Observation:
        spec = step.spec or self.space.default_spec()
        value = _apply_op(spec, _weight(state), _mode_scale(state), compiled=False, profile=self.profile)
        return Observation(
            step_index=index,
            step_id=step.step_id,
            output=value,
            state=self._observable(state),
            behavior_keys=(),
        )

    def _observe_compiled(
        self, index: int, step: Invocation, state: ProgramState, compiler: _Compiler
    ) -> tuple[Observation, CompilerEvent, str | None]:
        spec = step.spec or self.space.default_spec()
        artifact, decision = compiler.resolve(spec, state)
        divergence = compiler.divergence(artifact, spec, state, decision)
        weight = artifact.parameter_snapshot if artifact.folds_parameter else _weight(state)
        if (
            self.profile.mode_leak
            and decision is Decision.REUSED
            and artifact.mode != state.mode
        ):
            mode_scale = artifact.mode_scale
        else:
            mode_scale = _mode_scale(state)
        value = _apply_op(spec, weight, mode_scale, compiled=True, profile=self.profile)

        key = BehaviorKey(
            operator=spec.operator,
            lowering=artifact.path,
            specialization=artifact.tag,
            dtype_class=artifact.dtype_cls,
            decision=decision,
        )
        guard = Guard(
            artifact_id=artifact.uid,
            condition=f"op={spec.operator},tag={artifact.tag},dtype={artifact.dtype_cls}",
            outcome=artifact.tag,
            fields={"operator": spec.operator, "spec_tag": artifact.tag},
        )
        event = CompilerEvent(
            step_index=index,
            step_id=step.step_id,
            artifact_id=artifact.uid,
            decision=decision,
            guard=guard,
            key=key,
        )
        observation = Observation(
            step_index=index,
            step_id=step.step_id,
            output=value,
            state=self._observable(state),
            behavior_keys=(key.text,),
        )
        return observation, event, divergence

    def _observe_state(
        self, index: int, step: StateOperation, state: ProgramState
    ) -> Observation:
        if step.operation == "update":
            target = step.parameters.get("weight")
            if target is not None:
                _set_weight(state, float(target))
            state.extra["param_version"] = int(state.extra.get("param_version", 0)) + 1
        elif step.operation == "mode":
            state.mode = str(step.parameters.get("mode", state.mode))
        elif step.operation == "restore":
            _set_weight(state, float(state.extra.get("saved_weight", 2.0)))
            state.extra["param_version"] = int(state.extra.get("param_version", 0)) + 1
        elif step.operation == "backward":
            state.extra["requires_grad"] = True
        return Observation(
            step_index=index,
            step_id=step.step_id,
            output=self._observable(state),
            state=self._observable(state),
        )

    def _observable(self, state: ProgramState) -> dict:
        return {
            "weight": state.parameters["weight"],
            "mode": state.mode,
            "requires_grad": bool(state.extra.get("requires_grad", False)),
        }

    # declarations
    def guard_goals(self) -> tuple[GuardGoal, ...]:
        goals: list[GuardGoal] = []
        for operator in self.space.operators:
            goals.append(GuardGoal(op=operator, rank=2, shape_dim_1=0))
            goals.append(GuardGoal(op=operator, rank=2, shape_divisible_by=16))
        return tuple(goals)

    def semantic_probes(self, sequence: Sequence, traces: Mapping) -> tuple[SemanticProbe, ...]:
        probes: list[SemanticProbe] = []

        # (a) A repeated invocation with *no intervening program-state change*
        index_of = {step.step_id: index for index, step in enumerate(sequence.steps)}
        invocations = {step.step_id: step for step in sequence.invocations}
        state_changing = {"update", "restore", "mode"}
        for step in sequence.steps:
            if isinstance(step, Invocation) and step.repeat_of is not None:
                original = invocations.get(step.repeat_of)
                if original is None:
                    continue
                if step.spec is None or original.spec is None:
                    continue
                if step.spec.shape != original.spec.shape:
                    # The repeat no longer targets the same observable shape.
                    continue
                repeat_index = index_of[step.step_id]
                original_index = index_of[original.step_id]
                between = sequence.steps[original_index + 1 : repeat_index]
                if any(
                    isinstance(item, StateOperation) and item.operation in state_changing
                    for item in between
                ):
                    continue
                probes.append(
                    SemanticProbe(
                        kind="identity",
                        step_id=step.step_id,
                        label="return-invocation",
                        payload={
                            "left": _floats(traces, step.step_id),
                            "right": _floats(traces, original.step_id),
                        },
                    )
                )

        # (b) Mathematical bounds of the declared operators.
        for index, step in enumerate(sequence.steps):
            if not isinstance(step, Invocation) or step.spec is None:
                continue
            if step.spec.operator == "relu":
                probes.append(
                    SemanticProbe(
                        kind="bounds",
                        step_id=step.step_id,
                        label="relu-lower",
                        payload={"observations": _floats(traces, step.step_id), "lower": 0.0},
                    )
                )
            elif step.spec.operator == "softmax":
                probes.append(
                    SemanticProbe(
                        kind="bounds",
                        step_id=step.step_id,
                        label="softmax-range",
                        payload={
                            "observations": _floats(traces, step.step_id),
                            "lower": 0.0,
                            "upper": 1.0,
                        },
                    )
                )
        return tuple(probes)


# The modeled computation
def _apply_op(
    spec: StepSpec,
    weight: float,
    mode_scale: float,
    *,
    compiled: bool,
    profile: CompilerDefectProfile,
) -> TensorValue:
    data = [float(value) for value in materialize_values(spec)]
    operator = spec.operator

    if operator == "sum":
        result = [mode_scale * math.fsum(weight * value for value in data)]
        shape = (1,)
    elif operator == "softmax":
        scaled = [mode_scale * weight * value for value in data]
        if any(math.isnan(value) for value in scaled):
            result = [float("nan")] * len(scaled)
        else:
            maximum = max(scaled)
            exps = [math.exp(value - maximum) for value in scaled]
            total = sum(exps)
            result = [value / total for value in exps]
        shape = spec.shape
    elif operator == "relu":
        if compiled and profile.nan_unsound_relu:
            # Point-wise defect: the compiled kernel maps NaN to 0 as well.
            result = [
                0.0 if math.isnan(weight * value) else max(0.0, mode_scale * weight * value)
                for value in data
            ]
        else:
            result = [
                float("nan")
                if math.isnan(value)
                else max(0.0, mode_scale * weight * value)
                for value in data
            ]
        shape = spec.shape
    elif operator == "add":
        result = [mode_scale * (weight + value) for value in data]
        shape = spec.shape
    else:  # mul, matmul
        result = [mode_scale * weight * value for value in data]
        shape = spec.shape

    return TensorValue(
        dtype="float64",
        shape=shape,
        data=tuple(_cast(value, spec.dtype) for value in result),
    )


def _cast(value: float, dtype: str) -> float:
    # The reference model computes in float64 but reports the requested dtype;
    import struct

    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<f", value))[0]
    return value


def _floats(traces: Mapping, step_id: str) -> list[float]:
    # Semantic obligations concern the *compiled* behavior; using the reference
    trace = traces.get(Mode.COMPILED_WITH_STATE) or traces[Mode.WITHOUT_COMPILATION]
    observation = trace.observation(step_id)
    output = observation.output
    if isinstance(output, TensorValue):
        return [float(value) for value in output.data]
    if isinstance(output, Mapping):
        values = []
        for value in output.values():
            if isinstance(value, TensorValue):
                values.extend(float(item) for item in value.data)
        return values
    return []
