"""Framework adapter interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from ..domain import (
    GuardGoal,
    Invocation,
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

__all__ = ["FrameworkAdapter"]


class FrameworkAdapter(ABC):
    """Base class for PyTorch / TensorFlow / JAX / MindSpore adapters."""

    framework: str = "generic"

    def __init__(self, scenario: Scenario, space: OperatorSpace | None = None) -> None:
        self.scenario = scenario
        self.space = space or OperatorSpace()

    # lifecycle
    @abstractmethod
    def initial_state(self) -> ProgramState:
        """The scenario's fixed initial program state ``s_0``."""

    @abstractmethod
    def run(self, sequence: Sequence, mode) -> Trace:
        """Execute a whole sequence under one oracle mode."""

    @abstractmethod
    def observe_aligned(
        self, prefix: Sequence, target, *, compiled: bool, reset: bool
    ) -> Observation:
        """Controlled replay of one invocation (see the module docstring)."""

    # declarations
    def guard_goals(self) -> tuple[GuardGoal, ...]:
        """Declared guard outcomes; adapters override when they can read guards."""
        return ()

    def semantic_probes(self, sequence: Sequence, traces: Mapping) -> Iterable[SemanticProbe]:
        """Declared properties for the independent semantic checks."""
        return ()

    # step language
    def make_invocation(
        self, step_id: str, spec: StepSpec | None, repeat_of: str | None = None
    ) -> Invocation:
        spec = spec or self.space.default_spec()
        tensor = self.materialize(spec)
        return Invocation(
            step_id=step_id,
            args=(tensor,),
            operator=spec.operator,
            attributes={"dtype": spec.dtype, "layout": spec.layout},
            repeat_of=repeat_of,
            spec=spec,
        )

    def make_state(
        self, step_id: str, operation: str, parameters: Mapping | None = None
    ) -> StateOperation:
        return StateOperation(
            step_id=step_id, operation=operation, parameters=dict(parameters or {})
        )

    def legal(self, sequence: Sequence) -> bool:
        """Scenario preconditions; the base rule is structural."""
        try:
            for step in sequence.steps:
                if isinstance(step, Invocation) and step.spec is not None:
                    if not step.spec.shape or any(dim <= 0 for dim in step.spec.shape):
                        return False
                    if step.spec.operator not in self.space.operators:
                        return False
                    if step.spec.dtype not in self.space.dtypes:
                        return False
            return True
        except (ValueError, TypeError):
            return False

    # materialization
    def materialize(self, spec: StepSpec) -> TensorValue:
        """Deterministic tensor for a spec (shared by all adapters)."""
        values = materialize_values(spec)
        strides = _strides(spec)
        return TensorValue(
            dtype=spec.dtype,
            shape=spec.shape,
            data=values,
            device=spec.device,
            requires_grad=spec.requires_grad,
            strides=strides,
        )


# Deterministic input generation from a spec
def materialize_values(spec: StepSpec) -> tuple:
    size = 1
    for dim in spec.shape:
        size *= dim
    rng = random.Random(f"{spec.value_seed}:{spec.operator}:{spec.shape}")
    base = [round(rng.uniform(-1.0, 1.0), 6) for _ in range(size)]

    if spec.dtype.startswith("int") or spec.dtype == "bool":
        return tuple(int(value >= 0) if spec.dtype == "bool" else int(value) for value in base)

    special = {
        "zero": 0.0,
        "neg_zero": -0.0,
        "subnormal": 5e-324 if spec.dtype == "float64" else 1e-45,
        "inf": float("inf"),
        "nan": float("nan"),
    }
    if spec.value_class in special:
        return tuple(special[spec.value_class] for _ in range(size))
    return tuple(value * spec.scale for value in base)


def _strides(spec: StepSpec) -> tuple[int, ...]:
    if spec.layout == "transposed" and len(spec.shape) >= 2:
        # Column-major-ish: a genuinely different storage arrangement.
        strides = [1]
        for dim in reversed(spec.shape[:-1]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))
    strides = []
    running = 1
    for dim in reversed(spec.shape):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))
