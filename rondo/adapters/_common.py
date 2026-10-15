"""Shared mode logic for framework adapters."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..domain import (
    BehaviorKey,
    CompilerEvent,
    Decision,
    Invocation,
    Mode,
    Observation,
    ProgramState,
    Scenario,
    Sequence,
    TensorValue,
    Trace,
)
from ..generation import OperatorSpace
from .base import FrameworkAdapter
from .reference import dtype_class, spec_tag

__all__ = ["CompiledFrameworkAdapter"]


class CompiledFrameworkAdapter(FrameworkAdapter):
    """Base class implementing the three-mode protocol for real frameworks."""

    def __init__(
        self,
        scenario: Scenario,
        model_factory: Callable[[], Any],
        state_factory: Callable[[], ProgramState],
        space: OperatorSpace | None = None,
    ) -> None:
        super().__init__(scenario, space)
        self.model_factory = model_factory
        self.state_factory = state_factory

    # lifecycle
    def initial_state(self) -> ProgramState:
        return self.state_factory()

    def run(self, sequence: Sequence, mode: Mode) -> Trace:
        if not self.legal(sequence):
            return Trace(mode=mode, observations=(), infrastructure_error="illegal sequence")
        state = self.initial_state()
        model = self.model_factory()
        compiled = None
        observations: list[Observation] = []
        events: list[CompilerEvent] = []
        divergences: list[str] = []

        try:
            for index, step in enumerate(sequence.steps):
                if isinstance(step, Invocation):
                    if mode is Mode.COMPILED_WITHOUT_STATE:
                        self._reset_compiler_state()
                        model = self._fresh_model(model)
                        compiled = self._compile(model, dict(self.scenario.options))
                    if mode is Mode.WITHOUT_COMPILATION:
                        observation = self._eager_observation(index, step, model, state)
                    else:
                        if compiled is None:
                            compiled = self._compile(model, dict(self.scenario.options))
                        observation, event, divergence = self._compiled_observation(
                            index, step, model, compiled, state
                        )
                        events.append(event)
                        if divergence:
                            divergences.append(divergence)
                else:
                    self._apply_state(model, step.operation, step.parameters)
                    observation = Observation(
                        step_index=index,
                        step_id=step.step_id,
                        output=self._state_view(state),
                        state=self._state_view(state),
                    )
                observations.append(observation)
        except Exception as error:  # framework failure is data, not a crash
            return Trace(
                mode=mode,
                observations=tuple(observations),
                events=tuple(events),
                divergences=tuple(divergences),
                infrastructure_error=f"{type(error).__name__}: {error}",
            )

        return Trace(
            mode=mode,
            observations=tuple(observations),
            events=tuple(events),
            divergences=tuple(divergences),
        )

    def observe_aligned(
        self, prefix: Sequence, target, *, compiled: bool, reset: bool
    ) -> Observation:
        state = self.initial_state()
        model = self.model_factory()
        compiler = None

        compile_prefix = compiled and not reset
        for index, step in enumerate(prefix.steps):
            if isinstance(step, Invocation):
                if compile_prefix:
                    if compiler is None:
                        compiler = self._compile(model, dict(self.scenario.options))
                    self._compiled_observation(index, step, model, compiler, state)
                else:
                    self._eager_observation(index, step, model, state)
            else:
                self._apply_state(model, step.operation, step.parameters)

        if reset:
            self._reset_compiler_state()
            model = self._fresh_model(model)
            compiler = None

        if not compiled:
            return self._eager_observation(len(prefix.steps), target, model, state)
        if compiler is None:
            compiler = self._compile(model, dict(self.scenario.options))
        observation, _, _ = self._compiled_observation(
            len(prefix.steps), target, model, compiler, state
        )
        return observation

    # observation helpers
    def _eager_observation(
        self, index: int, step: Invocation, model, state: ProgramState
    ) -> Observation:
        args = tuple(self._materialize(step.spec))
        value = self._eager(model, args)
        return Observation(
            step_index=index,
            step_id=step.step_id,
            output=self._observe(value, step.spec, state),
            state=self._state_view(state),
            behavior_keys=(),
        )

    def _compiled_observation(
        self, index: int, step: Invocation, model, compiled, state: ProgramState
    ) -> tuple[Observation, CompilerEvent, str | None]:
        args = tuple(self._materialize(step.spec))
        value = self._compiled(compiled, args)
        decision = self._decision(step)
        key = BehaviorKey(
            operator=(step.spec.operator if step.spec else step.operator),
            lowering=self._lowering(step),
            specialization=(spec_tag(step.spec) if step.spec else "unknown"),
            dtype_class=(dtype_class(step.spec.dtype) if step.spec else "wide"),
            decision=decision,
        )
        event = CompilerEvent(
            step_index=index,
            step_id=step.step_id,
            artifact_id=key.text,
            decision=decision,
            key=key,
        )
        observation = Observation(
            step_index=index,
            step_id=step.step_id,
            output=self._observe(value, step.spec, state),
            state=self._state_view(state),
            behavior_keys=(key.text,),
        )
        return observation, event, None

    def _state_view(self, state: ProgramState) -> dict[str, Any]:
        return state.observable(self.scenario.observable_state_fields)

    # hooks
    def _materialize(self, spec) -> list:
        raise NotImplementedError

    def _eager(self, model, args):
        raise NotImplementedError

    def _compile(self, model, options):
        raise NotImplementedError

    def _compiled(self, compiled, args):
        raise NotImplementedError

    def _reset_compiler_state(self) -> None:
        raise NotImplementedError

    def _fresh_model(self, model):
        return self.model_factory()

    def _observe(self, value, spec, state: ProgramState) -> Any:
        """Framework output -> comparable value tree."""
        return framework_output_to_value(value)

    def _apply_state(self, model, operation: str, parameters: Mapping) -> None:
        raise NotImplementedError

    def _decision(self, step: Invocation) -> Decision:
        """Whether this call compiled or reused."""
        return Decision.REUSED

    def _lowering(self, step: Invocation) -> str:
        return "framework"


# Output conversion
def framework_output_to_value(value: Any) -> Any:
    """Recursively convert a framework output into comparable leaves."""
    if isinstance(value, Mapping):
        return {str(key): framework_output_to_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [framework_output_to_value(item) for item in value]
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        try:
            data = value.detach().cpu()
        except AttributeError:
            data = value
        flat = list(data.reshape(-1).tolist())
        dtype = _dtype_name(getattr(data, "dtype", "float32"))
        requires_grad = bool(getattr(data, "requires_grad", False))
        return TensorValue(
            dtype=dtype,
            shape=tuple(int(dim) for dim in data.shape),
            data=tuple(flat),
            requires_grad=requires_grad,
        )
    return value


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    for name in (
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "complex128",
        "complex64",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
        "bool",
    ):
        if name in text:
            return name
    return "float32"
