"""MindSpore adapter (``mindspore.jit`` / GRAPH_MODE)."""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import Decision, Invocation, Scenario
from ._common import CompiledFrameworkAdapter
from .reference import dtype_class, spec_tag

__all__ = ["MindSporeAdapter"]


def _ms_dtype(name: str):
    import mindspore as ms

    return {
        "float16": ms.float16,
        "bfloat16": ms.bfloat16,
        "float32": ms.float32,
        "float64": ms.float64,
        "int64": ms.int64,
        "int32": ms.int32,
        "bool": ms.bool_,
        "complex64": ms.complex64,
        "complex128": ms.complex128,
    }.get(name, ms.float32)


class MindSporeAdapter(CompiledFrameworkAdapter):
    framework = "mindspore"

    def __init__(self, scenario: Scenario, model_factory, state_factory, space=None):
        super().__init__(scenario, model_factory, state_factory, space)
        self._compiled_signatures: set = set()

    def _materialize(self, spec) -> list:
        import mindspore as ms

        from .base import materialize_values

        values = materialize_values(spec)
        tensor = ms.Tensor(values, dtype=_ms_dtype(spec.dtype))
        return [tensor.reshape(spec.shape)]

    def _eager(self, model, args):
        import mindspore as ms

        previous = ms.get_context("mode")
        ms.set_context(mode=ms.PYNATIVE_MODE)
        try:
            return model(*args)
        finally:
            ms.set_context(mode=previous)

    def _compile(self, model, options: Mapping):
        import mindspore as ms

        ms.set_context(mode=ms.GRAPH_MODE)
        return ms.jit(model)

    def _compiled(self, compiled, args):
        return compiled(*args)

    def _reset_compiler_state(self) -> None:
        import mindspore as ms

        ms.set_context(mode=ms.PYNATIVE_MODE)

    def _decision(self, step: Invocation) -> Decision:
        spec = step.spec
        if spec is None:
            return Decision.REUSED
        signature = (spec.operator, spec_tag(spec), dtype_class(spec.dtype))
        if signature in self._compiled_signatures:
            return Decision.REUSED
        self._compiled_signatures.add(signature)
        return Decision.COMPILED

    def _lowering(self, step: Invocation) -> str:
        return "mindspore-graph"

    def _apply_state(self, model, operation: str, parameters: Mapping) -> None:
        apply = getattr(model, "apply_state", None)
        if callable(apply):
            apply(operation, dict(parameters))

    def _observe(self, value: Any, spec, state) -> Any:
        return super()._observe(value, spec, state)
