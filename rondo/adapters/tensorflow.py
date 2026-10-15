"""TensorFlow adapter (``tf.function``)."""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import Decision, Invocation, Scenario
from ._common import CompiledFrameworkAdapter
from .reference import dtype_class, spec_tag

__all__ = ["TensorFlowAdapter"]


def _tf_dtype(name: str):
    import tensorflow as tf

    return {
        "float16": tf.float16,
        "bfloat16": tf.bfloat16,
        "float32": tf.float32,
        "float64": tf.float64,
        "int64": tf.int64,
        "int32": tf.int32,
        "bool": tf.bool,
        "complex64": tf.complex64,
        "complex128": tf.complex128,
    }.get(name, tf.float32)


class TensorFlowAdapter(CompiledFrameworkAdapter):
    framework = "tensorflow"

    def __init__(self, scenario: Scenario, model_factory, state_factory, space=None):
        super().__init__(scenario, model_factory, state_factory, space)
        self._compiled_signatures: set = set()

    def _materialize(self, spec) -> list:
        import tensorflow as tf

        from .base import materialize_values

        values = materialize_values(spec)
        tensor = tf.constant(values, dtype=_tf_dtype(spec.dtype))
        return [tf.reshape(tensor, spec.shape)]

    def _eager(self, model, args):
        return model(*args)

    def _compile(self, model, options: Mapping):
        import tensorflow as tf

        return tf.function(
            model,
            jit_compile=bool(options.get("jit_compile", False)),
            reduce_retracing=True,
        )

    def _compiled(self, compiled, args):
        return compiled(*args)

    def _reset_compiler_state(self) -> None:
        import tensorflow as tf

        tf.keras.backend.clear_session()

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
        return "tf.function"

    def _apply_state(self, model, operation: str, parameters: Mapping) -> None:
        variables = dict(getattr(model, "variables", {}) or {})
        if operation in ("update", "restore"):
            for name, value in parameters.items():
                variable = variables.get(name)
                if variable is not None:
                    variable.assign(value)
        elif operation == "mode":
            if hasattr(model, "training"):
                model.training = str(parameters.get("mode", "train")) == "train"

    def _observe(self, value: Any, spec, state) -> Any:
        return super()._observe(value, spec, state)
