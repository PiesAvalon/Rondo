"""JAX adapter (``jax.jit``)."""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import Decision, Invocation, Scenario
from ._common import CompiledFrameworkAdapter
from .reference import dtype_class, spec_tag

__all__ = ["JaxAdapter"]


def _jnp_dtype(name: str):
    import jax.numpy as jnp

    return {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "float64": jnp.float64,
        "int64": jnp.int64,
        "int32": jnp.int32,
        "bool": jnp.bool_,
        "complex64": jnp.complex64,
        "complex128": jnp.complex128,
    }.get(name, jnp.float32)


class JaxAdapter(CompiledFrameworkAdapter):
    framework = "jax"

    def __init__(self, scenario: Scenario, model_factory, state_factory, space=None):
        super().__init__(scenario, model_factory, state_factory, space)
        self._compiled_signatures: set = set()

    def _materialize(self, spec) -> list:
        import jax.numpy as jnp

        from .base import materialize_values

        values = materialize_values(spec)
        return [jnp.asarray(values, dtype=_jnp_dtype(spec.dtype)).reshape(spec.shape)]

    def _eager(self, model, args):
        return model(*args)

    def _compile(self, model, options: Mapping):
        import jax

        return jax.jit(model, donate_argnums=tuple(options.get("donate_argnums", ())))

    def _compiled(self, compiled, args):
        return compiled(*args)

    def _reset_compiler_state(self) -> None:
        import jax

        jax.clear_caches()

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
        return "jax.jit"

    def _apply_state(self, model, operation: str, parameters: Mapping) -> None:
        # JAX models are pure functions over their parameters; a state operation
        apply = getattr(model, "apply_state", None)
        if callable(apply):
            apply(operation, dict(parameters))

    def _observe(self, value: Any, spec, state) -> Any:
        return super()._observe(value, spec, state)
