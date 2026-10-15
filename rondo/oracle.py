"""The test oracle: three modes, structured comparison, semantic checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .domain import (
    Candidate,
    Mode,
    Observation,
    Sequence,
    SymptomSignature,
    Trace,
)
from .numeric import DEFAULT_NUMERIC_POLICY, NumericPolicy
from .observation import (
    Classification,
    Difference,
    classify_traces,
    compare_observations,
)
from .semantic import SemanticVerdict, evaluate_probe

__all__ = ["Oracle", "OracleResult", "MODES"]

#: The three modes in the order the paper presents them.
MODES: tuple[Mode, ...] = (
    Mode.WITHOUT_COMPILATION,
    Mode.COMPILED_WITHOUT_STATE,
    Mode.COMPILED_WITH_STATE,
)


@dataclass(frozen=True)
class OracleResult:
    """Everything the oracle learned about one sequence."""

    sequence: Sequence
    traces: Mapping[Mode, Trace]
    classification: Classification
    candidate: Candidate | None = None
    semantic: tuple[SemanticVerdict, ...] = ()

    @property
    def clean(self) -> bool:
        return self.candidate is None


class Oracle:
    """Runs sequences under the three modes and flags candidates."""

    def __init__(
        self,
        adapter,
        policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
    ) -> None:
        self.adapter = adapter
        self.policy = policy

    # running
    def run_modes(self, sequence: Sequence) -> Mapping[Mode, Trace]:
        """Run ``sequence`` under all three modes from the same initial state."""
        return {mode: self.adapter.run(sequence, mode) for mode in MODES}

    # checking
    def check(self, sequence: Sequence) -> OracleResult:
        traces = self.run_modes(sequence)
        classification = classify_traces(sequence, traces, self.policy)
        semantic = tuple(
            _safe_evaluate(probe, self.policy)
            for probe in self.adapter.semantic_probes(sequence, traces)
        )
        violations = tuple(verdict for verdict in semantic if not verdict.consistent)

        candidate: Candidate | None = None
        if classification.is_candidate:
            candidate = Candidate(
                sequence=sequence,
                target_step_id=classification.first_step_id or sequence.target().step_id,
                failing_modes=classification.failing_modes,
                symptoms=_symptoms_from_classification(classification),
                reason=classification.reason,
            )
        elif violations:
            probe = violations[0]
            candidate = Candidate(
                sequence=sequence,
                target_step_id=probe.step_id or sequence.target().step_id,
                failing_modes=(),
                symptoms=(
                    SymptomSignature(
                        step_id=probe.step_id or sequence.target().step_id,
                        path=f"$.semantic.{probe.kind}",
                        kind=f"semantic:{probe.kind}",
                        modes=(),
                    ),
                ),
                reason=probe.reason,
                semantic_obligation=probe.obligation,
            )

        return OracleResult(
            sequence=sequence,
            traces=traces,
            classification=classification,
            candidate=candidate,
            semantic=semantic,
        )

    # helpers
    def compare_step(self, step_id: str, traces: Mapping[Mode, Trace]):
        """Compare one step's observations across the three modes."""
        reference = _observation(traces[Mode.WITHOUT_COMPILATION], step_id)
        fresh = _observation(traces[Mode.COMPILED_WITHOUT_STATE], step_id)
        persistent = _observation(traces[Mode.COMPILED_WITH_STATE], step_id)
        return {
            Mode.COMPILED_WITHOUT_STATE: compare_observations(reference, fresh, self.policy),
            Mode.COMPILED_WITH_STATE: compare_observations(reference, persistent, self.policy),
        }


def _observation(trace: Trace, step_id: str) -> Observation:
    return trace.observation(step_id)


def _safe_evaluate(probe, policy: NumericPolicy) -> SemanticVerdict:
    """A probe whose preconditions do not hold makes no claim, ever."""
    try:
        return evaluate_probe(probe, policy)
    except Exception as error:
        return SemanticVerdict(
            kind=probe.kind,
            step_id=probe.step_id,
            label=probe.label,
            consistent=True,
            reason=f"check unavailable: {type(error).__name__}: {error}",
        )


def _symptoms_from_classification(
    classification: Classification,
) -> tuple[SymptomSignature, ...]:
    symptoms: list[SymptomSignature] = []
    for difference in classification.differences:
        symptoms.append(symptom_from_difference(classification, difference))
    return tuple(symptoms)


def symptom_from_difference(
    classification: Classification, difference: Difference
) -> SymptomSignature:
    modes = tuple(mode.value for mode in classification.failing_modes)
    return SymptomSignature(
        step_id=classification.first_step_id or "",
        path=difference.path,
        kind=difference.kind,
        modes=modes,
    )
