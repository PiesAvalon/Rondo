"""Stateful candidate refinement: confirmation, classification, minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .domain import (
    Candidate,
    Mode,
    Observation,
    Sequence,
    Step,
    SymptomSignature,
    Witness,
    WitnessClass,
)
from .numeric import DEFAULT_NUMERIC_POLICY, NumericPolicy
from .observation import Comparison, Equivalence, compare_observations
from .semantic import evaluate_probe

__all__ = [
    "RefinementInvariant",
    "Refiner",
    "RefinementConfig",
    "ControlledReplay",
    "classify_replay",
]


@dataclass(frozen=True)
class RefinementConfig:
    """Limits and switches of refinement."""

    max_evaluations: int = 400
    allow_input_simplification: bool = True


@dataclass(frozen=True)
class ControlledReplay:
    """The five aligned observations of one candidate target."""

    reference: Observation
    kept: Observation
    reset: Observation
    standalone_reference: Observation
    standalone_fresh: Observation
    valid: bool = True
    reason: str = ""


@dataclass(frozen=True)
class RefinementInvariant:
    """What minimization must preserve (the paper's ``ι``)."""

    witness_class: WitnessClass
    target_step_id: str
    symptom_kinds: frozenset[str]
    symptom_paths: frozenset[str] = frozenset()
    semantic_obligation: str | None = None


class Refiner:
    """Confirms and minimizes candidates produced by the oracle."""

    def __init__(
        self,
        adapter,
        config: RefinementConfig | None = None,
        policy: NumericPolicy = DEFAULT_NUMERIC_POLICY,
    ) -> None:
        self.adapter = adapter
        self.config = config or RefinementConfig()
        self.policy = policy
        self.evaluations = 0

    # confirmation
    def confirm(self, candidate: Candidate) -> Witness:
        prefix = candidate.prefix
        target = candidate.target
        replay = self._replay(prefix, target)

        if not replay.valid:
            return self._undetermined(candidate, replay.reason)

        kept_comparison = compare_observations(replay.reference, replay.kept, self.policy)
        reset_comparison = compare_observations(replay.reference, replay.reset, self.policy)

        kept_incorrect = kept_comparison.result is Equivalence.DIFFERENT
        reset_incorrect = reset_comparison.result is Equivalence.DIFFERENT

        standalone_reproduces: bool | None = None
        witness_class: WitnessClass
        reason: str

        if kept_incorrect and not reset_incorrect and prefix.length > 0:
            witness_class = WitnessClass.STATEFUL
            reason = "incorrect with compiler state retained, correct after reset"
        elif reset_incorrect:
            standalone_comparison = compare_observations(
                replay.standalone_reference, replay.standalone_fresh, self.policy
            )
            standalone_reproduces = standalone_comparison.result is Equivalence.DIFFERENT
            if standalone_reproduces:
                witness_class = WitnessClass.POINT_WISE
                reason = "reproduces without the prefix's program or compiler state"
            else:
                witness_class = WitnessClass.HISTORY_EXPOSED
                reason = "needs the prefix's program state but not its compiler state"
        else:
            witness_class = WitnessClass.UNDETERMINED
            reason = "no reproducible disagreement under the aligned replay"

        symptoms = _symptoms(candidate.target_step_id, kept_comparison, reset_comparison)
        if not symptoms:
            symptoms = candidate.symptoms

        return Witness(
            candidate=candidate,
            witness_class=witness_class,
            kept_correct=not kept_incorrect,
            reset_correct=not reset_incorrect,
            standalone_reproduces=standalone_reproduces,
            minimized_sequence=candidate.sequence,
            preserved=symptoms,
            reason=reason,
        )

    # minimization
    def minimize(self, witness: Witness) -> Witness:
        if not witness.is_confirmed:
            return witness

        target_step_id = witness.candidate.target_step_id
        invariant = RefinementInvariant(
            witness_class=witness.witness_class,
            target_step_id=target_step_id,
            symptom_kinds=frozenset(symptom.kind for symptom in witness.preserved),
            symptom_paths=frozenset(symptom.path for symptom in witness.preserved),
            semantic_obligation=witness.candidate.semantic_obligation,
        )

        original = witness.minimized_sequence
        # Anything after the exposing invocation cannot affect it.
        target_index = next(
            index
            for index, step in enumerate(original.steps)
            if step.step_id == target_step_id
        )
        current = original.prefix(target_index + 1)
        removed: list[str] = []
        evaluations = 0

        # 1. coarse-to-fine chunk deletion of prefix steps
        granularity = 2
        while evaluations <= self.config.max_evaluations:
            indices = _deletable_indices(current, target_step_id)
            if not indices:
                break
            chunk = max(1, -(-len(indices) // granularity))
            progressed = False
            for start in range(0, len(indices), chunk):
                block = indices[start : start + chunk]
                try:
                    candidate = current.drop(block)
                except ValueError:
                    continue  # e.g. a later step repeats a dropped step
                evaluations += 1
                if self._holds(candidate, invariant):
                    removed.extend(current.steps[index].step_id for index in block)
                    current = candidate
                    granularity = max(2, granularity - 1)
                    progressed = True
                    break
                if evaluations > self.config.max_evaluations:
                    break
            if progressed:
                continue
            if granularity >= len(indices):
                break
            granularity = min(len(indices), granularity * 2)

        # 2. single-step sweep (1-minimality)
        index = len(current.steps) - 1
        while index >= 0 and evaluations <= self.config.max_evaluations:
            step = current.steps[index]
            if step.step_id != target_step_id and _is_removable(current, index, target_step_id):
                try:
                    candidate = current.drop([index])
                except ValueError:
                    index -= 1
                    continue
                evaluations += 1
                if self._holds(candidate, invariant):
                    removed.append(step.step_id)
                    current = candidate
                    index = min(index, len(current.steps) - 1)
                    continue
            index -= 1

        # 3. input simplification of retained invocations
        if self.config.allow_input_simplification:
            current, extra_evaluations = self._simplify_inputs(current, invariant)
            evaluations += extra_evaluations

        return Witness(
            candidate=witness.candidate,
            witness_class=witness.witness_class,
            kept_correct=witness.kept_correct,
            reset_correct=witness.reset_correct,
            standalone_reproduces=witness.standalone_reproduces,
            minimized_sequence=current,
            preserved=witness.preserved,
            evaluations=evaluations,
            removed_step_ids=tuple(removed),
            reason=witness.reason,
        )

    # preservation predicate
    def _holds(self, sequence: Sequence, invariant: RefinementInvariant) -> bool:
        try:
            target = _step_by_id(sequence, invariant.target_step_id)
        except KeyError:
            return False
        prefix = _prefix_before(sequence, invariant.target_step_id)
        replay = self._replay(prefix, target)
        if not replay.valid:
            return False
        witness_class, symptoms = _classify_replay(replay, self.policy)
        if witness_class is not invariant.witness_class:
            return False
        kinds = {symptom.kind for symptom in symptoms}
        if invariant.symptom_kinds and not invariant.symptom_kinds <= kinds:
            return False
        if invariant.symptom_paths and not _paths_compatible(invariant.symptom_paths, symptoms):
            return False
        if invariant.semantic_obligation is not None:
            if not self._obligation_holds(sequence, invariant.semantic_obligation):
                return False
        return True

    def _obligation_holds(self, sequence: Sequence, obligation: str) -> bool:
        """The same declared semantic obligation must still be violated."""
        traces = {mode: self.adapter.run(sequence, mode) for mode in Mode}
        for probe in self.adapter.semantic_probes(sequence, traces):
            try:
                verdict = evaluate_probe(probe, self.policy)
            except Exception:
                continue
            if not verdict.consistent and verdict.obligation == obligation:
                return True
        return False

    def _simplify_inputs(
        self, sequence: Sequence, invariant: RefinementInvariant
    ) -> tuple[Sequence, int]:
        evaluations = 0
        for index in range(len(sequence.steps)):
            if evaluations > self.config.max_evaluations:
                break
            step = sequence.steps[index]
            spec = getattr(step, "spec", None)
            if spec is None:
                continue
            default = self.adapter.space.default_spec()
            simplified = step.spec
            changed = False
            for field_name in ("shape", "dtype", "layout", "device", "value_class", "value_seed", "scale"):
                if getattr(simplified, field_name) != getattr(default, field_name):
                    simplified = simplified.with_fields(**{field_name: getattr(default, field_name)})
                    changed = True
                    break
            if not changed:
                continue
            new_step = self.adapter.make_invocation(step.step_id, simplified, step.repeat_of)
            candidate = sequence.replace_step(index, new_step)
            evaluations += 1
            if self._holds(candidate, invariant):
                sequence = candidate
        return sequence, evaluations

    # adapter replay
    def _replay(self, prefix: Sequence, target: Step) -> ControlledReplay:
        try:
            reference = self.adapter.observe_aligned(prefix, target, compiled=False, reset=False)
            kept = self.adapter.observe_aligned(prefix, target, compiled=True, reset=False)
            reset = self.adapter.observe_aligned(prefix, target, compiled=True, reset=True)
            standalone_reference = self.adapter.observe_aligned(
                _empty(prefix), target, compiled=False, reset=False
            )
            standalone_fresh = self.adapter.observe_aligned(
                _empty(prefix), target, compiled=True, reset=True
            )
        except Exception as error:  # framework failure is unavailable evidence
            self.evaluations += 1
            return ControlledReplay(
                reference=_placeholder(target),
                kept=_placeholder(target),
                reset=_placeholder(target),
                standalone_reference=_placeholder(target),
                standalone_fresh=_placeholder(target),
                valid=False,
                reason=f"controlled replay unavailable: {type(error).__name__}: {error}",
            )
        self.evaluations += 1
        if any(
            observation.unsupported_fields
            for observation in (reference, kept, reset, standalone_reference, standalone_fresh)
        ):
            return ControlledReplay(
                reference=reference,
                kept=kept,
                reset=reset,
                standalone_reference=standalone_reference,
                standalone_fresh=standalone_fresh,
                valid=False,
                reason="a required observation field is unavailable",
            )
        return ControlledReplay(
            reference=reference,
            kept=kept,
            reset=reset,
            standalone_reference=standalone_reference,
            standalone_fresh=standalone_fresh,
        )

    def _undetermined(self, candidate: Candidate, reason: str) -> Witness:
        return Witness(
            candidate=candidate,
            witness_class=WitnessClass.UNDETERMINED,
            kept_correct=True,
            reset_correct=True,
            standalone_reproduces=None,
            minimized_sequence=candidate.sequence,
            preserved=candidate.symptoms,
            reason=reason,
        )


# Classification helpers (also used by the engine to explain decisions)
def classify_replay(
    replay: ControlledReplay, policy: NumericPolicy = DEFAULT_NUMERIC_POLICY
) -> tuple[WitnessClass, tuple[SymptomSignature, ...]]:
    return _classify_replay(replay, policy)


def _classify_replay(
    replay: ControlledReplay, policy: NumericPolicy
) -> tuple[WitnessClass, tuple[SymptomSignature, ...]]:
    if not replay.valid:
        return WitnessClass.UNDETERMINED, ()
    kept = compare_observations(replay.reference, replay.kept, policy)
    reset = compare_observations(replay.reference, replay.reset, policy)
    kept_incorrect = kept.result is Equivalence.DIFFERENT
    reset_incorrect = reset.result is Equivalence.DIFFERENT
    symptoms = _symptoms(replay.reference.step_id, kept, reset)
    # retained-wrong + reset-correct => stateful; otherwise standalone decides
    if kept_incorrect and not reset_incorrect:
        return WitnessClass.STATEFUL, symptoms
    if reset_incorrect:
        standalone = compare_observations(
            replay.standalone_reference, replay.standalone_fresh, policy
        )
        if standalone.result is Equivalence.DIFFERENT:
            return WitnessClass.POINT_WISE, symptoms
        return WitnessClass.HISTORY_EXPOSED, symptoms
    return WitnessClass.UNDETERMINED, symptoms


def _symptoms(
    target_step_id: str,
    kept: Comparison,
    reset: Comparison,
) -> tuple[SymptomSignature, ...]:
    symptoms: list[SymptomSignature] = []
    for mode, comparison in (
        (Mode.COMPILED_WITH_STATE.value, kept),
        (Mode.COMPILED_WITHOUT_STATE.value, reset),
    ):
        for difference in comparison.differences:
            symptoms.append(
                SymptomSignature(
                    step_id=target_step_id,
                    path=difference.path,
                    kind=difference.kind,
                    modes=(mode,),
                )
            )
    return tuple(symptoms)


def _paths_compatible(pinned: frozenset[str], symptoms: Iterable[SymptomSignature]) -> bool:
    """A pinned path is preserved when a symptom keeps its root and kind shape."""
    available = {_path_shape(symptom.path) for symptom in symptoms}
    return all(_path_shape(path) in available for path in pinned)


def _path_shape(path: str) -> str:
    out = []
    for char in path:
        if char == "[":
            break
        out.append(char)
    return "".join(out)


# Sequence helpers
def _empty(sequence: Sequence) -> Sequence:
    return sequence.truncate(0)


def _step_by_id(sequence: Sequence, step_id: str) -> Step:
    for step in sequence.steps:
        if step.step_id == step_id:
            return step
    raise KeyError(step_id)


def _prefix_before(sequence: Sequence, step_id: str) -> Sequence:
    for index, step in enumerate(sequence.steps):
        if step.step_id == step_id:
            return sequence.prefix(index)
    raise KeyError(step_id)


def _deletable_indices(sequence: Sequence, target_step_id: str) -> list[int]:
    indices = []
    for index, step in enumerate(sequence.steps):
        if step.step_id == target_step_id:
            break
        indices.append(index)
    return indices


def _is_removable(sequence: Sequence, index: int, target_step_id: str) -> bool:
    step = sequence.steps[index]
    if step.step_id == target_step_id:
        return False
    target_index = next(
        (i for i, item in enumerate(sequence.steps) if item.step_id == target_step_id),
        None,
    )
    if target_index is not None and index > target_index:
        return False
    # Removing a step that a later step repeats would make the sequence illegal.
    for later in sequence.steps:
        if getattr(later, "repeat_of", None) == step.step_id:
            return False
    return True


def _placeholder(target: Step) -> Observation:
    return Observation(step_index=0, step_id=target.step_id)
