"""Rondo: stateful testing for deep-learning compilers."""

from __future__ import annotations

from .domain import (
    BehaviorKey,
    Candidate,
    CompilerEvent,
    Decision,
    Guard,
    GuardGoal,
    Invocation,
    Mode,
    Observation,
    ProgramState,
    ResetBoundary,
    Scenario,
    Sequence,
    StateOperation,
    Step,
    StepSpec,
    SymptomSignature,
    TensorValue,
    Test,
    Trace,
    Witness,
    WitnessClass,
)
from .engine import CampaignResult, Rondo, RondoConfig
from .feedback import BehaviorFeedback, RewardConfig
from .generation import MUTATION_OPERATORS, Mutator, OperatorSpace, Proposal
from .numeric import DEFAULT_NUMERIC_POLICY, NumericPolicy
from .observation import (
    Classification,
    Comparison,
    DivergenceClass,
    Equivalence,
    classify_traces,
    compare_observations,
)
from .oracle import Oracle, OracleResult
from .refinement import RefinementConfig, Refiner
from .sampler import RondoMCMC, RoundRecord, SamplerConfig
from .semantic import CHECKS, SemanticProbe, SemanticVerdict, evaluate_probe
from .synthesis import SynthesisOutcome, Synthesizer

__all__ = [
    "BehaviorFeedback",
    "BehaviorKey",
    "CHECKS",
    "CampaignResult",
    "Candidate",
    "Classification",
    "CompilerEvent",
    "Comparison",
    "Decision",
    "DivergenceClass",
    "DEFAULT_NUMERIC_POLICY",
    "Equivalence",
    "Guard",
    "GuardGoal",
    "Invocation",
    "MUTATION_OPERATORS",
    "Mode",
    "Mutator",
    "NumericPolicy",
    "Observation",
    "OperatorSpace",
    "Oracle",
    "OracleResult",
    "ProgramState",
    "Proposal",
    "RefinementConfig",
    "Refiner",
    "ResetBoundary",
    "RewardConfig",
    "Rondo",
    "RondoConfig",
    "RondoMCMC",
    "RoundRecord",
    "SamplerConfig",
    "Scenario",
    "SemanticProbe",
    "SemanticVerdict",
    "Sequence",
    "StateOperation",
    "Step",
    "StepSpec",
    "SymptomSignature",
    "SynthesisOutcome",
    "Synthesizer",
    "TensorValue",
    "Test",
    "Trace",
    "Witness",
    "WitnessClass",
    "classify_traces",
    "compare_observations",
    "evaluate_probe",
]
__version__ = "1.0.0"
