from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import hashlib
import time
import uuid


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def stable_concept_id(name: str) -> str:
    canonical = " ".join(name.strip().split()).lower()
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"concept:{digest}"


@dataclass(frozen=True, slots=True)
class Atom:
    subject: str
    predicate: str
    object: str
    negated: bool = False
    confidence: float = 1.0
    source: str = "input"

    def key(self) -> tuple[str, str, str, bool]:
        return (self.subject, self.predicate, self.object, self.negated)

    def opposite_key(self) -> tuple[str, str, str, bool]:
        return (self.subject, self.predicate, self.object, not self.negated)

    def render(self) -> str:
        neg = "不" if self.negated else ""
        return f"{self.subject} {neg}{self.predicate} {self.object}"


@dataclass(frozen=True, slots=True)
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom
    name: str = "rule"
    confidence: float = 0.9


@dataclass(slots=True)
class Concept:
    canonical_name: str
    kind: str = "abstract"
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        self.canonical_name = " ".join(self.canonical_name.strip().split())
        if not self.id:
            self.id = stable_concept_id(self.canonical_name)
        self.aliases = list(dict.fromkeys(x.strip() for x in self.aliases if x.strip()))


@dataclass(slots=True)
class Event:
    actor: str
    action: str
    object: str | None = None
    time_ref: str | None = None
    negated: bool = False
    modality: str = "asserted"
    source: str = "input"
    confidence: float = 0.9
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def render(self) -> str:
        neg = "不" if self.negated else ""
        obj = f" {self.object}" if self.object else ""
        time_part = f" @ {self.time_ref}" if self.time_ref else ""
        return f"{self.actor} {neg}{self.action}{obj}{time_part}"


@dataclass(slots=True)
class SemanticFrame:
    raw_text: str
    concepts: list[str] = field(default_factory=list)
    concept_nodes: list[Concept] = field(default_factory=list)
    facts: list[Atom] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    question_intents: list[QuestionIntent] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    time_refs: list[str] = field(default_factory=list)
    source: str = "input"
    modality: str = "asserted"


@dataclass(slots=True)
class Hypothesis:
    statement: str
    supporting_atoms: list[Atom] = field(default_factory=list)
    evidence: float = 0.0
    consistency: float = 1.0
    novelty: float = 0.5
    uncertainty: float = 0.5
    tags: list[str] = field(default_factory=list)
    memory_support: float = 0.0
    counterevidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)




@dataclass(slots=True)
class QuestionIntent:
    kind: str
    raw: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    negated: bool = False


@dataclass(slots=True)
class SymbolicProof:
    conclusion: Atom
    kind: str
    confidence: float
    source: str
    rule_name: str | None = None
    premises: list["SymbolicProof"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": self.conclusion.render(),
            "kind": self.kind,
            "confidence": self.confidence,
            "source": self.source,
            "rule_name": self.rule_name,
            "premises": [p.to_dict() for p in self.premises],
        }


@dataclass(slots=True)
class SymbolicAnswer:
    answered: bool
    text: str
    confidence: float = 0.0
    atoms: list[Atom] = field(default_factory=list)
    proofs: list[SymbolicProof] = field(default_factory=list)
    intent: QuestionIntent | None = None


@dataclass(slots=True)
class Experience:
    content: str
    concepts: list[str] = field(default_factory=list)
    kind: str = "interaction"
    context: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None
    reward: float = 0.0
    confidence: float = 0.8
    source: str = "interaction"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SubconsciousTrace:
    cue: str
    association: str
    strength: float = 0.2
    valence: float = 0.0
    exposures: int = 1
    source: str = "experience"
    last_activated: float = field(default_factory=time.time)


@dataclass(slots=True)
class LearnedRuleCandidate:
    signature: str
    premises: list[Atom]
    conclusion: Atom
    support: int = 0
    counterexamples: int = 0
    confidence: float = 0.5
    status: str = "candidate"
    environments: list[str] = field(default_factory=list)
    evidence_experience_ids: list[str] = field(default_factory=list)
    counterexample_experience_ids: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def as_rule(self) -> Rule:
        return Rule(
            tuple(self.premises),
            self.conclusion,
            name=f"experience:{self.signature}",
            confidence=self.confidence,
        )


@dataclass(slots=True)
class EvolvedRuleVariant:
    id: str
    base_signature: str
    premises: list[Atom]
    conclusion: Atom
    conditions: list[Atom] = field(default_factory=list)
    support: int = 0
    counterexamples: int = 0
    confidence: float = 0.5
    status: str = "candidate"
    environments: list[str] = field(default_factory=list)
    supporting_experience_ids: list[str] = field(default_factory=list)
    opposing_experience_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    updated_at: float = field(default_factory=time.time)

    def as_rule(self) -> Rule:
        return Rule(
            tuple([*self.premises, *self.conditions]),
            self.conclusion,
            name=f"evolved:{self.id}",
            confidence=self.confidence,
        )


@dataclass(slots=True)
class PerspectiveChain:
    name: str
    stance: str
    conclusion: str
    confidence: float
    conditions: list[str] = field(default_factory=list)
    environment: str | None = None
    proof: dict[str, Any] | None = None
    source: str = "symbolic"


@dataclass(slots=True)
class MultiPerspectiveReport:
    target: str
    environment: str | None = None
    chains: list[PerspectiveChain] = field(default_factory=list)
    unresolved_conditions: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "environment": self.environment,
            "chains": [asdict(c) for c in self.chains],
            "unresolved_conditions": list(self.unresolved_conditions),
            "summary": self.summary,
        }


@dataclass(slots=True)
class PersonalSymbol:
    id: str
    name: str
    members: list[str]
    strength: float = 0.0
    valence: float = 0.0
    exposures: int = 0
    environments: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class WorldviewBelief:
    atom: Atom
    environment: str
    support: float = 0.0
    opposition: float = 0.0
    exposures: int = 0
    confidence: float = 0.5
    last_seen: float = field(default_factory=time.time)


@dataclass(slots=True)
class PerspectiveResonance:
    score: float
    shared_concepts: list[str] = field(default_factory=list)
    experience_ids: list[str] = field(default_factory=list)
    valence: float = 0.0
    activation: dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass(slots=True)
class KnowledgeClaim:
    subject: str
    predicate: str
    object: str
    negated: bool = False
    confidence: float = 0.5
    provider: str = "external"
    query: str = ""
    status: str = "quarantined"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def as_atom(self, source_prefix: str = "verified-general-knowledge") -> Atom:
        return Atom(
            self.subject, self.predicate, self.object, self.negated,
            self.confidence, f"{source_prefix}:{self.provider}",
        )


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    relation: str
    target: str
    weight: float = 1.0
    polarity: int = 1
    evidence_count: int = 1
    source_type: str = "fact"


@dataclass(slots=True)
class MemoryMatch:
    item_id: str
    kind: str
    text: str
    score: float
    concepts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Counterexample:
    statement: str
    strength: float
    reason: str
    supporting_atoms: list[Atom] = field(default_factory=list)


@dataclass(slots=True)
class MetacognitiveAssessment:
    original_statement: str
    adjusted_evidence: float
    adjusted_consistency: float
    adjusted_uncertainty: float
    memory_support: float
    counterevidence: float
    evidence_coverage: float
    needs_verification: bool
    counterexamples: list[Counterexample] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class Decision:
    hypothesis: Hypothesis
    score: float
    components: dict[str, float]
    rationale: str


@dataclass(slots=True)
class Thought:
    cycle: int
    kind: str
    content: str
    confidence: float
    concepts: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class RuleGenealogyNode:
    id: str
    rule_ref: str
    rule_kind: str
    base_signature: str
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    mutation: str = "origin"
    status: str = "candidate"
    premises: list[Atom] = field(default_factory=list)
    conclusion: Atom | None = None
    conditions: list[Atom] = field(default_factory=list)
    support: int = 0
    counterexamples: int = 0
    confidence: float = 0.5
    environments: list[str] = field(default_factory=list)
    rationale: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ActiveHypothesisTest:
    id: str
    base_signature: str
    hypothesis_a: str
    hypothesis_b: str
    target: str
    discriminating_conditions: list[str] = field(default_factory=list)
    question: str = ""
    expected_information_gain: float = 0.0
    priority: float = 0.0
    environment: str | None = None
    status: str = "proposed"
    outcome: str | None = None
    resolved_experience_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class CognitiveProfile:
    name: str = "balanced"
    skepticism: float = 0.65
    exploration: float = 0.75
    rule_min_support: int = 3
    rule_verification_confidence: float = 0.74
    reject_counterexamples: int = 2
    condition_min_support: int = 2
    condition_verification_confidence: float = 0.72
    condition_max_counterexample_rate: float = 0.12
    active_test_aggressiveness: float = 0.70
    active_test_budget: int = 3
    max_condition_terms: int = 2
    environment_diversity: float = 0.70
    experience_noise: float = 0.06


@dataclass(slots=True)
class AgeCognitionSnapshot:
    age: int
    experiences: int
    learned_candidates: int
    verified_rules: int
    evolved_variants: int
    active_evolved_rules: int
    genealogy_nodes: int
    genealogy_max_generation: int
    proposed_tests: int
    resolved_tests: int
    symbolic_accuracy: float
    abstention_accuracy: float
    contradiction_rate: float
    worldview_beliefs: int
    personal_symbols: int
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LifeSimulationReport:
    profile: CognitiveProfile
    total_experiences: int
    max_age: int
    seed: int
    checkpoints: list[AgeCognitionSnapshot] = field(default_factory=list)
    final_genealogy: dict[str, Any] = field(default_factory=dict)
    final_active_tests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": asdict(self.profile),
            "total_experiences": self.total_experiences,
            "max_age": self.max_age,
            "seed": self.seed,
            "checkpoints": [asdict(x) for x in self.checkpoints],
            "final_genealogy": self.final_genealogy,
            "final_active_tests": self.final_active_tests,
        }


@dataclass(slots=True)
class SelfState:
    identity: str = "XThink"
    values: dict[str, float] = field(default_factory=lambda: {
        "truth": 0.90,
        "curiosity": 0.90,
        "empathy": 0.75,
        "freedom": 0.75,
        "safety": 0.70,
    })
    traits: dict[str, float] = field(default_factory=lambda: {
        "skepticism": 0.65,
        "novelty_seeking": 0.80,
        "risk_tolerance": 0.45,
    })
    active_goals: list[str] = field(default_factory=lambda: ["形成可解释的连续思考"])
    unresolved_questions: list[str] = field(default_factory=list)
    thought_count: int = 0

    def normalize(self) -> None:
        self.values = {k: clamp(v) for k, v in self.values.items()}
        self.traits = {k: clamp(v) for k, v in self.traits.items()}
