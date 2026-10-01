from __future__ import annotations

from dataclasses import replace

from xithink.models import (
    Counterexample,
    Hypothesis,
    MemoryMatch,
    MetacognitiveAssessment,
    clamp,
)
from xithink.modules.concept_graph import ConceptGraph
from xithink.modules.symbolic import SymbolicReasoner


class CounterexampleEngine:
    """Actively searches existing knowledge for evidence against a candidate."""

    def find(
        self,
        hypothesis: Hypothesis,
        reasoner: SymbolicReasoner,
        graph: ConceptGraph,
        recalled: list[MemoryMatch],
    ) -> list[Counterexample]:
        out: list[Counterexample] = []
        fact_by_key = {f.key(): f for f in reasoner.facts}

        for atom in hypothesis.supporting_atoms:
            opposite = fact_by_key.get(atom.opposite_key())
            if opposite:
                strength = min(1.0, 0.55 + 0.45 * opposite.confidence)
                out.append(Counterexample(
                    statement=f"反例事实：{opposite.render()}",
                    strength=strength,
                    reason="存在与候选支持事实直接相反的已知事实",
                    supporting_atoms=[opposite],
                ))
            for edge in graph.opposite_edges(atom):
                strength = min(0.95, 0.45 + edge.weight * 0.45)
                statement = f"反向图关系：{edge.source} {'不' if edge.polarity < 0 else ''}{edge.relation} {edge.target}"
                if not any(c.statement == statement for c in out):
                    out.append(Counterexample(
                        statement=statement,
                        strength=strength,
                        reason="概念图中存在相反极性的关系",
                    ))

        # Recalled negative memories are weaker than exact symbolic conflicts,
        # but they are useful prompts for verification.
        key_terms = {a.subject for a in hypothesis.supporting_atoms} | {a.object for a in hypothesis.supporting_atoms}
        for match in recalled:
            if match.score < 0.30 or not key_terms.intersection(match.concepts):
                continue
            if any(marker in match.text for marker in (" 不", "不是", "不能", "反对", "冲突", "反例")):
                out.append(Counterexample(
                    statement=f"相关记忆可能构成反例：{match.text}",
                    strength=min(0.72, match.score * 0.78),
                    reason="语义相似的长期记忆包含否定或冲突信号",
                ))

        # Deduplicate and rank by strength.
        dedup: dict[str, Counterexample] = {}
        for item in out:
            old = dedup.get(item.statement)
            if old is None or item.strength > old.strength:
                dedup[item.statement] = item
        return sorted(dedup.values(), key=lambda x: (x.strength, x.statement), reverse=True)[:5]


class MetacognitionEngine:
    """Second-order review before a hypothesis is allowed into decision ranking."""

    def __init__(self, counterexamples: CounterexampleEngine | None = None) -> None:
        self.counterexamples = counterexamples or CounterexampleEngine()

    def assess(
        self,
        hypothesis: Hypothesis,
        reasoner: SymbolicReasoner,
        graph: ConceptGraph,
        recalled: list[MemoryMatch],
    ) -> tuple[Hypothesis, MetacognitiveAssessment]:
        # A detected contradiction is itself a meta-level claim. The opposite
        # atoms support "there is a conflict"; they are not counterexamples to it.
        counters = [] if "conflict" in hypothesis.tags else self.counterexamples.find(
            hypothesis, reasoner, graph, recalled
        )
        counterevidence = max((c.strength for c in counters), default=0.0)

        # Memory support must be candidate-specific. Candidates with explicit
        # supporting atoms can borrow related memory; a memory-recall candidate
        # carries its own score. Questions/reflections are not boosted merely
        # because some memory happened to be retrieved for the input.
        h_concepts = {a.subject for a in hypothesis.supporting_atoms} | {a.object for a in hypothesis.supporting_atoms}
        if h_concepts:
            relevant = [m.score for m in recalled if h_concepts.intersection(m.concepts)]
            memory_support = max(relevant, default=hypothesis.memory_support)
        else:
            memory_support = hypothesis.memory_support

        if hypothesis.supporting_atoms:
            supported = sum(1 for a in hypothesis.supporting_atoms if a.confidence >= 0.45)
            evidence_coverage = supported / len(hypothesis.supporting_atoms)
        else:
            evidence_coverage = 0.0

        adjusted_evidence = clamp(
            hypothesis.evidence
            + memory_support * 0.12
            + evidence_coverage * 0.08
            - counterevidence * 0.28
        )
        adjusted_consistency = clamp(hypothesis.consistency * (1.0 - 0.68 * counterevidence))
        adjusted_uncertainty = clamp(
            hypothesis.uncertainty
            + counterevidence * 0.40
            - memory_support * 0.10
            - evidence_coverage * 0.06
        )
        needs_verification = (
            adjusted_uncertainty >= 0.62
            or counterevidence >= 0.35
            or (evidence_coverage < 0.5 and "inference" not in hypothesis.tags)
        )

        rationale = (
            f"memory={memory_support:.2f}, counter={counterevidence:.2f}, "
            f"coverage={evidence_coverage:.2f}, verify={needs_verification}"
        )
        assessment = MetacognitiveAssessment(
            original_statement=hypothesis.statement,
            adjusted_evidence=adjusted_evidence,
            adjusted_consistency=adjusted_consistency,
            adjusted_uncertainty=adjusted_uncertainty,
            memory_support=memory_support,
            counterevidence=counterevidence,
            evidence_coverage=evidence_coverage,
            needs_verification=needs_verification,
            counterexamples=counters,
            rationale=rationale,
        )
        reviewed = replace(
            hypothesis,
            evidence=adjusted_evidence,
            consistency=adjusted_consistency,
            uncertainty=adjusted_uncertainty,
            memory_support=memory_support,
            counterevidence=counterevidence,
            metadata={
                **hypothesis.metadata,
                "metacognition": {
                    "memory_support": memory_support,
                    "counterevidence": counterevidence,
                    "evidence_coverage": evidence_coverage,
                    "needs_verification": needs_verification,
                    "counterexamples": [c.statement for c in counters],
                    "rationale": rationale,
                },
            },
        )
        return reviewed, assessment

    def counterexample_reflections(
        self,
        hypotheses: list[Hypothesis],
        assessments: list[MetacognitiveAssessment],
    ) -> list[Hypothesis]:
        """Turn strong falsification results into explicit second-order thoughts."""
        reflections: list[Hypothesis] = []
        for hypothesis, assessment in zip(hypotheses, assessments):
            if "conflict" in hypothesis.tags or assessment.counterevidence < 0.45:
                continue
            strongest = assessment.counterexamples[0] if assessment.counterexamples else None
            if strongest is None:
                continue
            statement = (
                f"反例审查：候选『{hypothesis.statement}』受到『{strongest.statement}』挑战，"
                "应先区分条件、来源或适用范围再继续推断"
            )
            reflections.append(Hypothesis(
                statement=statement,
                supporting_atoms=list(strongest.supporting_atoms),
                evidence=strongest.strength,
                consistency=0.96,
                novelty=0.70,
                uncertainty=max(0.20, 1.0 - strongest.strength),
                tags=["counterexample", "metacognition"],
                memory_support=hypothesis.memory_support,
                counterevidence=0.0,
                metadata={
                    "metacognition": {
                        "memory_support": hypothesis.memory_support,
                        "counterevidence": 0.0,
                        "evidence_coverage": 1.0 if strongest.supporting_atoms else 0.5,
                        "needs_verification": True,
                        "counterexamples": [strongest.statement],
                        "rationale": "explicit counterexample reflection",
                    }
                },
            ))
        return reflections[:3]

    def review(
        self,
        hypotheses: list[Hypothesis],
        reasoner: SymbolicReasoner,
        graph: ConceptGraph,
        recalled: list[MemoryMatch],
    ) -> tuple[list[Hypothesis], list[MetacognitiveAssessment]]:
        reviewed: list[Hypothesis] = []
        assessments: list[MetacognitiveAssessment] = []
        for hypothesis in hypotheses:
            h, a = self.assess(hypothesis, reasoner, graph, recalled)
            reviewed.append(h)
            assessments.append(a)
        return reviewed, assessments
