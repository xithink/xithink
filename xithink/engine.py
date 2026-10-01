from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from xithink.models import (
    Atom, CognitiveProfile, Experience, KnowledgeClaim, LifeSimulationReport, Rule, SelfState, SemanticFrame,
    SymbolicAnswer, Thought,
)
from xithink.modules.semantic import SemanticRecognizer
from xithink.modules.symbolic import SymbolicReasoner
from xithink.modules.association import AssociativeNetwork
from xithink.modules.memory import MemoryStore
from xithink.modules.memory_layers import LongTermMemory, ExperienceMemory, SubconsciousMemory
from xithink.modules.knowledge import GeneralKnowledgeProvider, KnowledgeGateway
from xithink.modules.concepts import ConceptRegistry
from xithink.modules.concept_graph import ConceptGraph
from xithink.modules.embedding_memory import EmbeddingMemory
from xithink.modules.hypothesis import HypothesisGenerator
from xithink.modules.metacognition import MetacognitionEngine
from xithink.modules.decision import DecisionEngine
from xithink.modules.self_model import SelfModel
from xithink.modules.experience_learning import ExperienceRuleLearner
from xithink.modules.personal_symbols import PersonalSymbolSystem
from xithink.modules.worldview import WorldviewModel
from xithink.modules.perspective import PerspectiveResonanceEngine
from xithink.modules.rule_evolution import RuleEvolutionEngine
from xithink.modules.multi_perspective import MultiPerspectiveAnalyzer
from xithink.modules.rule_genealogy import RuleGenealogyTracker
from xithink.modules.active_testing import ActiveHypothesisTester


class XThinkEngine:
    def __init__(
        self,
        memory_path: str | Path = ":memory:",
        state: SelfState | None = None,
        knowledge_provider: GeneralKnowledgeProvider | None = None,
    ) -> None:
        self.semantic = SemanticRecognizer()
        self.reasoner = SymbolicReasoner()
        self.association = AssociativeNetwork()
        self.memory = MemoryStore(memory_path)
        self.concepts = ConceptRegistry(self.memory)
        self.graph = ConceptGraph(self.memory)
        self.embedding_memory = EmbeddingMemory(self.memory)

        # 0.4 memory authority layers.
        self.long_term = LongTermMemory(self.memory)
        self.experience_memory = ExperienceMemory(self.memory, self.embedding_memory)
        self.subconscious_memory = SubconsciousMemory(self.memory)
        self.knowledge = KnowledgeGateway(self.memory, knowledge_provider)

        # 0.5 experience-to-rule, private symbols, worldview and perspective resonance.
        self.experience_rules = ExperienceRuleLearner(self.memory)
        self.rule_evolution = RuleEvolutionEngine(self.memory)
        self.rule_genealogy = RuleGenealogyTracker(self.memory)
        self.active_testing = ActiveHypothesisTester(self.memory)
        self.personal_symbols = PersonalSymbolSystem(self.memory)
        self.worldview = WorldviewModel(self.memory)
        self.perspective = PerspectiveResonanceEngine()
        self.multi_perspective = MultiPerspectiveAnalyzer()
        self.cognitive_profile = CognitiveProfile()

        self.hypotheses = HypothesisGenerator()
        self.metacognition = MetacognitionEngine()
        self.decision = DecisionEngine()
        self.self_model = SelfModel()
        self.state = state or self._load_state() or SelfState()
        self.last_trace: dict = {}

        persisted_facts = self.long_term.facts()
        # Rebuild inferred facts from their premises on every restart so their
        # proof trees stay reconstructable instead of degrading into opaque
        # persisted assertions.
        for fact in persisted_facts:
            if not fact.source.startswith("inference:"):
                self.reasoner.add_fact(fact)
        for rule in self.long_term.rules():
            # Adaptive rules are rebuilt from their evidence stores so a rule
            # that later gains counterexamples cannot survive as a stale truth.
            if rule.name.startswith(("experience:", "evolved:")):
                continue
            self.reasoner.add_rule(rule)
        self._install_core_rules()
        self._refresh_adaptive_rules()
        self.reasoner.infer()

        if self.memory.graph_edge_count() == 0:
            for fact in persisted_facts:
                self.graph.ingest_fact(fact)
        if self.memory.embedding_count() == 0:
            for fact in persisted_facts:
                self.embedding_memory.index_fact(fact)

    def _load_state(self) -> SelfState | None:
        data = self.memory.load_json("self_state")
        if not data:
            return None
        return SelfState(
            identity=data.get("identity", "XThink"),
            values=dict(data.get("values", {})),
            traits=dict(data.get("traits", {})),
            active_goals=list(data.get("active_goals", [])),
            unresolved_questions=list(data.get("unresolved_questions", [])),
            thought_count=int(data.get("thought_count", 0)),
        )

    def _save_state(self) -> None:
        self.memory.save_json("self_state", asdict(self.state))

    def _install_core_rules(self) -> None:
        self.reasoner.add_rule(Rule(
            (Atom("?x", "意味着", "?y"), Atom("?y", "导致", "?z")),
            Atom("?x", "间接导致", "?z"),
            name="meaning-cause-chain", confidence=0.82,
        ))
        self.reasoner.add_rule(Rule(
            (Atom("?x", "需要", "?y"), Atom("?z", "限制", "?y")),
            Atom("?z", "可能影响", "?x"),
            name="requirement-limitation", confidence=0.78,
        ))

    def _refresh_adaptive_rules(self) -> None:
        self.reasoner.remove_rules_by_prefix("experience:", "evolved:")
        for learned_rule in self.experience_rules.verified_rules():
            self.reasoner.add_rule(learned_rule)
        for evolved_rule in self.rule_evolution.verified_rules():
            self.reasoner.add_rule(evolved_rule)

    def apply_cognitive_profile(self, profile: CognitiveProfile) -> None:
        """Adjust learning/verification policy without changing symbolic truth data."""
        self.cognitive_profile = profile
        self.state.traits["skepticism"] = max(0.0, min(1.0, profile.skepticism))
        self.state.traits["novelty_seeking"] = max(0.0, min(1.0, profile.exploration))
        self.experience_rules.min_support = max(2, int(profile.rule_min_support))
        self.experience_rules.verification_confidence = max(0.5, min(0.99, profile.rule_verification_confidence))
        self.experience_rules.reject_counterexamples = max(1, int(profile.reject_counterexamples))
        self.rule_evolution.min_condition_support = max(2, int(profile.condition_min_support))
        self.rule_evolution.verification_confidence = max(0.5, min(0.99, profile.condition_verification_confidence))
        self.rule_evolution.max_condition_terms = max(1, min(4, int(profile.max_condition_terms)))
        self.rule_evolution.max_counterexample_rate = max(0.0, min(0.35, profile.condition_max_counterexample_rate))
        self.active_testing.aggressiveness = max(0.0, min(1.0, profile.active_test_aggressiveness))
        self.active_testing.budget = max(1, int(profile.active_test_budget))
        self._save_state()

    def ingest_structured_experience(self, experience: Experience, *, learn: bool = True) -> Experience:
        """Ingest a structured episode without routing it through natural-language assertion parsing.

        Used by sensors/importers and the life simulator. The episode can shape
        experience/worldview/subconscious memory, but it does not become a
        universal long-term fact merely because it was observed in one context.
        """
        self.experience_memory.remember(experience)
        facts: list[Atom] = []
        for data in experience.context.get("fact_atoms", []):
            if not isinstance(data, dict) or not {"subject", "predicate", "object"} <= set(data):
                continue
            facts.append(Atom(
                str(data["subject"]), str(data["predicate"]), str(data["object"]),
                bool(data.get("negated", False)), float(data.get("confidence", 1.0)),
                str(data.get("source", experience.source)),
            ))
        environment = str(experience.context.get("environment") or "unspecified")
        if environment != "unspecified":
            self.worldview.observe(facts, environment)
        self.subconscious_memory.reinforce(
            list(experience.concepts), reward=max(-1.0, min(1.0, experience.reward)),
            source=experience.source, learning_rate=0.04,
        )
        self.personal_symbols.observe(experience)
        if learn:
            self.relearn_from_experience_history()
        return experience

    def relearn_from_experience_history(self, *, limit: int = 2000) -> tuple[list, list]:
        experiences = self.memory.recent_experiences(limit)
        candidates = self.experience_rules.learn(experiences)
        variants = self.rule_evolution.evolve(candidates, experiences)
        self.rule_genealogy.sync(candidates, variants)
        self._refresh_adaptive_rules()
        self.active_testing.plan(candidates, variants, max_tests=self.active_testing.budget)
        return candidates, variants

    def simulate_life(
        self, *, total_experiences: int = 1000, max_age: int = 40,
        age_checkpoints: list[int] | None = None, seed: int = 7,
        profile: CognitiveProfile | None = None, learning_batch: int = 25,
    ) -> LifeSimulationReport:
        from xithink.modules.life_simulation import LifeExperienceSimulator
        simulator = LifeExperienceSimulator(self, profile or self.cognitive_profile)
        return simulator.run(
            total_experiences=total_experiences, max_age=max_age,
            age_checkpoints=age_checkpoints, seed=seed, learning_batch=learning_batch,
        )

    def add_fact(self, atom: Atom) -> bool:
        changed = self.reasoner.add_fact(atom)
        self.long_term.remember_fact(atom)
        self.concepts.get_or_create(atom.subject)
        self.concepts.get_or_create(atom.object)
        self.graph.ingest_fact(atom)
        self.embedding_memory.index_fact(atom)
        return changed

    def add_rule(self, rule: Rule) -> None:
        self.reasoner.add_rule(rule)
        self.long_term.remember_rule(rule)
        self.graph.ingest_rule(rule)
        for atom in (*rule.premises, rule.conclusion):
            self.concepts.get_or_create(atom.subject)
            self.concepts.get_or_create(atom.object)

    def _ingest_frame(self, frame: SemanticFrame) -> None:
        for name in frame.concepts:
            self.concepts.get_or_create(name)
        for event in frame.events:
            self.memory.add_event(event)
            self.concepts.get_or_create(event.actor, kind="entity")
            if event.object:
                self.concepts.get_or_create(event.object)
            self.graph.ingest_event(event)
            self.embedding_memory.index_event(event)

    def analyze(self, text: str) -> SemanticFrame:
        frame = self.semantic.parse(text)
        self._ingest_frame(frame)
        return frame

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        return list(dict.fromkeys(x for x in items if x))

    @staticmethod
    def _merge_activation(*maps: dict[str, float]) -> dict[str, float]:
        merged: dict[str, float] = {}
        for mapping in maps:
            for key, value in mapping.items():
                merged[key] = max(merged.get(key, 0.0), value)
        return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True))

    @staticmethod
    def _affect_valence(text: str) -> float:
        """Small deterministic salience cue, not an emotion classifier.

        It gives perspective resonance a rough signed trace of how the engine's
        own episode was framed without asking a language model to interpret it.
        """
        positive = ("帮助", "安全", "保护", "喜欢", "自由", "成功", "支持", "理解", "信任", "希望")
        negative = ("受伤", "疼痛", "危险", "风险", "伤害", "害怕", "恐惧", "失败", "失去", "限制", "反对")
        pos = sum(text.count(x) for x in positive)
        neg = sum(text.count(x) for x in negative)
        if pos + neg == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))

    def _experience_from_frame(self, frame: SemanticFrame, environment: str | None = None) -> Experience:
        return Experience(
            content=frame.raw_text,
            concepts=list(frame.concepts),
            kind="interaction",
            context={
                "facts": [a.render() for a in frame.facts],
                "fact_atoms": [
                    {
                        "subject": a.subject, "predicate": a.predicate, "object": a.object,
                        "negated": a.negated, "confidence": a.confidence, "source": a.source,
                    }
                    for a in frame.facts
                ],
                "rules": [r.name for r in frame.rules],
                "questions": list(frame.questions),
                "events": [e.render() for e in frame.events],
                "source": frame.source,
                "modality": frame.modality,
                "environment": environment or "unspecified",
                "affect_valence": self._affect_valence(frame.raw_text),
            },
            confidence=0.85,
            source=frame.source,
        )

    def _context_reasoner(self, environment: str, current_facts: list[Atom] | None = None) -> SymbolicReasoner:
        persisted = [f for f in self.long_term.facts() if not f.source.startswith("inference:")]
        return self.worldview.build_reasoner(
            base_facts=persisted,
            base_rules=self.reasoner.rules,
            learned_rules=[
                *self.experience_rules.verified_rules(environment),
                *self.rule_evolution.verified_rules(environment),
            ],
            environment=environment,
            current_facts=current_facts or [],
        )

    def answer_frame(
        self,
        frame: SemanticFrame,
        environment: str | None = None,
        reasoner: SymbolicReasoner | None = None,
    ) -> SymbolicAnswer:
        if not frame.question_intents:
            return SymbolicAnswer(False, "没有检测到可结构化回答的问题。")
        # The answer path remains symbolic-only. Environment-scoped experiential
        # beliefs/rules may join a temporary reasoner without becoming universal facts.
        active = reasoner or (self._context_reasoner(environment, frame.facts) if environment else self.reasoner)
        return active.answer(frame.question_intents[0])

    def consult_general_knowledge(self, query: str) -> list[KnowledgeClaim]:
        """Ask the external/general-knowledge provider without changing thought state."""
        return self.knowledge.consult(query)

    def verify_general_knowledge(self, claim_id: str) -> bool:
        """Explicitly promote a quarantined claim into long-term symbolic memory."""
        claim = self.knowledge.mark_verified(claim_id)
        if claim is None:
            return False
        self.add_fact(claim.as_atom())
        return True

    def reject_general_knowledge(self, claim_id: str) -> bool:
        return self.knowledge.reject(claim_id)

    def think(self, text: str, cycles: int = 3, environment: str | None = None) -> list[Thought]:
        if cycles < 1 or cycles > 20:
            raise ValueError("cycles must be between 1 and 20")

        thoughts: list[Thought] = []
        parent_ids: list[str] = []

        # Parse once. Retrieve old memory before writing the current episode.
        frame = self.semantic.parse(text)
        semantic_recalled = self.embedding_memory.query(
            text,
            concepts=frame.concepts,
            top_k=6,
            min_score=0.12,
            kinds={"fact", "event", "thought"},
        )
        experience_recalled = self.experience_memory.recall(
            text, concepts=frame.concepts, top_k=5, min_score=0.12
        )
        recalled = semantic_recalled + experience_recalled
        recalled.sort(key=lambda x: x.score, reverse=True)
        recalled = recalled[:8]

        subconscious_before = self.subconscious_memory.activate(frame.concepts)
        symbol_activation_before, active_personal_symbols = self.personal_symbols.activate(
            frame.concepts, environment=environment
        )
        resonance = self.perspective.assess(
            frame.concepts, experience_recalled,
            empathy_weight=self.state.values.get("empathy", 0.75),
        )
        self._ingest_frame(frame)

        # Without an explicit environment, input facts keep the 0.4 behavior and
        # enter universal long-term memory. With an environment, they are stored
        # as scoped experiential observations instead.
        if environment:
            self.worldview.observe(frame.facts, environment)
            active_reasoner = self._context_reasoner(environment, frame.facts)
        else:
            for fact in frame.facts:
                self.add_fact(fact)
            active_reasoner = self.reasoner
        for rule in frame.rules:
            self.add_rule(rule)

        episode = self._experience_from_frame(frame, environment)
        self.experience_memory.remember(episode)
        self.subconscious_memory.reinforce(frame.concepts, source="input")

        base_questions = list(frame.questions)
        carry_concepts = list(frame.concepts)
        trace_cycles: list[dict] = []
        best_score = 0.0
        best_statement: str | None = None

        for cycle in range(cycles):
            inferred = active_reasoner.infer()
            if not environment:
                for fact in inferred:
                    self.long_term.remember_fact(fact)
                    self.concepts.get_or_create(fact.subject)
                    self.concepts.get_or_create(fact.object)
                    self.graph.ingest_fact(fact)
                    self.embedding_memory.index_fact(fact)

            symbolic_activation = self.association.activate(carry_concepts, active_reasoner.facts)
            graph_activation = self.graph.activate(carry_concepts, max_depth=3)
            subconscious_activation = self.subconscious_memory.activate(carry_concepts)
            recall_activation: dict[str, float] = {}
            for match in recalled:
                for concept in match.concepts:
                    recall_activation[concept] = max(
                        recall_activation.get(concept, 0.0),
                        min(0.72, match.score * 0.68),
                    )
            symbol_activation, cycle_symbols = self.personal_symbols.activate(
                carry_concepts, environment=environment
            )
            activation = self._merge_activation(
                symbolic_activation, graph_activation, recall_activation, subconscious_activation,
                symbol_activation, resonance.activation,
            )

            internal_frame = SemanticFrame(
                raw_text=text if cycle == 0 else "<internal-thought>",
                concepts=carry_concepts,
                facts=[] if cycle else frame.facts,
                rules=[] if cycle else frame.rules,
                questions=base_questions if cycle == 0 else [],
                question_intents=frame.question_intents if cycle == 0 else [],
                source="internal" if cycle else frame.source,
                modality=frame.modality,
            )
            candidates = self.hypotheses.generate(
                internal_frame,
                active_reasoner,
                activation,
                self.memory,
                recalled=recalled,
                personal_symbols=cycle_symbols,
            )
            reviewed, assessments = self.metacognition.review(
                candidates, active_reasoner, self.graph, recalled
            )
            reviewed.extend(self.metacognition.counterexample_reflections(reviewed, assessments))
            selected = self.decision.select(reviewed, self.state)

            cycle_trace = {
                "cycle": cycle,
                "activation": list(activation.items())[:10],
                "subconscious_activation": list(subconscious_activation.items())[:10],
                "candidates": [
                    {
                        "statement": h.statement,
                        "evidence": h.evidence,
                        "consistency": h.consistency,
                        "uncertainty": h.uncertainty,
                        "memory_support": h.memory_support,
                        "counterevidence": h.counterevidence,
                        "tags": list(h.tags),
                        "metacognition": h.metadata.get("metacognition", {}),
                    }
                    for h in reviewed
                ],
            }

            if selected is None:
                thought = Thought(
                    cycle=cycle,
                    kind="halt",
                    content="当前没有新的可检验假设，思维循环暂停。",
                    confidence=1.0,
                    concepts=carry_concepts[:8],
                    parents=parent_ids,
                )
                self.memory.add_thought(thought)
                self.embedding_memory.index_thought(thought)
                thoughts.append(thought)
                self.self_model.update(self.state, None)
                self._save_state()
                cycle_trace["selected"] = None
                trace_cycles.append(cycle_trace)
                break

            thought_concepts: list[str] = []
            for atom in selected.hypothesis.supporting_atoms:
                thought_concepts.extend([atom.subject, atom.object])
            thought_concepts.extend(list(activation.keys())[:5])
            thought_concepts = self._dedupe(thought_concepts)

            meta = selected.hypothesis.metadata.get("metacognition", {})
            if meta.get("needs_verification"):
                verification = f"待验证：{selected.hypothesis.statement}"
                if verification not in self.state.unresolved_questions:
                    self.state.unresolved_questions.append(verification)
                    self.state.unresolved_questions = self.state.unresolved_questions[-50:]

            thought = Thought(
                cycle=cycle,
                kind="decision",
                content=selected.hypothesis.statement,
                confidence=selected.score,
                concepts=thought_concepts[:8],
                parents=parent_ids,
            )
            self.memory.add_thought(thought)
            self.embedding_memory.index_thought(thought)
            thoughts.append(thought)
            self.self_model.update(self.state, selected)
            self._save_state()

            self.subconscious_memory.reinforce(
                self._dedupe(frame.concepts + thought_concepts),
                reward=max(-1.0, min(1.0, selected.score * 2 - 1)),
                source="thought-outcome",
                learning_rate=0.06,
            )

            if selected.score > best_score:
                best_score = selected.score
                best_statement = selected.hypothesis.statement

            cycle_trace["selected"] = {
                "statement": selected.hypothesis.statement,
                "score": selected.score,
                "rationale": selected.rationale,
                "metacognition": meta,
            }
            trace_cycles.append(cycle_trace)

            carry_concepts = thought_concepts or carry_concepts
            parent_ids = [thought.id]

        episode.outcome = best_statement
        episode.reward = best_score
        self.experience_memory.remember(episode)
        self.personal_symbols.observe(episode)

        learned_candidates = self.experience_rules.learn()
        evolved_variants = self.rule_evolution.evolve(learned_candidates)
        self.rule_genealogy.sync(learned_candidates, evolved_variants)
        self.active_testing.plan(learned_candidates, evolved_variants, environment=environment)
        # Adaptive rules are rebuilt from evidence every turn. They participate
        # in reasoning, but are not copied into immutable long-term rule memory.
        self._refresh_adaptive_rules()
        if environment:
            active_reasoner = self._context_reasoner(environment, frame.facts)
        else:
            active_reasoner = self.reasoner

        symbolic_answer = self.answer_frame(
            frame, environment=environment, reasoner=active_reasoner
        ) if frame.question_intents else SymbolicAnswer(False, "")
        alternative_contexts: dict[str, SymbolicReasoner] = {}
        if frame.question_intents:
            intent = frame.question_intents[0]
            if intent.subject and intent.predicate and intent.object:
                for alt_env in self.worldview.relevant_environments(intent.subject, intent.predicate, intent.object)[:6]:
                    if alt_env == environment:
                        continue
                    alternative_contexts[alt_env] = self._context_reasoner(alt_env)
        perspective_report = self.multi_perspective.analyze(
            frame.question_intents[0] if frame.question_intents else None,
            universal_reasoner=self.reasoner,
            contextual_reasoner=active_reasoner,
            evolved_variants=evolved_variants,
            environment=environment,
            alternative_contexts=alternative_contexts,
        )
        self.last_trace = {
            "input": text,
            "environment": environment,
            "worldview": self.worldview.snapshot(environment),
            "experience_rule_learning": self.experience_rules.snapshot(),
            "rule_evolution": self.rule_evolution.snapshot(),
            "rule_genealogy": self.rule_genealogy.snapshot(),
            "active_hypothesis_testing": self.active_testing.snapshot(),
            "cognitive_profile": asdict(self.cognitive_profile),
            "multi_perspective": perspective_report.to_dict(),
            "personal_symbols": {
                **self.personal_symbols.snapshot(),
                "activation_before_input": list(symbol_activation_before.items()),
                "active_before_input": [s.name for s in active_personal_symbols],
            },
            "perspective_resonance": {
                "score": resonance.score,
                "shared_concepts": resonance.shared_concepts,
                "experience_ids": resonance.experience_ids,
                "valence": resonance.valence,
                "activation": resonance.activation,
                "note": resonance.note,
            },
            "memory_layers": {
                "long_term": self.long_term.snapshot(),
                "experience": {
                    "count": self.experience_memory.count(),
                    "recall": [
                        {"id": m.item_id, "text": m.text, "score": m.score, "concepts": m.concepts}
                        for m in experience_recalled
                    ],
                    "authority": "case-context-only",
                },
                "subconscious": {
                    **self.subconscious_memory.snapshot(frame.concepts),
                    "activation_before_input": list(subconscious_before.items()),
                },
            },
            "recall": [
                {
                    "id": m.item_id,
                    "kind": m.kind,
                    "text": m.text,
                    "score": m.score,
                    "concepts": m.concepts,
                }
                for m in recalled
            ],
            "symbolic_answer": {
                "answered": symbolic_answer.answered,
                "text": symbolic_answer.text,
                "confidence": symbolic_answer.confidence,
                "proofs": [p.to_dict() for p in symbolic_answer.proofs],
            },
            "external_knowledge": {
                "pending_claims": self.memory.knowledge_claim_count("quarantined"),
                "used_in_thought_loop": False,
            },
            "cycles": trace_cycles,
        }
        return thoughts

    def snapshot(self) -> dict:
        return {
            "state": {
                "identity": self.state.identity,
                "values": dict(self.state.values),
                "traits": dict(self.state.traits),
                "goals": list(self.state.active_goals),
                "unresolved_questions": list(self.state.unresolved_questions),
                "thought_count": self.state.thought_count,
            },
            "facts": [f.render() for f in self.reasoner.facts],
            "rules": len(self.reasoner.rules),
            "concepts": [
                {"id": c.id, "name": c.canonical_name, "kind": c.kind, "aliases": c.aliases}
                for c in self.concepts.all()
            ],
            "concept_graph": self.graph.snapshot(),
            "embedding_memory": {"items": self.embedding_memory.count()},
            "rule_evolution": self.rule_evolution.snapshot(),
            "rule_genealogy": self.rule_genealogy.snapshot(),
            "active_hypothesis_testing": self.active_testing.snapshot(),
            "cognitive_profile": asdict(self.cognitive_profile),
            "memory_layers": {
                "long_term": self.long_term.snapshot(),
                "experience": {"count": self.experience_memory.count(), "authority": "case-context-only"},
                "subconscious": self.subconscious_memory.snapshot(),
            },
            "experience_rule_learning": self.experience_rules.snapshot(),
            "personal_symbols": self.personal_symbols.snapshot(),
            "worldview": self.worldview.snapshot(),
            "external_knowledge": {
                "provider": getattr(self.knowledge.provider, "name", "unknown"),
                "pending": self.memory.knowledge_claim_count("quarantined"),
                "verified": self.memory.knowledge_claim_count("verified"),
            },
            "recent_events": self.memory.recent_events(10),
            "recent_memory": self.memory.recent(10),
            "last_trace": self.last_trace,
        }

    def close(self) -> None:
        self._save_state()
        self.memory.close()
