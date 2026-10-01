from __future__ import annotations

import random
from dataclasses import asdict
from typing import TYPE_CHECKING

from xithink.models import (
    ActiveHypothesisTest, AgeCognitionSnapshot, Atom, CognitiveProfile,
    Experience, LifeSimulationReport,
)
from xithink.modules.symbolic import SymbolicReasoner

if TYPE_CHECKING:
    from xithink.engine import XThinkEngine


BUILTIN_PROFILES: dict[str, CognitiveProfile] = {
    "balanced": CognitiveProfile(),
    "conservative": CognitiveProfile(
        name="conservative", skepticism=0.86, exploration=0.42,
        rule_min_support=5, rule_verification_confidence=0.82,
        reject_counterexamples=1, condition_min_support=3,
        condition_verification_confidence=0.80, condition_max_counterexample_rate=0.07,
        active_test_aggressiveness=0.52, active_test_budget=2,
        environment_diversity=0.58, experience_noise=0.04,
    ),
    "exploratory": CognitiveProfile(
        name="exploratory", skepticism=0.48, exploration=0.94,
        rule_min_support=2, rule_verification_confidence=0.67,
        reject_counterexamples=3, condition_min_support=2,
        condition_verification_confidence=0.66, condition_max_counterexample_rate=0.18,
        active_test_aggressiveness=0.94, active_test_budget=5,
        environment_diversity=0.90, experience_noise=0.09,
    ),
}


class SyntheticLifeWorld:
    """Deterministic symbolic environment used only by the 0.7 simulator."""

    ENVIRONMENTS = ["家庭", "学校", "城市", "工作", "研究"]
    PATTERNS = [
        ([('是', '安全支持')], False),
        ([('是', '高风险')], True),
        ([('有', '合作')], False),
        ([('处于', '孤立')], True),
        ([('处于', '高压力'), ('有', '充分休息')], False),
        ([('处于', '高压力'), ('有', '睡眠不足')], True),
    ]

    def __init__(self, *, seed: int = 7, noise: float = 0.06, environment_diversity: float = 0.70) -> None:
        self.rng = random.Random(seed)
        self.noise = max(0.0, min(0.35, noise))
        self.environment_diversity = max(0.0, min(1.0, environment_diversity))
        self.counter = 0

    def _available_envs(self) -> list[str]:
        count = max(1, round(1 + self.environment_diversity * (len(self.ENVIRONMENTS) - 1)))
        return self.ENVIRONMENTS[:count]

    @staticmethod
    def _atom_dict(a: Atom) -> dict:
        return {
            "subject": a.subject, "predicate": a.predicate, "object": a.object,
            "negated": a.negated, "confidence": a.confidence, "source": a.source,
        }

    def _episode(self, conditions: list[tuple[str, str]], negated: bool, *, environment: str | None = None,
                 source: str = "synthetic-passive") -> Experience:
        self.counter += 1
        i = self.counter
        subject, middle, result = f"主体{i:04d}", f"中介{i:04d}", f"结果{i:04d}"
        actual_negated = negated
        if self.rng.random() < self.noise:
            actual_negated = not actual_negated
        atoms = [
            Atom(subject, "导致", middle, source=source),
            Atom(middle, "导致", result, source=source),
            *[Atom(subject, predicate, obj, source=source) for predicate, obj in conditions],
            Atom(subject, "间接导致", result, negated=actual_negated, source=source),
        ]
        env = environment or self.rng.choice(self._available_envs())
        concepts = [subject, middle, result] + [obj for _, obj in conditions]
        text = "。".join(a.render().replace(" ", "") for a in atoms) + "。"
        return Experience(
            content=text,
            concepts=concepts,
            kind="synthetic-life",
            context={
                "facts": [a.render() for a in atoms],
                "fact_atoms": [self._atom_dict(a) for a in atoms],
                "rules": [], "questions": [], "events": [],
                "source": source, "modality": "asserted", "environment": env,
                "affect_valence": 0.15 if not actual_negated else -0.15,
                "synthetic_truth": {"conditions": conditions, "negated": negated},
            },
            confidence=0.92,
            source=source,
        )

    def passive_experience(self) -> Experience:
        conditions, negated = self.rng.choice(self.PATTERNS)
        return self._episode(list(conditions), negated)

    def targeted_experience(self, test: ActiveHypothesisTest) -> tuple[Experience, str]:
        condition_objects: list[str] = []
        for rendered in test.discriminating_conditions:
            parts = rendered.split()
            if len(parts) >= 3 and not parts[-1].startswith("?"):
                condition_objects.append(parts[-1])
        matching = [p for p in self.PATTERNS if any(obj in {x[1] for x in p[0]} for obj in condition_objects)]
        conditions, negated = self.rng.choice(matching or self.PATTERNS)
        exp = self._episode(list(conditions), negated, environment=test.environment, source="synthetic-active-test")
        return exp, ("opposite-supported" if negated else "positive-supported")

    def benchmark_cases(self) -> list[tuple[list[Atom], bool]]:
        cases: list[tuple[list[Atom], bool]] = []
        for idx, (conditions, negated) in enumerate(self.PATTERNS):
            s, m, z = f"测试主体{idx}", f"测试中介{idx}", f"测试结果{idx}"
            premises = [Atom(s, "导致", m), Atom(m, "导致", z)]
            premises.extend(Atom(s, pred, obj) for pred, obj in conditions)
            cases.append((premises, negated))
        return cases


class LifeExperienceSimulator:
    def __init__(self, engine: "XThinkEngine", profile: CognitiveProfile | None = None) -> None:
        self.engine = engine
        self.profile = profile or CognitiveProfile()
        self.engine.apply_cognitive_profile(self.profile)

    def _evaluate(self, world: SyntheticLifeWorld) -> tuple[float, float, float]:
        rules = [
            *self.engine.experience_rules.verified_rules(),
            *self.engine.rule_evolution.verified_rules(),
        ]
        correct = 0
        contradictions = 0
        total = 0
        for premises, expected_negated in world.benchmark_cases():
            r = SymbolicReasoner()
            for a in premises:
                r.add_fact(a)
            for rule in rules:
                r.add_rule(rule)
            r.infer()
            s = premises[0].subject
            z = premises[1].object
            expected = (s, "间接导致", z, expected_negated)
            opposite = (s, "间接导致", z, not expected_negated)
            keys = {a.key() for a in r.facts}
            total += 1
            if expected in keys and opposite not in keys:
                correct += 1
            if expected in keys and opposite in keys:
                contradictions += 1

        # Unknown-condition benchmark: a mature conditional theory should abstain.
        abstain_total = 8
        abstain_correct = 0
        for i in range(abstain_total):
            r = SymbolicReasoner()
            s, m, z = f"未知主体{i}", f"未知中介{i}", f"未知结果{i}"
            r.add_fact(Atom(s, "导致", m)); r.add_fact(Atom(m, "导致", z))
            for rule in rules:
                r.add_rule(rule)
            r.infer()
            keys = {a.key() for a in r.facts}
            if (s, "间接导致", z, False) not in keys and (s, "间接导致", z, True) not in keys:
                abstain_correct += 1
        return (
            correct / max(1, total),
            abstain_correct / abstain_total,
            contradictions / max(1, total),
        )

    def _snapshot(self, age: int, experiences: int, world: SyntheticLifeWorld) -> AgeCognitionSnapshot:
        candidates = self.engine.memory.load_learned_rule_candidates()
        variants = self.engine.memory.load_evolved_rule_variants()
        genealogy = self.engine.rule_genealogy.snapshot(limit=5000)
        tests = self.engine.memory.load_active_hypothesis_tests()
        accuracy, abstention, contradiction = self._evaluate(world)
        return AgeCognitionSnapshot(
            age=age,
            experiences=experiences,
            learned_candidates=len(candidates),
            verified_rules=sum(c.status.startswith("verified") for c in candidates),
            evolved_variants=len(variants),
            active_evolved_rules=sum(v.status.startswith("verified") for v in variants),
            genealogy_nodes=len(genealogy["nodes"]),
            genealogy_max_generation=genealogy["max_generation"],
            proposed_tests=sum(t.status == "proposed" for t in tests),
            resolved_tests=sum(t.status == "resolved" for t in tests),
            symbolic_accuracy=accuracy,
            abstention_accuracy=abstention,
            contradiction_rate=contradiction,
            worldview_beliefs=len(self.engine.worldview.snapshot().get("beliefs", [])),
            personal_symbols=sum(s.exposures >= 2 for s in self.engine.personal_symbols.all()),
            notes=[
                "零经历阶段只测先验规则，不注入合成经验。" if experiences == 0 else
                "同一留出题集用于不同年龄检查点，便于比较经验增长造成的变化。"
            ],
        )

    def run(
        self,
        *,
        total_experiences: int = 1000,
        max_age: int = 40,
        age_checkpoints: list[int] | None = None,
        seed: int = 7,
        learning_batch: int = 25,
    ) -> LifeSimulationReport:
        total_experiences = max(1, int(total_experiences))
        max_age = max(1, int(max_age))
        checkpoints = sorted(set(age_checkpoints or [0, 10, 20, 30, max_age]))
        checkpoints = [x for x in checkpoints if 0 <= x <= max_age]
        if 0 not in checkpoints:
            checkpoints.insert(0, 0)
        if max_age not in checkpoints:
            checkpoints.append(max_age)
        targets = {age: round(total_experiences * age / max_age) for age in checkpoints}

        world = SyntheticLifeWorld(
            seed=seed,
            noise=self.profile.experience_noise,
            environment_diversity=self.profile.environment_diversity,
        )
        snapshots: list[AgeCognitionSnapshot] = []
        ingested = 0
        if targets.get(0) == 0:
            snapshots.append(self._snapshot(0, 0, world))

        rng = random.Random(seed + 991)
        for age in checkpoints:
            target = targets[age]
            if age == 0:
                continue
            while ingested < target:
                use_active = False
                selected_test: ActiveHypothesisTest | None = None
                if ingested > 0 and ingested % learning_batch == 0:
                    self.engine.relearn_from_experience_history(limit=max(total_experiences + 100, 1200))
                    proposals = self.engine.active_testing.plan(max_tests=self.profile.active_test_budget)
                    if proposals and rng.random() < self.profile.active_test_aggressiveness:
                        selected_test = next((x for x in proposals if x.status == "proposed"), None)
                        use_active = selected_test is not None
                if use_active and selected_test is not None:
                    exp, outcome = world.targeted_experience(selected_test)
                else:
                    exp = world.passive_experience()
                    outcome = "passive"
                self.engine.ingest_structured_experience(exp, learn=False)
                ingested += 1
                if selected_test is not None:
                    self.engine.active_testing.resolve(selected_test.id, outcome=outcome, experience_id=exp.id)

            self.engine.relearn_from_experience_history(limit=max(total_experiences + 100, 1200))
            self.engine.active_testing.plan(max_tests=self.profile.active_test_budget)
            snapshots.append(self._snapshot(age, ingested, world))

        return LifeSimulationReport(
            profile=self.profile,
            total_experiences=total_experiences,
            max_age=max_age,
            seed=seed,
            checkpoints=snapshots,
            final_genealogy=self.engine.rule_genealogy.snapshot(limit=5000),
            final_active_tests=self.engine.active_testing.snapshot(limit=50)["tests"],
        )


def profile_from_name(name: str) -> CognitiveProfile:
    profile = BUILTIN_PROFILES.get((name or "balanced").strip().lower(), BUILTIN_PROFILES["balanced"])
    return CognitiveProfile(**asdict(profile))
