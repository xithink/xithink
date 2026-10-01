from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict

from xithink.models import ActiveHypothesisTest, EvolvedRuleVariant, LearnedRuleCandidate
from xithink.modules.memory import MemoryStore


class ActiveHypothesisTester:
    """Select observations that best discriminate competing symbolic theories.

    The planner never fabricates evidence. It proposes what should be observed.
    A real sensor/human can later resolve the test. The life simulator has a
    separate synthetic oracle, explicitly isolated from the production path.
    """

    def __init__(self, store: MemoryStore, *, aggressiveness: float = 0.70, budget: int = 3) -> None:
        self.store = store
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))
        self.budget = max(1, budget)

    @staticmethod
    def _id(base: str, a: str, b: str, conditions: list[str], environment: str | None) -> str:
        raw = "|".join([base, a, b, *sorted(conditions), environment or "*"])
        return "test:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:18]

    @staticmethod
    def _condition_set(v: EvolvedRuleVariant) -> set[str]:
        return {x.render() for x in v.conditions}

    @staticmethod
    def _balance_gain(a: EvolvedRuleVariant, b: EvolvedRuleVariant) -> float:
        # Maximum uncertainty is most useful to test. Balanced competing support
        # gets higher gain than a 99:1 split.
        x = max(1, a.support)
        y = max(1, b.support)
        p = x / (x + y)
        entropy = 0.0
        for q in (p, 1 - p):
            if q > 0:
                entropy -= q * math.log2(q)
        confidence_term = 1.0 - abs(a.confidence - b.confidence)
        return max(0.0, min(1.0, 0.7 * entropy + 0.3 * confidence_term))

    def plan(
        self,
        candidates: list[LearnedRuleCandidate] | None = None,
        variants: list[EvolvedRuleVariant] | None = None,
        *,
        environment: str | None = None,
        max_tests: int | None = None,
    ) -> list[ActiveHypothesisTest]:
        candidates = candidates if candidates is not None else self.store.load_learned_rule_candidates()
        variants = variants if variants is not None else self.store.load_evolved_rule_variants()
        existing = {t.id: t for t in self.store.load_active_hypothesis_tests()}
        by_base: dict[str, list[EvolvedRuleVariant]] = defaultdict(list)
        for v in variants:
            if v.status == "superseded":
                continue
            if environment and v.status.endswith("_scoped") and environment not in v.environments:
                continue
            by_base[v.base_signature].append(v)

        proposals: list[ActiveHypothesisTest] = []
        for base, group in by_base.items():
            positive = [v for v in group if not v.conclusion.negated]
            negative = [v for v in group if v.conclusion.negated]
            for a in positive:
                for b in negative:
                    a_conditions = self._condition_set(a)
                    b_conditions = self._condition_set(b)
                    discriminators = sorted(a_conditions ^ b_conditions)
                    if not discriminators:
                        discriminators = sorted(a_conditions | b_conditions)
                    if not discriminators:
                        continue
                    gain = self._balance_gain(a, b)
                    priority = max(0.0, min(1.0, gain * (0.55 + 0.45 * self.aggressiveness)))
                    if priority < 0.30:
                        continue
                    target = a.conclusion.render().replace("?x", "目标对象").replace("?z", "目标结果")
                    question = (
                        f"为了区分两套相反理论，优先观察：{'；'.join(discriminators[:4])}。"
                        f"随后检验『{target}』是否成立。"
                    )
                    tid = self._id(base, a.id, b.id, discriminators, environment)
                    old = existing.get(tid)
                    test = ActiveHypothesisTest(
                        id=tid, base_signature=base, hypothesis_a=a.id, hypothesis_b=b.id,
                        target=target, discriminating_conditions=discriminators,
                        question=question, expected_information_gain=gain, priority=priority,
                        environment=environment,
                        status=old.status if old else "proposed",
                        outcome=old.outcome if old else None,
                        resolved_experience_id=old.resolved_experience_id if old else None,
                        created_at=old.created_at if old else time.time(), updated_at=time.time(),
                    )
                    self.store.save_active_hypothesis_test(test)
                    proposals.append(test)

        # If evidence is mixed but condition splitting has not yet succeeded,
        # create a theory-search test rather than pretending the missing
        # condition is known.
        variant_bases = set(by_base)
        for c in candidates:
            if c.support < 1 or c.counterexamples < 1 or c.signature in variant_bases:
                continue
            uncertainty = 1.0 - abs(c.support - c.counterexamples) / max(1, c.support + c.counterexamples)
            priority = max(0.0, min(1.0, uncertainty * self.aggressiveness))
            if priority < 0.30:
                continue
            tid = self._id(c.signature, c.signature, "opposite", ["未知区分条件"], environment)
            old = existing.get(tid)
            test = ActiveHypothesisTest(
                id=tid, base_signature=c.signature,
                hypothesis_a=c.conclusion.render(), hypothesis_b=("非 " + c.conclusion.render()),
                target=c.conclusion.render(), discriminating_conditions=["未知区分条件"],
                question="正反案例同时存在，但尚未找到区分条件。应主动收集环境、主体状态和上下文条件。",
                expected_information_gain=uncertainty, priority=priority, environment=environment,
                status=old.status if old else "proposed", outcome=old.outcome if old else None,
                resolved_experience_id=old.resolved_experience_id if old else None,
                created_at=old.created_at if old else time.time(), updated_at=time.time(),
            )
            self.store.save_active_hypothesis_test(test)
            proposals.append(test)

        proposals.sort(key=lambda x: (x.status == "proposed", x.priority, x.expected_information_gain), reverse=True)
        return proposals[: max_tests or self.budget]

    def resolve(self, test_id: str, *, outcome: str, experience_id: str | None = None) -> ActiveHypothesisTest | None:
        tests = {t.id: t for t in self.store.load_active_hypothesis_tests()}
        test = tests.get(test_id)
        if test is None:
            return None
        test.status = "resolved" if outcome not in {"unknown", "inconclusive"} else "inconclusive"
        test.outcome = outcome
        test.resolved_experience_id = experience_id
        test.updated_at = time.time()
        self.store.save_active_hypothesis_test(test)
        return test

    def snapshot(self, limit: int = 20) -> dict:
        tests = self.store.load_active_hypothesis_tests()[:limit]
        return {
            "tests": [
                {
                    "id": t.id, "base_signature": t.base_signature,
                    "hypothesis_a": t.hypothesis_a, "hypothesis_b": t.hypothesis_b,
                    "target": t.target, "conditions": t.discriminating_conditions,
                    "question": t.question, "expected_information_gain": t.expected_information_gain,
                    "priority": t.priority, "environment": t.environment,
                    "status": t.status, "outcome": t.outcome,
                    "resolved_experience_id": t.resolved_experience_id,
                }
                for t in tests
            ],
            "proposed": sum(t.status == "proposed" for t in tests),
            "resolved": sum(t.status == "resolved" for t in tests),
            "authority": "observation-request-only-never-fabricates-evidence",
        }
