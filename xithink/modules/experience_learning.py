from __future__ import annotations

import hashlib
import time
from itertools import permutations

from xithink.models import Atom, Experience, LearnedRuleCandidate, Rule
from xithink.modules.memory import MemoryStore


class ExperienceRuleLearner:
    """Discover symbolic relation-composition rules from repeated episodes.

    The learner deliberately does not treat co-occurrence as causation. 0.5 only
    learns an inspectable triangle pattern:

        X --p1--> Y, Y --p2--> Z, X --p3--> Z

    Repeated episodes with the same predicate pattern support the generalized
    rule `(X p1 Y) & (Y p2 Z) -> (X p3 Z)`. Explicit opposite conclusions count
    as counterexamples; missing conclusions do not (open-world assumption).
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        min_support: int = 3,
        verification_confidence: float = 0.74,
        reject_counterexamples: int = 2,
    ) -> None:
        self.store = store
        self.min_support = max(2, min_support)
        self.verification_confidence = verification_confidence
        self.reject_counterexamples = max(1, reject_counterexamples)

    @staticmethod
    def _atom_from_context(data: dict) -> Atom:
        return Atom(
            str(data["subject"]), str(data["predicate"]), str(data["object"]),
            bool(data.get("negated", False)), float(data.get("confidence", 1.0)),
            str(data.get("source", "experience")),
        )

    def _facts(self, experience: Experience) -> list[Atom]:
        raw = experience.context.get("fact_atoms", [])
        return [self._atom_from_context(x) for x in raw if isinstance(x, dict) and {"subject", "predicate", "object"} <= set(x)]

    @staticmethod
    def _signature(p1: Atom, p2: Atom, conclusion: Atom) -> str:
        plain = "|".join([
            f"{int(p1.negated)}:{p1.predicate}",
            f"{int(p2.negated)}:{p2.predicate}",
            f"{int(conclusion.negated)}:{conclusion.predicate}",
        ])
        digest = hashlib.sha1(plain.encode("utf-8")).hexdigest()[:12]
        return f"tri-{digest}"

    def _patterns_in_episode(self, experience: Experience) -> dict[str, LearnedRuleCandidate]:
        facts = self._facts(experience)
        out: dict[str, LearnedRuleCandidate] = {}
        for first, second in permutations(facts, 2):
            if first is second or first.object != second.subject:
                continue
            for conclusion in facts:
                if conclusion is first or conclusion is second:
                    continue
                if conclusion.subject != first.subject or conclusion.object != second.object:
                    continue
                sig = self._signature(first, second, conclusion)
                premises = [
                    Atom("?x", first.predicate, "?y", first.negated, 1.0, "experience-pattern"),
                    Atom("?y", second.predicate, "?z", second.negated, 1.0, "experience-pattern"),
                ]
                generalized = Atom(
                    "?x", conclusion.predicate, "?z", conclusion.negated, 1.0, "experience-pattern"
                )
                out[sig] = LearnedRuleCandidate(sig, premises, generalized)
        return out

    @staticmethod
    def _episode_evidence(candidate: LearnedRuleCandidate, experience: Experience) -> tuple[bool, bool]:
        raw = experience.context.get("fact_atoms", [])
        facts = [ExperienceRuleLearner._atom_from_context(x) for x in raw if isinstance(x, dict) and {"subject", "predicate", "object"} <= set(x)]
        if len(candidate.premises) != 2:
            return False, False
        p1, p2 = candidate.premises
        conclusion = candidate.conclusion
        support = False
        counter = False
        for first in facts:
            if first.predicate != p1.predicate or first.negated != p1.negated:
                continue
            for second in facts:
                if second.predicate != p2.predicate or second.negated != p2.negated:
                    continue
                if first.object != second.subject:
                    continue
                target = (first.subject, conclusion.predicate, second.object, conclusion.negated)
                opposite = (first.subject, conclusion.predicate, second.object, not conclusion.negated)
                keys = {f.key() for f in facts}
                support = support or target in keys
                counter = counter or opposite in keys
        return support, counter

    def learn(self, experiences: list[Experience] | None = None) -> list[LearnedRuleCandidate]:
        experiences = experiences if experiences is not None else self.store.recent_experiences(500)
        templates: dict[str, LearnedRuleCandidate] = {
            c.signature: c for c in self.store.load_learned_rule_candidates()
        }
        for experience in experiences:
            templates.update({k: v for k, v in self._patterns_in_episode(experience).items() if k not in templates})

        updated: list[LearnedRuleCandidate] = []
        for candidate in templates.values():
            support_ids: list[str] = []
            counter_ids: list[str] = []
            environments: list[str] = []
            for experience in experiences:
                support, counter = self._episode_evidence(candidate, experience)
                if support:
                    support_ids.append(experience.id)
                    env = str(experience.context.get("environment") or "unspecified")
                    if env not in environments:
                        environments.append(env)
                if counter:
                    counter_ids.append(experience.id)

            support_n = len(set(support_ids))
            counter_n = len(set(counter_ids))
            confidence = (support_n + 1.0) / (support_n + counter_n + 2.0)
            non_default_envs = {x for x in environments if x != "unspecified"}
            if counter_n >= self.reject_counterexamples and counter_n >= support_n:
                status = "rejected"
            elif support_n >= self.min_support and counter_n == 0 and confidence >= self.verification_confidence:
                if len(non_default_envs) == 1:
                    status = "verified_scoped"
                else:
                    status = "verified_global"
            elif support_n >= 2:
                status = "provisional"
            else:
                status = "candidate"

            candidate.support = support_n
            candidate.counterexamples = counter_n
            candidate.confidence = confidence
            candidate.status = status
            candidate.environments = sorted(environments)
            candidate.evidence_experience_ids = sorted(set(support_ids))
            candidate.counterexample_experience_ids = sorted(set(counter_ids))
            candidate.updated_at = time.time()
            self.store.save_learned_rule_candidate(candidate)
            updated.append(candidate)
        return sorted(updated, key=lambda c: (c.status.startswith("verified"), c.confidence, c.support), reverse=True)

    def verified_rules(self, environment: str | None = None) -> list[Rule]:
        rules: list[Rule] = []
        for candidate in self.store.load_learned_rule_candidates():
            if candidate.status == "verified_global":
                rules.append(candidate.as_rule())
            elif candidate.status == "verified_scoped" and environment and environment in candidate.environments:
                rules.append(candidate.as_rule())
        return rules

    def snapshot(self, limit: int = 20) -> dict:
        candidates = self.store.load_learned_rule_candidates()[:limit]
        return {
            "candidates": [
                {
                    "signature": c.signature,
                    "status": c.status,
                    "support": c.support,
                    "counterexamples": c.counterexamples,
                    "confidence": c.confidence,
                    "environments": c.environments,
                    "premises": [a.render() for a in c.premises],
                    "conclusion": c.conclusion.render(),
                }
                for c in candidates
            ],
            "authority": "verified-rules-only",
        }
