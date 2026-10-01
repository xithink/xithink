from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from itertools import combinations
from dataclasses import replace

from xithink.models import Atom, EvolvedRuleVariant, Experience, LearnedRuleCandidate, Rule
from xithink.modules.memory import MemoryStore


class RuleEvolutionEngine:
    """Refine experiential rules when positive and negative cases diverge.

    0.6 treats a counterexample as information, not merely a penalty. For a
    two-premise learned rule it searches the supporting and opposing episodes
    for *extra symbolic conditions* that distinguish the two groups. Those
    conditions can then create narrower rules, for example::

        (?x 导致 ?y) & (?y 导致 ?z) & (?x 是 条件甲) -> (?x 间接导致 ?z)
        (?x 导致 ?y) & (?y 导致 ?z) & (?x 是 条件乙) -> (?x 不间接导致 ?z)

    Missing conclusions are still treated as unknown (open-world), not false.
    Only explicit opposite conclusions form the opposing group.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        min_condition_support: int = 2,
        verification_confidence: float = 0.72,
        max_condition_terms: int = 2,
        max_counterexample_rate: float = 0.12,
    ) -> None:
        self.store = store
        self.min_condition_support = max(2, min_condition_support)
        self.verification_confidence = verification_confidence
        self.max_condition_terms = max(1, min(4, int(max_condition_terms)))
        self.max_counterexample_rate = max(0.0, min(0.35, float(max_counterexample_rate)))

    @staticmethod
    def _atom(data: dict) -> Atom:
        return Atom(
            str(data["subject"]), str(data["predicate"]), str(data["object"]),
            bool(data.get("negated", False)), float(data.get("confidence", 1.0)),
            str(data.get("source", "experience")),
        )

    def _facts(self, experience: Experience) -> list[Atom]:
        raw = experience.context.get("fact_atoms", [])
        return [
            self._atom(x) for x in raw
            if isinstance(x, dict) and {"subject", "predicate", "object"} <= set(x)
        ]

    @staticmethod
    def _matches_premise(atom: Atom, pattern: Atom) -> bool:
        return atom.predicate == pattern.predicate and atom.negated == pattern.negated

    @staticmethod
    def _replace_bound(value: str, bindings: dict[str, str]) -> str:
        for var, concrete in bindings.items():
            if value == concrete:
                return var
        return value

    def _normalize_extra(self, atom: Atom, bindings: dict[str, str]) -> Atom | None:
        subject = self._replace_bound(atom.subject, bindings)
        obj = self._replace_bound(atom.object, bindings)
        # A useful condition must touch at least one variable from the base rule;
        # unrelated co-occurrence must not silently become a causal condition.
        if not (subject.startswith("?") or obj.startswith("?")):
            return None
        return Atom(
            subject, atom.predicate, obj, atom.negated,
            max(0.5, atom.confidence), "experience-condition",
        )

    def _episode_instances(
        self, candidate: LearnedRuleCandidate, experience: Experience
    ) -> list[tuple[str, dict[str, str], list[Atom]]]:
        if len(candidate.premises) != 2:
            return []
        p1, p2 = candidate.premises
        conclusion = candidate.conclusion
        facts = self._facts(experience)
        keys = {f.key() for f in facts}
        out: list[tuple[str, dict[str, str], list[Atom]]] = []

        for first in facts:
            if not self._matches_premise(first, p1):
                continue
            for second in facts:
                if first is second or not self._matches_premise(second, p2):
                    continue
                if first.object != second.subject:
                    continue
                bindings = {"?x": first.subject, "?y": first.object, "?z": second.object}
                target = (first.subject, conclusion.predicate, second.object, conclusion.negated)
                opposite = (first.subject, conclusion.predicate, second.object, not conclusion.negated)
                if target in keys:
                    stance = "support"
                elif opposite in keys:
                    stance = "oppose"
                else:
                    continue

                excluded = {first.key(), second.key(), target, opposite}
                extras: list[Atom] = []
                seen: set[tuple[str, str, str, bool]] = set()
                for fact in facts:
                    if fact.key() in excluded:
                        continue
                    normalized = self._normalize_extra(fact, bindings)
                    if normalized is None or normalized.key() in seen:
                        continue
                    seen.add(normalized.key())
                    extras.append(normalized)
                out.append((stance, bindings, extras))
        return out

    @staticmethod
    def _feature_key(atom: Atom) -> str:
        return f"{atom.subject}|{atom.predicate}|{atom.object}|{int(atom.negated)}"

    @staticmethod
    def _variant_id(base_signature: str, conclusion: Atom, conditions: list[Atom]) -> str:
        condition_parts = [
            f"{a.subject}:{a.predicate}:{a.object}:{int(a.negated)}" for a in conditions
        ]
        raw = "|".join([
            base_signature, str(int(conclusion.negated)), conclusion.predicate,
            *sorted(condition_parts),
        ])
        return "evo-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _build_variants_for_candidate(
        self, candidate: LearnedRuleCandidate, experiences: list[Experience]
    ) -> list[EvolvedRuleVariant]:
        groups: dict[str, list[tuple[Experience, list[Atom]]]] = {"support": [], "oppose": []}
        for exp in experiences:
            instances = self._episode_instances(candidate, exp)
            # One episode counts once per stance/feature to avoid repetition bias.
            by_stance: dict[str, dict[str, Atom]] = {"support": {}, "oppose": {}}
            for stance, _, extras in instances:
                for atom in extras:
                    by_stance[stance][self._feature_key(atom)] = atom
            for stance in ("support", "oppose"):
                if by_stance[stance]:
                    groups[stance].append((exp, list(by_stance[stance].values())))

        support_n = len(groups["support"])
        oppose_n = len(groups["oppose"])
        if support_n == 0 or oppose_n == 0:
            return []

        feature_atoms: dict[str, Atom] = {}
        episode_features: dict[str, list[tuple[Experience, set[str]]]] = {
            "support": [], "oppose": []
        }
        for stance in ("support", "oppose"):
            for exp, extras in groups[stance]:
                keys: set[str] = set()
                for atom in extras:
                    key = self._feature_key(atom)
                    feature_atoms[key] = atom
                    keys.add(key)
                if keys:
                    episode_features[stance].append((exp, keys))

        def count_combo(stance: str, combo: tuple[str, ...]) -> tuple[int, set[str], set[str]]:
            matched_ids: set[str] = set()
            matched_envs: set[str] = set()
            wanted = set(combo)
            for exp, keys in episode_features[stance]:
                if wanted <= keys:
                    matched_ids.add(exp.id)
                    matched_envs.add(str(exp.context.get("environment") or "unspecified"))
            return len(matched_ids), matched_ids, matched_envs

        variants: list[EvolvedRuleVariant] = []
        all_feature_keys = sorted(feature_atoms)
        # Search single conditions first and then two-condition conjunctions.
        # Bounded size keeps rule evolution inspectable and avoids combinatorial
        # explosion while still allowing interaction effects to be discovered.
        condition_combos: list[tuple[str, ...]] = []
        for size in range(1, min(self.max_condition_terms, len(all_feature_keys)) + 1):
            condition_combos.extend(combinations(all_feature_keys, size))

        for stance, other in (("support", "oppose"), ("oppose", "support")):
            group_n = support_n if stance == "support" else oppose_n
            other_n = oppose_n if stance == "support" else support_n
            ranked: list[tuple[float, int, tuple[str, ...], set[str], set[str]]] = []
            for combo in condition_combos:
                count, matched_ids, matched_envs = count_combo(stance, combo)
                if count < self.min_condition_support:
                    continue
                other_count, _, _ = count_combo(other, combo)
                coverage = count / max(1, group_n)
                leakage = other_count / max(1, other_n)
                # 0.7: a condition may define a valid minority branch rather than
                # explain the whole positive/negative population. Purity therefore
                # carries more weight than global coverage, while min_condition_support
                # still prevents one-off anecdotes from becoming rules.
                discrimination = 0.65 * (1.0 - leakage) + 0.35 * coverage
                if discrimination >= 0.55:
                    # Prefer higher discrimination, then fewer conditions.
                    ranked.append((discrimination, -len(combo), combo, matched_ids, matched_envs))
            ranked.sort(reverse=True)

            emitted: list[set[str]] = []
            for discrimination, _, combo, matched_ids, matched_envs in ranked:
                combo_set = set(combo)
                # If a simpler already-emitted condition is a subset with equal
                # explanatory power, the larger conjunction adds no information.
                if any(existing <= combo_set for existing in emitted):
                    continue
                emitted.append(combo_set)
                conditions = [feature_atoms[k] for k in combo]
                conclusion = candidate.conclusion
                if stance == "oppose":
                    conclusion = replace(conclusion, negated=not conclusion.negated)
                other_count, counter_ids, _ = count_combo(other, combo)
                support = len(matched_ids)
                counters = other_count
                confidence = (support + 1.0) / (support + counters + 2.0)
                supporting_envs = sorted(matched_envs)
                non_default = {x for x in supporting_envs if x != "unspecified"}
                counter_rate = counters / max(1, support + counters)
                if (
                    support >= self.min_condition_support
                    and confidence >= self.verification_confidence
                    and counter_rate <= self.max_counterexample_rate
                ):
                    status = "verified_conditional_global" if len(non_default) >= 2 else "verified_conditional_scoped"
                elif support >= self.min_condition_support:
                    status = "provisional_conditional"
                else:
                    status = "candidate"

                condition_text = " AND ".join(c.render() for c in conditions)
                variant = EvolvedRuleVariant(
                    id=self._variant_id(candidate.signature, conclusion, conditions),
                    base_signature=candidate.signature,
                    premises=list(candidate.premises),
                    conclusion=replace(conclusion, source="experience-evolution"),
                    conditions=conditions,
                    support=support,
                    counterexamples=counters,
                    confidence=confidence,
                    status=status,
                    environments=supporting_envs,
                    supporting_experience_ids=sorted(matched_ids),
                    opposing_experience_ids=sorted(counter_ids),
                    rationale=(
                        f"正反案例分离条件：{condition_text}；"
                        f"本组覆盖 {support}/{group_n}，对侧命中 {counters}/{other_n}，"
                        f"区分度 {discrimination:.2f}，例外率 {counter_rate:.2%}。"
                    ),
                    updated_at=time.time(),
                )
                variants.append(variant)
                if len(emitted) >= 3:
                    break
        return variants

    def evolve(
        self,
        candidates: list[LearnedRuleCandidate] | None = None,
        experiences: list[Experience] | None = None,
    ) -> list[EvolvedRuleVariant]:
        candidates = candidates if candidates is not None else self.store.load_learned_rule_candidates()
        experiences = experiences if experiences is not None else self.store.recent_experiences(800)
        existing: dict[str, EvolvedRuleVariant] = {
            v.id: v for v in self.store.load_evolved_rule_variants()
        }
        created: dict[str, EvolvedRuleVariant] = dict(existing)

        # Complementary positive/negative candidates carry the same evidence in
        # opposite orientation. Prefer the positive orientation when both exist
        # so we do not create duplicate conditional variants.
        canonical_keys = {
            (
                tuple((p.subject, p.predicate, p.object, p.negated) for p in c.premises),
                c.conclusion.predicate,
                c.conclusion.subject,
                c.conclusion.object,
            )
            for c in candidates if not c.conclusion.negated
        }
        processed_bases: set[str] = set()
        fresh_ids_by_base: dict[str, set[str]] = defaultdict(set)
        experience_by_id = {e.id: e for e in experiences}

        for candidate in candidates:
            candidate_key = (
                tuple((p.subject, p.predicate, p.object, p.negated) for p in candidate.premises),
                candidate.conclusion.predicate, candidate.conclusion.subject, candidate.conclusion.object,
            )
            if candidate.conclusion.negated and candidate_key in canonical_keys:
                continue
            processed_bases.add(candidate.signature)
            # Evolution only makes sense when the base pattern has seen both
            # explicit positive and negative outcomes.
            if candidate.support < 1 or candidate.counterexamples < 1:
                continue
            relevant_ids = set(candidate.evidence_experience_ids) | set(candidate.counterexample_experience_ids)
            relevant_experiences = [experience_by_id[i] for i in relevant_ids if i in experience_by_id]
            if not relevant_experiences:
                relevant_experiences = experiences
            for variant in self._build_variants_for_candidate(candidate, relevant_experiences):
                fresh_ids_by_base[candidate.signature].add(variant.id)
                created[variant.id] = variant
                self.store.save_evolved_rule_variant(variant)

        # A condition may look perfect after the first counterexample and later
        # fail when more evidence arrives. Keep it for audit, but remove its
        # authority instead of letting an obsolete conditional rule survive.
        for variant in existing.values():
            if variant.base_signature not in processed_bases:
                continue
            if variant.id in fresh_ids_by_base.get(variant.base_signature, set()):
                continue
            variant.status = "superseded"
            variant.confidence = min(variant.confidence, 0.49)
            variant.rationale = (variant.rationale + " 条件在新增正反案例后不再具有足够区分度，已撤销推理权限。").strip()
            variant.updated_at = time.time()
            created[variant.id] = variant
            self.store.save_evolved_rule_variant(variant)

        return sorted(
            created.values(),
            key=lambda v: (v.status.startswith("verified"), v.confidence, v.support),
            reverse=True,
        )

    def verified_rules(self, environment: str | None = None) -> list[Rule]:
        rules: list[Rule] = []
        for variant in self.store.load_evolved_rule_variants():
            if variant.status == "verified_conditional_global":
                rules.append(variant.as_rule())
            elif (
                variant.status == "verified_conditional_scoped"
                and environment
                and environment in variant.environments
            ):
                rules.append(variant.as_rule())
        return rules

    def relevant_variants(
        self,
        *,
        predicate: str | None = None,
        negated: bool | None = None,
        environment: str | None = None,
    ) -> list[EvolvedRuleVariant]:
        out = []
        for variant in self.store.load_evolved_rule_variants():
            if predicate is not None and variant.conclusion.predicate != predicate:
                continue
            if negated is not None and variant.conclusion.negated != negated:
                continue
            if variant.status.endswith("_scoped") and environment and environment not in variant.environments:
                continue
            out.append(variant)
        return out

    def snapshot(self, limit: int = 30) -> dict:
        variants = self.store.load_evolved_rule_variants()[:limit]
        return {
            "variants": [
                {
                    "id": v.id,
                    "base_signature": v.base_signature,
                    "status": v.status,
                    "support": v.support,
                    "counterexamples": v.counterexamples,
                    "confidence": v.confidence,
                    "environments": v.environments,
                    "premises": [a.render() for a in v.premises],
                    "conditions": [a.render() for a in v.conditions],
                    "conclusion": v.conclusion.render(),
                    "rationale": v.rationale,
                }
                for v in variants
            ],
            "principle": "counterexamples refine conditions instead of only lowering confidence",
        }
