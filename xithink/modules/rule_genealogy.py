from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict

from xithink.models import EvolvedRuleVariant, LearnedRuleCandidate, RuleGenealogyNode
from xithink.modules.memory import MemoryStore


class RuleGenealogyTracker:
    """Persist an auditable lineage of learned and evolved rules.

    A genealogy node is an immutable snapshot. When evidence changes the status,
    confidence or conditions of a rule, a new node is appended and linked to the
    previous snapshot. New conditional variants also link back to the base
    experiential rule and, when possible, the latest sibling they refine.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    @staticmethod
    def _node_id(rule_ref: str, status: str, support: int, counters: int, confidence: float,
                 conditions: list[str], parents: list[str]) -> str:
        payload = json.dumps({
            "rule_ref": rule_ref,
            "status": status,
            "support": support,
            "counterexamples": counters,
            "confidence": round(confidence, 6),
            "conditions": sorted(conditions),
            "parents": sorted(parents),
        }, ensure_ascii=False, sort_keys=True)
        return "gene:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:18]

    @staticmethod
    def _same_snapshot(node: RuleGenealogyNode, *, status: str, support: int,
                       counters: int, confidence: float, conditions: list[str]) -> bool:
        return (
            node.status == status
            and node.support == support
            and node.counterexamples == counters
            and abs(node.confidence - confidence) < 1e-9
            and sorted(x.render() for x in node.conditions) == sorted(conditions)
        )

    def _latest_by_rule(self) -> dict[str, RuleGenealogyNode]:
        out: dict[str, RuleGenealogyNode] = {}
        for node in self.store.load_rule_genealogy_nodes():
            old = out.get(node.rule_ref)
            if old is None or (node.updated_at, node.generation) > (old.updated_at, old.generation):
                out[node.rule_ref] = node
        return out

    def sync(self, candidates: list[LearnedRuleCandidate], variants: list[EvolvedRuleVariant]) -> list[RuleGenealogyNode]:
        latest = self._latest_by_rule()
        created: list[RuleGenealogyNode] = []

        # Roots and revisions for generalized experience rules.
        for candidate in candidates:
            rule_ref = f"candidate:{candidate.signature}"
            prev = latest.get(rule_ref)
            if prev and self._same_snapshot(
                prev, status=candidate.status, support=candidate.support,
                counters=candidate.counterexamples, confidence=candidate.confidence,
                conditions=[],
            ):
                continue
            parents = [prev.id] if prev else []
            mutation = "evidence_revision" if prev else "experience_rule_origin"
            generation = (prev.generation + 1) if prev else 0
            node = RuleGenealogyNode(
                id=self._node_id(rule_ref, candidate.status, candidate.support, candidate.counterexamples,
                                 candidate.confidence, [], parents),
                rule_ref=rule_ref,
                rule_kind="learned_candidate",
                base_signature=candidate.signature,
                parent_ids=parents,
                generation=generation,
                mutation=mutation,
                status=candidate.status,
                premises=list(candidate.premises),
                conclusion=candidate.conclusion,
                support=candidate.support,
                counterexamples=candidate.counterexamples,
                confidence=candidate.confidence,
                environments=list(candidate.environments),
                rationale=(
                    "经验规则首次形成。" if prev is None else
                    f"新增证据后由 {prev.status} 演化为 {candidate.status}。"
                ),
                created_at=time.time(), updated_at=time.time(),
            )
            self.store.save_rule_genealogy_node(node)
            latest[rule_ref] = node
            created.append(node)

        by_base_variant_nodes: dict[str, list[RuleGenealogyNode]] = defaultdict(list)
        for node in self.store.load_rule_genealogy_nodes():
            if node.rule_kind == "evolved_variant":
                by_base_variant_nodes[node.base_signature].append(node)

        # Conditional branches and their later revisions.
        for variant in variants:
            rule_ref = f"variant:{variant.id}"
            prev = latest.get(rule_ref)
            conditions = [x.render() for x in variant.conditions]
            if prev and self._same_snapshot(
                prev, status=variant.status, support=variant.support,
                counters=variant.counterexamples, confidence=variant.confidence,
                conditions=conditions,
            ):
                continue

            parents: list[str] = []
            mutation = "variant_revision"
            if prev:
                parents.append(prev.id)
            else:
                base = latest.get(f"candidate:{variant.base_signature}")
                if base:
                    parents.append(base.id)
                # Link the new branch to the latest prior branch with the same polarity.
                siblings = [
                    n for n in by_base_variant_nodes.get(variant.base_signature, [])
                    if n.conclusion is not None and n.conclusion.negated == variant.conclusion.negated
                ]
                if siblings:
                    sibling = max(siblings, key=lambda n: (n.updated_at, n.generation))
                    if sibling.id not in parents:
                        parents.append(sibling.id)
                    mutation = "condition_refinement"
                else:
                    mutation = "counterexample_split"

            parent_nodes = [n for n in self.store.load_rule_genealogy_nodes() if n.id in parents]
            generation = 1 + max((n.generation for n in parent_nodes), default=-1)
            node = RuleGenealogyNode(
                id=self._node_id(rule_ref, variant.status, variant.support, variant.counterexamples,
                                 variant.confidence, conditions, parents),
                rule_ref=rule_ref,
                rule_kind="evolved_variant",
                base_signature=variant.base_signature,
                parent_ids=parents,
                generation=generation,
                mutation=mutation if variant.status != "superseded" else "superseded_by_evidence",
                status=variant.status,
                premises=list(variant.premises),
                conclusion=variant.conclusion,
                conditions=list(variant.conditions),
                support=variant.support,
                counterexamples=variant.counterexamples,
                confidence=variant.confidence,
                environments=list(variant.environments),
                rationale=variant.rationale,
                created_at=time.time(), updated_at=time.time(),
            )
            self.store.save_rule_genealogy_node(node)
            latest[rule_ref] = node
            by_base_variant_nodes[variant.base_signature].append(node)
            created.append(node)
        return created

    def ancestry(self, rule_ref: str) -> list[RuleGenealogyNode]:
        nodes = {n.id: n for n in self.store.load_rule_genealogy_nodes()}
        latest = self._latest_by_rule().get(rule_ref)
        if latest is None:
            return []
        ordered: list[RuleGenealogyNode] = []
        seen: set[str] = set()

        def walk(node: RuleGenealogyNode) -> None:
            if node.id in seen:
                return
            for pid in node.parent_ids:
                parent = nodes.get(pid)
                if parent:
                    walk(parent)
            seen.add(node.id)
            ordered.append(node)

        walk(latest)
        return ordered

    def snapshot(self, limit: int = 60) -> dict:
        nodes = self.store.load_rule_genealogy_nodes()
        latest = self._latest_by_rule()
        max_generation = max((n.generation for n in nodes), default=0)
        return {
            "nodes": [
                {
                    "id": n.id, "rule_ref": n.rule_ref, "rule_kind": n.rule_kind,
                    "base_signature": n.base_signature, "parent_ids": n.parent_ids,
                    "generation": n.generation, "mutation": n.mutation, "status": n.status,
                    "premises": [a.render() for a in n.premises],
                    "conclusion": n.conclusion.render() if n.conclusion else None,
                    "conditions": [a.render() for a in n.conditions],
                    "support": n.support, "counterexamples": n.counterexamples,
                    "confidence": n.confidence, "environments": n.environments,
                    "rationale": n.rationale,
                }
                for n in nodes[-limit:]
            ],
            "total_nodes": len(nodes),
            "latest_rule_refs": len(latest),
            "max_generation": max_generation,
            "authority": "audit-history-not-direct-proof",
        }
