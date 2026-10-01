from __future__ import annotations

from xithink.models import (
    Atom, EvolvedRuleVariant, MultiPerspectiveReport, PerspectiveChain, QuestionIntent,
)
from xithink.modules.symbolic import SymbolicReasoner


class MultiPerspectiveAnalyzer:
    """Compare support, opposition and conditional alternatives symbolically.

    This is deliberately not an LLM debate. Every concrete chain is either a
    SymbolicProof already present in a reasoner, or an inspectable evolved rule
    that states which extra condition would make a side applicable.
    """

    @staticmethod
    def _target(intent: QuestionIntent | None) -> Atom | None:
        if not intent or not intent.subject or not intent.predicate or intent.object is None:
            return None
        if intent.kind not in {"yes_no", "why_relation"}:
            return None
        return Atom(
            intent.subject, intent.predicate, intent.object, intent.negated,
            1.0, "perspective-query",
        )

    @staticmethod
    def _find(reasoner: SymbolicReasoner, target: Atom) -> tuple[Atom | None, dict | None]:
        reasoner.infer()
        for atom in reasoner.facts:
            if atom.key() == target.key():
                proof = reasoner.proof_for(atom)
                return atom, proof.to_dict() if proof else None
        return None, None

    @staticmethod
    def _variant_applies_to_target(variant: EvolvedRuleVariant, target: Atom) -> bool:
        c = variant.conclusion
        if c.predicate != target.predicate or c.negated != target.negated:
            return False
        # Generalized experience rules use variables. Constants, when present,
        # must still match the concrete query.
        if not c.subject.startswith("?") and c.subject != target.subject:
            return False
        if not c.object.startswith("?") and c.object != target.object:
            return False
        return True

    def analyze(
        self,
        intent: QuestionIntent | None,
        *,
        universal_reasoner: SymbolicReasoner,
        contextual_reasoner: SymbolicReasoner,
        evolved_variants: list[EvolvedRuleVariant] | None = None,
        environment: str | None = None,
        alternative_contexts: dict[str, SymbolicReasoner] | None = None,
    ) -> MultiPerspectiveReport:
        target = self._target(intent)
        if target is None:
            return MultiPerspectiveReport(
                target=intent.raw if intent else "",
                environment=environment,
                summary="当前问题不是可直接比较正反逻辑链的关系命题。",
            )

        opposite = Atom(
            target.subject, target.predicate, target.object, not target.negated,
            1.0, "perspective-query",
        )
        chains: list[PerspectiveChain] = []
        seen: set[tuple[str, str, str]] = set()

        def add_proof(name: str, stance: str, atom: Atom, proof: dict | None, env: str | None, source: str) -> None:
            key = (stance, atom.render(), str(proof))
            if key in seen:
                return
            seen.add(key)
            chains.append(PerspectiveChain(
                name=name,
                stance=stance,
                conclusion=atom.render(),
                confidence=atom.confidence,
                environment=env,
                proof=proof,
                source=source,
            ))

        ua, up = self._find(universal_reasoner, target)
        uo, uop = self._find(universal_reasoner, opposite)
        if ua:
            add_proof("长期/通用视角", "support", ua, up, None, "long-term-symbolic")
        if uo:
            add_proof("长期/通用视角", "oppose", uo, uop, None, "long-term-symbolic")

        ca, cp = self._find(contextual_reasoner, target)
        co, cop = self._find(contextual_reasoner, opposite)
        if ca:
            add_proof("当前环境视角", "support", ca, cp, environment, "contextual-symbolic")
        if co:
            add_proof("当前环境视角", "oppose", co, cop, environment, "contextual-symbolic")

        for alt_env, alt_reasoner in (alternative_contexts or {}).items():
            aa, ap = self._find(alt_reasoner, target)
            ao, aop = self._find(alt_reasoner, opposite)
            if aa:
                add_proof(f"环境对照：{alt_env}", "support", aa, ap, alt_env, "alternative-environment")
            if ao:
                add_proof(f"环境对照：{alt_env}", "oppose", ao, aop, alt_env, "alternative-environment")

        unresolved: list[str] = []
        for variant in evolved_variants or []:
            if not variant.status.startswith("verified"):
                continue
            if variant.status.endswith("_scoped") and (not environment or environment not in variant.environments):
                continue
            if self._variant_applies_to_target(variant, target):
                stance = "support"
            elif self._variant_applies_to_target(variant, opposite):
                stance = "oppose"
            else:
                continue
            conditions = [x.render() for x in variant.conditions]
            condition_text = " AND ".join(conditions) or "无额外条件"
            conclusion = target.render() if stance == "support" else opposite.render()
            key = (stance, conclusion, condition_text)
            if key in seen:
                continue
            seen.add(key)
            chains.append(PerspectiveChain(
                name="条件化经验视角",
                stance=stance,
                conclusion=conclusion,
                confidence=variant.confidence,
                conditions=conditions,
                environment=environment,
                proof=None,
                source=f"rule-evolution:{variant.id}:{variant.status}",
            ))
            unresolved.extend(x for x in conditions if x not in unresolved)

        has_support = any(c.stance == "support" and c.proof for c in chains)
        has_oppose = any(c.stance == "oppose" and c.proof for c in chains)
        conditional_support = any(c.stance == "support" and c.conditions for c in chains)
        conditional_oppose = any(c.stance == "oppose" and c.conditions for c in chains)

        if has_support and has_oppose:
            summary = "正反结论都有符号证据；不应直接二选一，应继续比较适用条件与环境。"
        elif has_support and conditional_oppose:
            summary = "当前证据支持命题，但历史反例表明在另一组条件下可能得到相反结论。"
        elif has_oppose and conditional_support:
            summary = "当前证据反对命题，但历史正例表明在另一组条件下可能成立。"
        elif conditional_support and conditional_oppose:
            summary = "当前前提不足以直接证明任一侧，但经验已发现可区分正反结论的条件。"
        elif has_support:
            summary = "当前符号证据支持命题，尚未找到可证明的相反链。"
        elif has_oppose:
            summary = "当前符号证据支持相反命题，尚未找到可证明的正向链。"
        else:
            summary = "当前符号前提不足；需要补充能区分正反两侧的条件事实。"

        return MultiPerspectiveReport(
            target=target.render(),
            environment=environment,
            chains=chains,
            unresolved_conditions=unresolved,
            summary=summary,
        )
