from __future__ import annotations

from itertools import combinations
from xithink.models import Hypothesis, MemoryMatch, PersonalSymbol, SemanticFrame
from xithink.modules.symbolic import SymbolicReasoner
from xithink.modules.memory import MemoryStore


class HypothesisGenerator:
    def generate(
        self,
        frame: SemanticFrame,
        reasoner: SymbolicReasoner,
        activation: dict[str, float],
        memory: MemoryStore,
        recalled: list[MemoryMatch] | None = None,
        personal_symbols: list[PersonalSymbol] | None = None,
        limit: int = 8,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        recalled = recalled or []
        personal_symbols = personal_symbols or []

        # 1. Explicit questions become investigation hypotheses.
        for q in frame.questions:
            statement = f"需要检验：{q}"
            if not memory.contains_content(statement):
                out.append(Hypothesis(statement, evidence=0.25, consistency=1.0, novelty=0.75, uncertainty=0.75, tags=["question"]))

        # 2. Contradictions receive highest priority because they can drive thought.
        for a, b in reasoner.conflicts():
            statement = f"存在冲突：『{a.render()}』与『{b.render()}』不能同时直接成立，需要区分条件或语境"
            if not memory.contains_content(statement):
                out.append(Hypothesis(statement, [a, b], evidence=0.88, consistency=0.92, novelty=0.70, uncertainty=0.38, tags=["conflict"]))

        # 3. Newly inferred facts are legitimate thought candidates.
        for atom in reasoner.facts[-8:]:
            if atom.source.startswith("inference:"):
                statement = f"由现有规则可推出：{atom.render()}"
                if not memory.contains_content(statement):
                    out.append(Hypothesis(statement, [atom], evidence=atom.confidence, consistency=0.95, novelty=0.55, uncertainty=1 - atom.confidence, tags=["inference"]))

        # 4. Association creates exploratory hypotheses, not asserted facts.
        ranked = [x for x in activation.items() if x[1] < 0.999]
        for (a, av), (b, bv) in combinations(ranked[:5], 2):
            if a == b:
                continue
            statement = f"探索假设：『{a}』与『{b}』之间可能存在尚未显式建模的关系"
            if not memory.contains_content(statement):
                strength = min(0.8, (av + bv) / 2)
                out.append(Hypothesis(statement, evidence=strength * 0.45, consistency=0.9, novelty=0.85, uncertainty=0.7, tags=["association"]))

        # 5. Semantic memory can reopen a previous experience under new wording.
        # It remains a hypothesis: recall is context, never automatically truth.
        for match in recalled[:3]:
            if match.score < 0.30 or match.kind not in {"fact", "event", "thought", "experience"}:
                continue
            statement = f"记忆关联：当前问题与既有{match.kind}『{match.text}』相似，应检验两者是否共享条件或因果关系"
            if not memory.contains_content(statement):
                out.append(Hypothesis(
                    statement, evidence=min(0.55, match.score * 0.55),
                    consistency=0.92, novelty=0.52, uncertainty=0.58,
                    tags=["experience-recall" if match.kind == "experience" else "memory-recall"], memory_support=match.score,
                    metadata={"memory_id": match.item_id, "memory_score": match.score},
                ))

        # 6. Stable personal symbols may become explicit thought vocabulary.
        # They remain exploratory representations rather than facts.
        for symbol in personal_symbols[:2]:
            statement = (
                f"经验符号激活：『{symbol.name}』由反复经历中的"
                f"{'、'.join(symbol.members)}共同形成，可用于组织当前思考，但不能单独证明事实"
            )
            if not memory.contains_content(statement):
                out.append(Hypothesis(
                    statement, evidence=min(0.45, symbol.strength * 0.40),
                    consistency=0.95, novelty=0.68, uncertainty=0.62,
                    tags=["personal-symbol"], memory_support=min(0.65, symbol.strength),
                    metadata={"personal_symbol_id": symbol.id, "members": symbol.members},
                ))

        # 7. If we still have no candidate, reflect on the strongest concept.
        if not out and activation:
            top = next(iter(activation))
            statement = f"反思：关于『{top}』，当前知识不足，应寻找可验证的新关系或反例"
            if not memory.contains_content(statement):
                out.append(Hypothesis(statement, evidence=0.15, consistency=1.0, novelty=0.65, uncertainty=0.85, tags=["reflection"]))

        # stable deterministic ordering before downstream scoring
        return out[:limit]
