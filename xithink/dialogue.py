from __future__ import annotations

from dataclasses import asdict
from xithink.engine import XThinkEngine


class DialogueService:
    """Conversation adapter whose final factual answers are symbolic-only."""

    def __init__(self, engine: XThinkEngine) -> None:
        self.engine = engine

    def respond(self, text: str, cycles: int = 3, environment: str | None = None) -> dict:
        frame = self.engine.semantic.parse(text)
        thoughts = self.engine.think(text, cycles=cycles, environment=environment)
        symbolic = self.engine.answer_frame(frame, environment=environment)
        useful = [t for t in thoughts if t.kind != "halt"]

        def proof_uses_experience(proof) -> bool:
            if proof.source.startswith("worldview:"):
                return True
            if (proof.rule_name or "").startswith("experience:"):
                return True
            return any(proof_uses_experience(p) for p in proof.premises)

        experience_grounded = any(proof_uses_experience(p) for p in symbolic.proofs)
        perspective_report = self.engine.last_trace.get("multi_perspective", {})
        conditional_chains = [c for c in perspective_report.get("chains", []) if c.get("conditions")]

        if symbolic.answered:
            if environment and experience_grounded:
                reply = f"基于‘{environment}’中积累的个人经验世界观，{symbolic.text}"
            else:
                reply = symbolic.text
        elif conditional_chains:
            conditions = []
            for chain in conditional_chains:
                for condition in chain.get("conditions", []):
                    if condition not in conditions:
                        conditions.append(condition)
            reply = (
                f"当前没有唯一符号证明答案。{perspective_report.get('summary', '')}"
                f" 需要进一步确认的条件：{'；'.join(conditions[:6])}。"
            )
        elif useful:
            # Exploratory thought is clearly labelled; it is not promoted to a
            # factual answer just because it scored highly.
            best = max(useful, key=lambda t: (t.confidence, -t.cycle))
            reply = f"当前没有符号证明答案。探索结果：{best.content}"
        elif frame.questions:
            reply = f"关于“{frame.questions[0]}”，当前符号知识不足。我会把它保留为待验证问题。"
        elif frame.events:
            reply = f"已记录事件：{frame.events[0].render()}。"
        elif frame.facts:
            reply = f"已写入长期符号记忆：{frame.facts[0].render()}。"
        else:
            reply = "我已经记录这次输入，但还没有足够符号前提形成可证明判断。"

        return {
            "reply": reply,
            "answer_mode": (
                "contextual-symbolic-proof" if symbolic.answered and experience_grounded
                else "symbolic-proof" if symbolic.answered
                else "conditional-analysis" if conditional_chains
                else "exploratory"
            ),
            "symbolic_answer": {
                "answered": symbolic.answered,
                "text": symbolic.text,
                "confidence": symbolic.confidence,
                "proofs": [p.to_dict() for p in symbolic.proofs],
                "environment": environment,
                "experience_grounded": experience_grounded,
            },
            "thoughts": [t.to_dict() for t in thoughts],
            "cognition": {
                "trace": self.engine.last_trace,
                "concept_graph": self.engine.graph.snapshot(limit=12),
                "embedding_items": self.engine.embedding_memory.count(),
                "memory_layers": self.engine.snapshot()["memory_layers"],
                "experience_rule_learning": self.engine.experience_rules.snapshot(),
                "rule_evolution": self.engine.rule_evolution.snapshot(),
                "rule_genealogy": self.engine.rule_genealogy.snapshot(),
                "active_hypothesis_testing": self.engine.active_testing.snapshot(),
                "cognitive_profile": asdict(self.engine.cognitive_profile),
                "multi_perspective": perspective_report,
                "personal_symbols": self.engine.personal_symbols.snapshot(),
                "worldview": self.engine.worldview.snapshot(environment),
                "perspective_resonance": self.engine.last_trace.get("perspective_resonance", {}),
            },
            "semantic": {
                "environment": environment,
                "source": frame.source,
                "modality": frame.modality,
                "concepts": frame.concepts,
                "questions": frame.questions,
                "question_intents": [asdict(q) for q in frame.question_intents],
                "facts": [f.render() for f in frame.facts],
                "events": [asdict(e) for e in frame.events],
                "time_refs": frame.time_refs,
            },
        }
