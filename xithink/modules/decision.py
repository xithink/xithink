from __future__ import annotations

from xithink.models import Decision, Hypothesis, SelfState
from xithink.modules.value import ValueSystem


class DecisionEngine:
    # 0.3 adds explicit second-order signals. Counterevidence is not rewarded;
    # counter_resistance measures how well the candidate survived falsification.
    WEIGHTS = {
        "evidence": 0.30,
        "consistency": 0.22,
        "value_alignment": 0.15,
        "novelty": 0.10,
        "certainty": 0.10,
        "memory_support": 0.08,
        "counter_resistance": 0.05,
    }

    def __init__(self, value_system: ValueSystem | None = None) -> None:
        self.value_system = value_system or ValueSystem()

    def evaluate(self, h: Hypothesis, state: SelfState) -> Decision:
        components = {
            "evidence": h.evidence,
            "consistency": h.consistency,
            "value_alignment": self.value_system.alignment(h.statement, state),
            "novelty": h.novelty * state.traits.get("novelty_seeking", 0.5),
            "certainty": 1.0 - h.uncertainty,
            "memory_support": h.memory_support,
            "counter_resistance": 1.0 - h.counterevidence,
        }
        score = sum(self.WEIGHTS[k] * components[k] for k in self.WEIGHTS)
        rationale = ", ".join(f"{k}={v:.2f}" for k, v in components.items())
        return Decision(h, score, components, rationale)

    def select(self, hypotheses: list[Hypothesis], state: SelfState) -> Decision | None:
        if not hypotheses:
            return None
        decisions = [self.evaluate(h, state) for h in hypotheses]
        decisions.sort(key=lambda d: (d.score, d.hypothesis.statement), reverse=True)
        return decisions[0]
