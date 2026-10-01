from __future__ import annotations

from xithink.models import SelfState


class ValueSystem:
    KEYWORDS = {
        "truth": ("事实", "检验", "证据", "真实", "反例", "推出"),
        "curiosity": ("探索", "问题", "未知", "关系", "为什么", "可能"),
        "empathy": ("人", "感受", "伤害", "尊重", "帮助"),
        "freedom": ("自由", "选择", "自主", "限制"),
        "safety": ("安全", "风险", "危险", "保护", "冲突"),
    }

    def alignment(self, text: str, state: SelfState) -> float:
        matches = []
        for value, keywords in self.KEYWORDS.items():
            if any(k in text for k in keywords):
                matches.append(state.values.get(value, 0.5))
        return sum(matches) / len(matches) if matches else 0.5
