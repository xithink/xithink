from xithink.engine import XThinkEngine
from xithink.models import MemoryMatch
from xithink.modules.perspective import PerspectiveResonanceEngine


def test_same_engine_can_hold_different_environment_worldviews():
    e = XThinkEngine(":memory:")
    e.think("陌生人意味着风险。", cycles=1, environment="城市")
    e.think("陌生人意味着帮助。", cycles=1, environment="村庄")

    city = e.semantic.parse("陌生人是什么？")
    village = e.semantic.parse("陌生人是什么？")
    city_answer = e.answer_frame(city, environment="城市")
    village_answer = e.answer_frame(village, environment="村庄")
    assert city_answer.answered and "风险" in city_answer.text
    assert village_answer.answered and "帮助" in village_answer.text
    assert city_answer.text != village_answer.text

    # Environment observations never silently become universal facts.
    universal = e.answer_frame(e.semantic.parse("陌生人是什么？"))
    assert universal.answered is False
    e.close()


def test_contradictory_experience_reduces_worldview_confidence():
    e = XThinkEngine(":memory:")
    e.think("陌生人意味着风险。", cycles=1, environment="城市")
    e.think("陌生人不意味着风险。", cycles=1, environment="城市")
    beliefs = [b for b in e.memory.load_worldview_beliefs("城市") if b.atom.subject == "陌生人"]
    assert beliefs
    assert all(abs(b.confidence - 0.5) < 1e-9 for b in beliefs)
    assert e.worldview.facts("城市") == []
    e.close()


def test_repeated_experience_forms_personal_symbol_used_in_activation():
    e = XThinkEngine(":memory:")
    e.think("自由意味着选择。", cycles=1, environment="成长环境")
    e.think("自由意味着选择。", cycles=1, environment="成长环境")
    symbols = [s for s in e.personal_symbols.all() if set(s.members) == {"自由", "选择"}]
    assert symbols and symbols[0].exposures >= 2
    activation, active = e.personal_symbols.activate(["自由"], environment="成长环境")
    assert "选择" in activation
    assert any(s.name in activation for s in active)
    # A private symbol is a thought primitive, not a truth-bearing Atom.
    assert not any(f.subject.startswith("经验符号[") for f in e.reasoner.facts)
    e.close()


def test_perspective_resonance_uses_own_analogous_experience_without_asserting_feelings():
    engine = PerspectiveResonanceEngine()
    recalled = [
        MemoryMatch(
            "experience:e1", "experience", "受伤以后寻求保护", 0.82,
            concepts=["受伤", "保护", "害怕"],
            metadata={"reward": -0.6},
        ),
        MemoryMatch(
            "experience:e2", "experience", "遇到危险后得到帮助", 0.65,
            concepts=["危险", "帮助", "安全"],
            metadata={"reward": 0.4},
        ),
    ]
    resonance = engine.assess(["受伤", "危险"], recalled, empathy_weight=0.8)
    assert resonance.score > 0
    assert "受伤" in resonance.shared_concepts
    assert resonance.activation
    assert "不代表对他人主观感受的事实判断" in resonance.note


def test_worldview_rules_symbols_survive_restart(tmp_path):
    db = tmp_path / "worldview.db"
    e = XThinkEngine(db)
    for triple in [
        ("高温", "电池膨胀", "设备故障"),
        ("高湿", "传感器漂移", "系统误报"),
        ("强震", "结构变形", "建筑失效"),
    ]:
        x, y, z = triple
        e.think(f"{x}导致{y}。{y}导致{z}。{x}间接导致{z}。", cycles=1, environment="实验室A")
    e.think("自由意味着选择。", cycles=1, environment="实验室A")
    e.think("自由意味着选择。", cycles=1, environment="实验室A")
    e.close()

    e2 = XThinkEngine(db)
    assert any(c.status == "verified_scoped" for c in e2.memory.load_learned_rule_candidates())
    assert any(set(s.members) == {"自由", "选择"} and s.exposures >= 2 for s in e2.personal_symbols.all())
    assert e2.worldview.facts("实验室A")
    e2.think(
        "高压导致材料变形。材料变形导致设备失效。为什么高压间接导致设备失效？",
        cycles=1,
        environment="实验室A",
    )
    assert e2.last_trace["symbolic_answer"]["answered"] is True
    e2.close()


def test_experience_resonance_is_visible_in_integrated_thought_trace():
    e = XThinkEngine(":memory:")
    e.think("受伤意味着疼痛。疼痛导致寻找帮助。", cycles=1, environment="过去经历")
    e.think("受伤是什么？", cycles=1, environment="当前环境")
    resonance = e.last_trace["perspective_resonance"]
    assert resonance["score"] > 0
    assert "受伤" in resonance["shared_concepts"]
    assert "疼痛" in resonance["activation"] or "寻找帮助" in resonance["activation"]
    assert resonance["valence"] < 0
    assert "不代表对他人主观感受的事实判断" in resonance["note"]
    e.close()
