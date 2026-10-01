from xithink.engine import XThinkEngine


def _teach_cause_chain(engine: XThinkEngine, environment: str, examples: list[tuple[str, str, str]]):
    for x, y, z in examples:
        engine.think(
            f"{x}导致{y}。{y}导致{z}。{x}间接导致{z}。",
            cycles=1,
            environment=environment,
        )


def test_repeated_experience_generates_scoped_symbolic_rule():
    e = XThinkEngine(":memory:")
    _teach_cause_chain(e, "实验室A", [
        ("高温", "电池膨胀", "设备故障"),
        ("高湿", "传感器漂移", "系统误报"),
        ("强震", "结构变形", "建筑失效"),
    ])
    candidates = e.memory.load_learned_rule_candidates()
    learned = [c for c in candidates if c.conclusion.predicate == "间接导致"]
    assert learned
    assert learned[0].status == "verified_scoped"
    assert learned[0].support == 3
    assert learned[0].counterexamples == 0

    frame = e.semantic.parse("为什么高压间接导致设备失效？")
    # Current premises plus the learned scoped rule can prove a new case.
    e.think(
        "高压导致材料变形。材料变形导致设备失效。为什么高压间接导致设备失效？",
        cycles=1,
        environment="实验室A",
    )
    assert e.last_trace["symbolic_answer"]["answered"] is True
    proof = e.last_trace["symbolic_answer"]["proofs"][0]
    assert proof["rule_name"].startswith("experience:")
    e.close()


def test_scoped_rule_does_not_leak_to_other_environment():
    e = XThinkEngine(":memory:")
    _teach_cause_chain(e, "实验室A", [
        ("高温", "电池膨胀", "设备故障"),
        ("高湿", "传感器漂移", "系统误报"),
        ("强震", "结构变形", "建筑失效"),
    ])
    e.think(
        "高压导致材料变形。材料变形导致设备失效。为什么高压间接导致设备失效？",
        cycles=1,
        environment="实验室B",
    )
    assert e.last_trace["symbolic_answer"]["answered"] is False
    e.close()


def test_cross_environment_rule_can_become_global():
    e = XThinkEngine(":memory:")
    for env, triple in [
        ("实验室A", ("高温", "电池膨胀", "设备故障")),
        ("工厂B", ("高湿", "传感器漂移", "系统误报")),
        ("野外C", ("强震", "结构变形", "建筑失效")),
    ]:
        _teach_cause_chain(e, env, [triple])
    candidates = e.memory.load_learned_rule_candidates()
    learned = [c for c in candidates if c.conclusion.predicate == "间接导致"]
    assert learned[0].status == "verified_global"

    # No environment is supplied: the promoted rule is now part of universal
    # long-term symbolic reasoning, but the current premises still need facts.
    e.think("高压导致材料变形。材料变形导致设备失效。为什么高压间接导致设备失效？", cycles=1)
    assert e.last_trace["symbolic_answer"]["answered"] is True
    assert e.last_trace["symbolic_answer"]["proofs"][0]["rule_name"].startswith("experience:")
    e.close()


def test_explicit_counterexamples_prevent_verification():
    e = XThinkEngine(":memory:")
    _teach_cause_chain(e, "A", [("甲", "乙", "丙"), ("丁", "戊", "己")])
    # Same premises, explicit opposite conclusions.
    e.think("庚导致辛。辛导致壬。庚不间接导致壬。", cycles=1, environment="A")
    e.think("癸导致子。子导致丑。癸不间接导致丑。", cycles=1, environment="A")
    positive = [
        c for c in e.memory.load_learned_rule_candidates()
        if c.conclusion.predicate == "间接导致" and not c.conclusion.negated
    ][0]
    assert positive.counterexamples >= 2
    assert not positive.status.startswith("verified")
    e.close()
