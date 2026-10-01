from xithink.engine import XThinkEngine


def _teach_split(engine: XThinkEngine, environment: str = "测试场") -> None:
    for text in [
        "甲1导致乙1。乙1导致丙1。甲1间接导致丙1。甲1是条件甲。",
        "甲2导致乙2。乙2导致丙2。甲2间接导致丙2。甲2是条件甲。",
        "丁1导致戊1。戊1导致己1。丁1不间接导致己1。丁1是条件乙。",
        "丁2导致戊2。戊2导致己2。丁2不间接导致己2。丁2是条件乙。",
    ]:
        engine.think(text, cycles=1, environment=environment)


def test_counterexamples_split_base_rule_into_opposite_conditional_rules():
    e = XThinkEngine(":memory:")
    _teach_split(e)
    variants = e.memory.load_evolved_rule_variants()
    positive = [v for v in variants if not v.conclusion.negated and any(c.object == "条件甲" for c in v.conditions)]
    negative = [v for v in variants if v.conclusion.negated and any(c.object == "条件乙" for c in v.conditions)]
    assert positive and negative
    assert positive[0].status == "verified_conditional_scoped"
    assert negative[0].status == "verified_conditional_scoped"

    e.think("新甲导致新乙。新乙导致新丙。新甲是条件甲。为什么新甲间接导致新丙？", cycles=1, environment="测试场")
    assert e.last_trace["symbolic_answer"]["answered"] is True
    proof = e.last_trace["symbolic_answer"]["proofs"][0]
    assert proof["rule_name"].startswith("evolved:")
    assert any(p["conclusion"] == "新甲 是 条件甲" for p in proof["premises"])

    e.think("新丁导致新戊。新戊导致新己。新丁是条件乙。为什么新丁间接导致新己？", cycles=1, environment="测试场")
    answer = e.last_trace["symbolic_answer"]
    assert answer["answered"] is True
    assert "相反结论" in answer["text"]
    assert answer["proofs"][0]["rule_name"].startswith("evolved:")
    e.close()


def test_multi_perspective_trace_keeps_both_sides_and_conditions():
    e = XThinkEngine(":memory:")
    _teach_split(e)
    e.think("样本甲导致样本乙。样本乙导致样本丙。样本甲是条件甲。为什么样本甲间接导致样本丙？", cycles=1, environment="测试场")
    report = e.last_trace["multi_perspective"]
    assert report["target"] == "样本甲 间接导致 样本丙"
    assert any(c["stance"] == "support" and c["proof"] for c in report["chains"])
    assert any(c["stance"] == "oppose" and "条件乙" in " ".join(c["conditions"]) for c in report["chains"])
    assert "历史反例" in report["summary"] or "正反" in report["summary"]
    e.close()


def test_when_both_conditions_hold_symbolic_answer_exposes_both_conclusions():
    e = XThinkEngine(":memory:")
    _teach_split(e)
    e.think(
        "双条件甲导致双条件乙。双条件乙导致双条件丙。双条件甲是条件甲。双条件甲是条件乙。为什么双条件甲间接导致双条件丙？",
        cycles=1,
        environment="测试场",
    )
    answer = e.last_trace["symbolic_answer"]
    assert answer["answered"] is True
    assert "同时存在正反结论" in answer["text"]
    assert len(answer["proofs"]) >= 2
    report = e.last_trace["multi_perspective"]
    assert any(c["stance"] == "support" and c["proof"] for c in report["chains"])
    assert any(c["stance"] == "oppose" and c["proof"] for c in report["chains"])
    assert "不应直接二选一" in report["summary"]
    e.close()


def test_rejected_old_experience_rule_is_removed_from_active_reasoning():
    e = XThinkEngine(":memory:")
    for x, y, z in [("甲", "乙", "丙"), ("丁", "戊", "己"), ("庚", "辛", "壬")]:
        e.think(f"{x}导致{y}。{y}导致{z}。{x}间接导致{z}。", cycles=1, environment="A")
    assert any(c.status == "verified_scoped" for c in e.memory.load_learned_rule_candidates() if not c.conclusion.negated)

    for x, y, z in [("子", "丑", "寅"), ("卯", "辰", "巳"), ("午", "未", "申")]:
        e.think(f"{x}导致{y}。{y}导致{z}。{x}不间接导致{z}。", cycles=1, environment="A")
    positive = [c for c in e.memory.load_learned_rule_candidates() if c.conclusion.predicate == "间接导致" and not c.conclusion.negated][0]
    assert not positive.status.startswith("verified")

    e.think("新一导致新二。新二导致新三。为什么新一间接导致新三？", cycles=1, environment="A")
    assert e.last_trace["symbolic_answer"]["answered"] is False
    e.close()


def test_evolved_rules_survive_restart_and_keep_conditional_proof(tmp_path):
    db = tmp_path / "evolution.db"
    e = XThinkEngine(db)
    _teach_split(e, "道路环境")
    assert e.memory.load_evolved_rule_variants()
    e.close()

    e2 = XThinkEngine(db)
    e2.think("重启甲导致重启乙。重启乙导致重启丙。重启甲是条件甲。为什么重启甲间接导致重启丙？", cycles=1, environment="道路环境")
    answer = e2.last_trace["symbolic_answer"]
    assert answer["answered"] is True
    assert answer["proofs"][0]["rule_name"].startswith("evolved:")
    assert e2.last_trace["rule_evolution"]["variants"]
    e2.close()


def test_rule_evolution_can_discover_two_condition_conjunction():
    e = XThinkEngine(":memory:")
    # In positive cases A and B occur together. Each condition also appears in
    # some negative case, so neither single condition is sufficient; only A+B
    # separates the groups.
    for text in [
        "正1导致中1。中1导致果1。正1间接导致果1。正1是条件A。正1需要条件B。",
        "正2导致中2。中2导致果2。正2间接导致果2。正2是条件A。正2需要条件B。",
        "反1导致中3。中3导致果3。反1不间接导致果3。反1是条件A。反1需要条件C。",
        "反2导致中4。中4导致果4。反2不间接导致果4。反2需要条件B。反2需要条件C。",
    ]:
        e.think(text, cycles=1, environment="组合条件场")

    variants = [
        v for v in e.memory.load_evolved_rule_variants()
        if not v.conclusion.negated and len(v.conditions) == 2
    ]
    assert variants
    chosen = variants[0]
    rendered = {c.render() for c in chosen.conditions}
    assert any("条件A" in x for x in rendered)
    assert any("条件B" in x for x in rendered)

    e.think(
        "新正导致新中。新中导致新果。新正是条件A。新正需要条件B。为什么新正间接导致新果？",
        cycles=1,
        environment="组合条件场",
    )
    assert e.last_trace["symbolic_answer"]["answered"] is True
    proof = e.last_trace["symbolic_answer"]["proofs"][0]
    assert proof["rule_name"].startswith("evolved:")
    assert len(proof["premises"]) >= 4
    e.close()


def test_dialogue_requests_missing_conditions_instead_of_guessing():
    from xithink.dialogue import DialogueService

    e = XThinkEngine(":memory:")
    _teach_split(e)
    service = DialogueService(e)
    result = service.respond("为什么未知甲间接导致未知丙？", cycles=1, environment="测试场")
    assert result["answer_mode"] == "conditional-analysis"
    assert "需要进一步确认的条件" in result["reply"]
    assert "条件甲" in result["reply"] or "条件乙" in result["reply"]
    e.close()


def test_multi_perspective_compares_different_environment_logic_chains():
    e = XThinkEngine(":memory:")
    e.think("陌生人意味着风险。", cycles=1, environment="城市")
    e.think("陌生人不意味着风险。", cycles=1, environment="村庄")
    e.think("陌生人是否意味着风险？", cycles=1, environment="城市")
    report = e.last_trace["multi_perspective"]
    assert any(c["environment"] == "城市" and c["stance"] == "support" and c["proof"] for c in report["chains"])
    assert any(c["environment"] == "村庄" and c["stance"] == "oppose" and c["proof"] for c in report["chains"])
    assert "正反结论" in report["summary"] or "适用条件" in report["summary"] or "二选一" in report["summary"]
    e.close()
