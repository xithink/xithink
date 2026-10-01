from xithink.modules.semantic import SemanticRecognizer


def test_parse_facts_and_question():
    p = SemanticRecognizer()
    frame = p.parse("自由意味着选择。选择导致风险。什么是自由？")
    assert len(frame.facts) == 2
    assert frame.facts[0].subject == "自由"
    assert frame.facts[0].predicate == "意味着"
    assert frame.facts[0].object == "选择"
    assert frame.questions == ["什么是自由"]


def test_parse_rule():
    p = SemanticRecognizer()
    frame = p.parse("如果自由需要选择，那么限制选择限制自由。")
    assert len(frame.rules) == 1
    assert frame.rules[0].premises[0].predicate == "需要"
