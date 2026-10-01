from xithink.modules.semantic import SemanticRecognizer


def test_semantic_v2_source_modality_time_event():
    parser = SemanticRecognizer()
    frame = parser.parse("据导师说,今天机器人可能不支持这个决定。")
    assert frame.source == "导师"
    assert "今天" in frame.time_refs
    assert frame.modality == "possible"
    assert len(frame.events) == 1
    event = frame.events[0]
    assert event.actor == "机器人"
    assert event.action == "支持"
    assert event.negated is True
    assert event.object == "这个决定"
    assert event.source == "导师"
    assert event.time_ref == "今天"


def test_semantic_v2_negated_fact():
    parser = SemanticRecognizer()
    frame = parser.parse("自由不意味着任性。")
    assert len(frame.facts) == 1
    fact = frame.facts[0]
    assert fact.subject == "自由"
    assert fact.predicate == "意味着"
    assert fact.object == "任性"
    assert fact.negated is True


def test_question_without_question_mark_is_not_fact():
    parser = SemanticRecognizer()
    frame = parser.parse("什么是自由")
    assert frame.questions == ["什么是自由"]
    assert frame.facts == []


def test_fact_relation_is_not_double_counted_as_event():
    parser = SemanticRecognizer()
    frame = parser.parse("自由意味着选择。")
    assert len(frame.facts) == 1
    assert frame.events == []


def test_question_extracts_target_concept():
    frame = SemanticRecognizer().parse("什么是自由？")
    assert "自由" in frame.concepts


def test_negated_inferred_relation_can_be_parsed():
    frame = SemanticRecognizer().parse("自由不间接导致风险。")
    assert frame.facts and frame.facts[0].predicate == "间接导致"
    assert frame.facts[0].negated is True
