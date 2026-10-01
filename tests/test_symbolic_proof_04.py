from xithink.engine import XThinkEngine
from xithink.modules.semantic import SemanticRecognizer


def test_question_intents_are_structured():
    p = SemanticRecognizer()
    definition = p.parse("什么是自由？").question_intents[0]
    assert definition.kind == "definition"
    assert definition.subject == "自由"

    why = p.parse("为什么自由间接导致风险？").question_intents[0]
    assert why.kind == "why_relation"
    assert why.subject == "自由"
    assert why.predicate == "间接导致"
    assert why.object == "风险"


def test_symbolic_answer_contains_proof_tree(tmp_path):
    e = XThinkEngine(tmp_path / "proof.db")
    e.think("自由意味着选择。选择导致风险。", cycles=1)
    frame = e.semantic.parse("为什么自由间接导致风险？")
    answer = e.answer_frame(frame)
    assert answer.answered is True
    assert answer.proofs
    proof = answer.proofs[0]
    assert proof.kind == "inference"
    assert proof.rule_name == "meaning-cause-chain"
    assert len(proof.premises) == 2
    assert {p.conclusion.render() for p in proof.premises} == {
        "自由 意味着 选择",
        "选择 导致 风险",
    }
    e.close()


def test_dialogue_prefers_symbolic_proof_over_exploratory_thought(tmp_path):
    from xithink.dialogue import DialogueService

    e = XThinkEngine(tmp_path / "dialogue04.db")
    result = DialogueService(e).respond("自由意味着选择。选择导致风险。为什么自由间接导致风险？", cycles=2)
    assert result["answer_mode"] == "symbolic-proof"
    assert "自由 间接导致 风险" in result["reply"]
    assert result["symbolic_answer"]["proofs"]
    e.close()


def test_proof_tree_is_reconstructed_after_restart(tmp_path):
    path = tmp_path / "proof-restart.db"
    e1 = XThinkEngine(path)
    e1.think("自由意味着选择。选择导致风险。", cycles=1)
    e1.close()

    e2 = XThinkEngine(path)
    answer = e2.answer_frame(e2.semantic.parse("为什么自由间接导致风险？"))
    assert answer.answered
    assert answer.proofs[0].kind == "inference"
    assert answer.proofs[0].rule_name == "meaning-cause-chain"
    assert len(answer.proofs[0].premises) == 2
    e2.close()
