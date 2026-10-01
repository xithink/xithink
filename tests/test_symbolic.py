from xithink.models import Atom, Rule
from xithink.modules.symbolic import SymbolicReasoner


def test_variable_rule_forward_chain():
    r = SymbolicReasoner()
    r.add_fact(Atom("自由", "意味着", "选择"))
    r.add_fact(Atom("选择", "导致", "责任"))
    r.add_rule(Rule(
        (Atom("?x", "意味着", "?y"), Atom("?y", "导致", "?z")),
        Atom("?x", "间接导致", "?z"), name="chain", confidence=0.9,
    ))
    inferred = r.infer()
    assert any(x.key()[:3] == ("自由", "间接导致", "责任") for x in inferred)


def test_conflict_detection():
    r = SymbolicReasoner()
    r.add_fact(Atom("A", "是", "B"))
    r.add_fact(Atom("A", "是", "B", negated=True))
    assert len(r.conflicts()) == 1
