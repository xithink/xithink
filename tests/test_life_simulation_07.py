from xithink.engine import XThinkEngine
from xithink.modules.life_simulation import profile_from_name


def test_cognitive_profile_adjusts_learning_policy():
    e = XThinkEngine(":memory:")
    p = profile_from_name("conservative")
    e.apply_cognitive_profile(p)
    assert e.experience_rules.min_support == p.rule_min_support
    assert e.rule_evolution.max_condition_terms == p.max_condition_terms
    assert e.active_testing.budget == p.active_test_budget
    assert e.state.traits["skepticism"] == p.skepticism
    e.close()


def test_life_simulation_has_age_checkpoints_and_growing_rule_lineage():
    e = XThinkEngine(":memory:")
    report = e.simulate_life(
        total_experiences=120,
        max_age=40,
        age_checkpoints=[0, 10, 20, 30, 40],
        seed=17,
        profile=profile_from_name("balanced"),
        learning_batch=20,
    )
    assert [x.age for x in report.checkpoints] == [0, 10, 20, 30, 40]
    assert [x.experiences for x in report.checkpoints] == [0, 30, 60, 90, 120]
    assert report.checkpoints[0].genealogy_nodes == 0
    assert report.checkpoints[-1].genealogy_nodes > 0
    assert report.checkpoints[-1].active_evolved_rules > 0
    assert report.checkpoints[-1].symbolic_accuracy >= report.checkpoints[0].symbolic_accuracy
    assert report.checkpoints[-1].abstention_accuracy >= 0.5
    e.close()
