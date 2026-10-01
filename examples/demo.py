from xithink import XThinkEngine

engine = XThinkEngine(":memory:")
try:
    thoughts = engine.think(
        "自由意味着选择。选择导致风险。风险限制安全。什么是自由？",
        cycles=5,
    )
    for t in thoughts:
        print(f"[{t.cycle}] {t.content} (score={t.confidence:.3f})")
finally:
    engine.close()
