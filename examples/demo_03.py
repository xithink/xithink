from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xithink.engine import XThinkEngine


def main() -> None:
    db = Path("data/demo_03.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    engine = XThinkEngine(db)
    try:
        inputs = [
            "自由意味着选择。选择导致风险。风险限制安全。什么是自由？",
            "选择和自由之间是什么关系？",
            "自由不间接导致风险。现在重新判断自由与风险的关系？",
        ]
        for text in inputs:
            print(f"\nUSER > {text}")
            for thought in engine.think(text, cycles=4):
                print(f"X[{thought.cycle}] {thought.confidence:.3f} > {thought.content}")
            print("RECALL >", [
                (m["kind"], round(m["score"], 3), m["text"])
                for m in engine.last_trace.get("recall", [])[:3]
            ])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
