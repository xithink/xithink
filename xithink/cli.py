from __future__ import annotations

import argparse
import json
from xithink.engine import XThinkEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="XThink MVP thought-loop runner")
    parser.add_argument("text", nargs="?", default="自由意味着选择。选择导致风险。什么是自由？")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--memory", default="xithink_memory.sqlite3")
    parser.add_argument("--environment", default=None, help="Optional experience/worldview environment")
    args = parser.parse_args()

    engine = XThinkEngine(args.memory)
    try:
        thoughts = engine.think(args.text, cycles=args.cycles, environment=args.environment)
        print("\n=== THOUGHTS ===")
        for t in thoughts:
            print(f"[{t.cycle}] {t.kind} score={t.confidence:.3f} :: {t.content}")
        print("\n=== SNAPSHOT ===")
        print(json.dumps(engine.snapshot(), ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
