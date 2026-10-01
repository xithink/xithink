from __future__ import annotations

import argparse
import json

from xithink.engine import XThinkEngine
from xithink.modules.life_simulation import profile_from_name


def main() -> None:
    parser = argparse.ArgumentParser(description="XThink 0.7 life experience simulation")
    parser.add_argument("--experiences", type=int, default=1000)
    parser.add_argument("--max-age", type=int, default=40)
    parser.add_argument("--profile", choices=["balanced", "conservative", "exploratory"], default="balanced")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    engine = XThinkEngine(":memory:")
    try:
        report = engine.simulate_life(
            total_experiences=args.experiences,
            max_age=args.max_age,
            age_checkpoints=list(range(0, args.max_age + 1, 10)),
            seed=args.seed,
            profile=profile_from_name(args.profile),
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
