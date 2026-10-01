from __future__ import annotations

import hashlib
import math
import time
from itertools import combinations

from xithink.models import Experience, PersonalSymbol
from xithink.modules.memory import MemoryStore


class PersonalSymbolSystem:
    """Build experience-grounded private symbols from recurring concept clusters.

    A personal symbol is an attention/representation primitive. It may join the
    thought activation graph but is not a fact and cannot independently prove a
    proposition.
    """

    def __init__(self, store: MemoryStore, *, min_exposures: int = 2) -> None:
        self.store = store
        self.min_exposures = max(2, min_exposures)
        self._symbols: dict[str, PersonalSymbol] = {s.id: s for s in store.load_personal_symbols()}

    @staticmethod
    def _clean(concepts: list[str]) -> list[str]:
        return list(dict.fromkeys(x.strip() for x in concepts if x and x.strip()))[:10]

    @staticmethod
    def _id(members: tuple[str, ...]) -> str:
        key = "\x1f".join(sorted(members))
        return "psym:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:14]

    def observe(self, experience: Experience) -> list[PersonalSymbol]:
        concepts = self._clean(experience.concepts)
        env = str(experience.context.get("environment") or "unspecified")
        changed: list[PersonalSymbol] = []
        # Pair symbols are deliberately conservative and inspectable in 0.5.
        for pair in combinations(sorted(concepts), 2):
            sid = self._id(pair)
            previous = self._symbols.get(sid)
            exposures = (previous.exposures if previous else 0) + 1
            strength = 1.0 - math.exp(-0.42 * exposures)
            old_valence = previous.valence if previous else 0.0
            episode_valence = float(experience.context.get("affect_valence", 0.0))
            valence = max(-1.0, min(1.0, old_valence * 0.75 + episode_valence * 0.25))
            environments = list(previous.environments) if previous else []
            if env not in environments:
                environments.append(env)
            symbol = PersonalSymbol(
                id=sid,
                name=f"经验符号[{pair[0]}·{pair[1]}]",
                members=list(pair),
                strength=strength,
                valence=valence,
                exposures=exposures,
                environments=sorted(environments),
                created_at=previous.created_at if previous else time.time(),
                updated_at=time.time(),
            )
            self._symbols[sid] = symbol
            changed.append(symbol)
        self.store.save_personal_symbols(changed)
        return changed

    def activate(self, cues: list[str], *, environment: str | None = None, limit: int = 12) -> tuple[dict[str, float], list[PersonalSymbol]]:
        wanted = set(self._clean(cues))
        activation: dict[str, float] = {}
        active_symbols: list[PersonalSymbol] = []
        for symbol in self._symbols.values():
            if symbol.exposures < self.min_exposures:
                continue
            if environment and symbol.environments and environment not in symbol.environments and "unspecified" not in symbol.environments:
                continue
            overlap = wanted.intersection(symbol.members)
            if not overlap:
                continue
            active_symbols.append(symbol)
            # The private symbol itself can enter the thought vocabulary.
            activation[symbol.name] = max(activation.get(symbol.name, 0.0), min(0.62, symbol.strength * 0.55))
            for member in symbol.members:
                if member not in wanted:
                    activation[member] = max(activation.get(member, 0.0), min(0.52, symbol.strength * 0.45))
        active_symbols.sort(key=lambda s: (s.strength, s.exposures), reverse=True)
        activation = dict(sorted(activation.items(), key=lambda kv: kv[1], reverse=True)[:limit])
        return activation, active_symbols[:limit]

    def all(self) -> list[PersonalSymbol]:
        return sorted(self._symbols.values(), key=lambda s: (s.strength, s.exposures), reverse=True)

    def snapshot(self, limit: int = 20) -> dict:
        return {
            "symbols": [
                {
                    "id": s.id, "name": s.name, "members": s.members,
                    "strength": s.strength, "valence": s.valence,
                    "exposures": s.exposures, "environments": s.environments,
                }
                for s in self.all()[:limit]
            ],
            "authority": "thought-vocabulary-not-truth",
        }
