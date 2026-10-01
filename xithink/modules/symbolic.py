from __future__ import annotations

from dataclasses import replace

from xithink.models import Atom, QuestionIntent, Rule, SymbolicAnswer, SymbolicProof


class SymbolicReasoner:
    def __init__(self) -> None:
        self._facts: dict[tuple[str, str, str, bool], Atom] = {}
        self._fact_index: dict[tuple[str, bool], set[tuple[str, str, str, bool]]] = {}
        self._rules: list[Rule] = []
        self._proofs: dict[tuple[str, str, str, bool], SymbolicProof] = {}

    @property
    def facts(self) -> list[Atom]:
        return list(self._facts.values())

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    def add_fact(self, atom: Atom, proof: SymbolicProof | None = None) -> bool:
        old = self._facts.get(atom.key())
        if old is None or atom.confidence > old.confidence:
            is_new = old is None
            self._facts[atom.key()] = atom
            if is_new:
                self._fact_index.setdefault((atom.predicate, atom.negated), set()).add(atom.key())
            self._proofs[atom.key()] = proof or SymbolicProof(
                conclusion=atom,
                kind="fact",
                confidence=atom.confidence,
                source=atom.source,
            )
            return True
        return False

    def add_rule(self, rule: Rule) -> None:
        if rule not in self._rules:
            self._rules.append(rule)

    def remove_rules_by_prefix(self, *prefixes: str) -> None:
        if not prefixes:
            return
        self._rules = [r for r in self._rules if not any(r.name.startswith(p) for p in prefixes)]

    def conflicts(self) -> list[tuple[Atom, Atom]]:
        found = []
        seen = set()
        for fact in self._facts.values():
            opposite = self._facts.get(fact.opposite_key())
            if opposite and fact.key() not in seen and opposite.key() not in seen:
                found.append((fact, opposite))
                seen.add(fact.key())
                seen.add(opposite.key())
        return found

    def infer(self, max_rounds: int = 5) -> list[Atom]:
        inferred: list[Atom] = []
        for _ in range(max_rounds):
            round_new: list[Atom] = []
            for rule in self._rules:
                for bindings, matched in self._match_rule(rule):
                    conclusion = self._substitute(rule.conclusion, bindings)
                    confidence = rule.confidence
                    premise_proofs: list[SymbolicProof] = []
                    for atom in matched:
                        confidence *= atom.confidence
                        premise_proofs.append(self.proof_for(atom) or SymbolicProof(
                            atom, "fact", atom.confidence, atom.source
                        ))
                    conclusion = replace(
                        conclusion,
                        confidence=max(0.0, min(1.0, confidence)),
                        source=f"inference:{rule.name}",
                    )
                    proof = SymbolicProof(
                        conclusion=conclusion,
                        kind="inference",
                        confidence=conclusion.confidence,
                        source=conclusion.source,
                        rule_name=rule.name,
                        premises=premise_proofs,
                    )
                    if self.add_fact(conclusion, proof=proof):
                        round_new.append(conclusion)
            inferred.extend(round_new)
            if not round_new:
                break
        return inferred

    def proof_for(self, atom: Atom | tuple[str, str, str, bool]) -> SymbolicProof | None:
        key = atom if isinstance(atom, tuple) else atom.key()
        return self._proofs.get(key)

    def answer(self, intent: QuestionIntent) -> SymbolicAnswer:
        """Answer exclusively from the symbolic fact/rule base.

        No embedding, experience, subconscious activation or model output is
        permitted to create an answer here. Those layers may help exploration,
        but an answered=True result must have a SymbolicProof.
        """
        self.infer()
        if not intent or intent.kind == "open":
            return SymbolicAnswer(False, "当前符号知识不足，无法形成可证明答案。", intent=intent)

        atoms: list[Atom] = []
        if intent.kind == "definition" and intent.subject:
            atoms = [
                a for a in self._facts.values()
                if a.subject == intent.subject and not a.negated
            ]
        elif intent.kind == "relation_object" and intent.subject and intent.predicate:
            atoms = [
                a for a in self._facts.values()
                if a.subject == intent.subject and a.predicate == intent.predicate and not a.negated
            ]
        elif intent.kind in {"yes_no", "why_relation"} and intent.subject and intent.predicate and intent.object:
            key = (intent.subject, intent.predicate, intent.object, intent.negated)
            opposite_key = (intent.subject, intent.predicate, intent.object, not intent.negated)
            atom = self._facts.get(key)
            opposite = self._facts.get(opposite_key)
            if atom is not None and opposite is not None:
                p1 = self.proof_for(atom)
                p2 = self.proof_for(opposite)
                text = (
                    f"符号系统同时存在正反结论：{atom.render()}；{opposite.render()}。"
                    "需要继续区分适用条件、环境或证据来源。"
                )
                return SymbolicAnswer(
                    True, text, min(atom.confidence, opposite.confidence),
                    [atom, opposite], [p for p in (p1, p2) if p], intent,
                )
            if atom is not None:
                atoms = [atom]
            elif opposite is not None:
                proof = self.proof_for(opposite)
                text = f"符号系统目前支持相反结论：{opposite.render()}。"
                return SymbolicAnswer(
                    True, text, opposite.confidence, [opposite], [proof] if proof else [], intent
                )

        if intent.kind == "definition":
            priority = {"是": 3, "意味着": 2}
            atoms.sort(key=lambda a: (priority.get(a.predicate, 1), a.confidence, a.render()), reverse=True)
        else:
            atoms.sort(key=lambda a: (a.confidence, a.render()), reverse=True)
        if not atoms:
            return SymbolicAnswer(False, "当前长期符号记忆中没有足够前提形成证明。", intent=intent)

        proofs = [p for a in atoms if (p := self.proof_for(a)) is not None]
        confidence = min((a.confidence for a in atoms), default=0.0)
        if intent.kind == "definition":
            text = "；".join(a.render() for a in atoms[:4])
            text = f"根据长期符号记忆：{text}。"
        elif intent.kind == "why_relation":
            text = f"根据符号证明链，可推出：{atoms[0].render()}。"
        else:
            text = "；".join(a.render() for a in atoms[:4])
            text = f"根据符号事实/规则，可推出：{text}。"
        return SymbolicAnswer(True, text, confidence, atoms[:4], proofs[:4], intent)

    def _match_rule(self, rule: Rule):
        """Indexed backtracking matcher.

        0.4 used a Cartesian product over every predicate-compatible fact set.
        With experiential memory this becomes expensive quickly. Here each
        matched premise binds variables immediately, so later candidate sets are
        filtered by those bindings before recursion.
        """
        matches: list[tuple[dict[str, str], tuple[Atom, ...]]] = []

        def walk(index: int, bindings: dict[str, str], chosen: list[Atom]) -> None:
            if index >= len(rule.premises):
                matches.append((dict(bindings), tuple(chosen)))
                return
            pattern = rule.premises[index]
            for fact in self._candidate_facts(pattern, bindings):
                next_bindings = dict(bindings)
                if self._unify(pattern, fact, next_bindings):
                    chosen.append(fact)
                    walk(index + 1, next_bindings, chosen)
                    chosen.pop()

        walk(0, {}, [])
        return matches

    def _candidate_facts(self, pattern: Atom, bindings: dict[str, str] | None = None) -> list[Atom]:
        bindings = bindings or {}
        keys = self._fact_index.get((pattern.predicate, pattern.negated), set())

        def expected(value: str) -> str | None:
            if value.startswith("?"):
                return bindings.get(value)
            return value

        expected_subject = expected(pattern.subject)
        expected_object = expected(pattern.object)
        out: list[Atom] = []
        for key in sorted(keys):
            fact = self._facts[key]
            if expected_subject is not None and fact.subject != expected_subject:
                continue
            if expected_object is not None and fact.object != expected_object:
                continue
            out.append(fact)
        return out

    @staticmethod
    def _unify(pattern: Atom, fact: Atom, bindings: dict[str, str]) -> bool:
        for pval, fval in ((pattern.subject, fact.subject), (pattern.object, fact.object)):
            if pval.startswith("?"):
                existing = bindings.get(pval)
                if existing is not None and existing != fval:
                    return False
                bindings[pval] = fval
            elif pval != fval:
                return False
        return True

    @staticmethod
    def _substitute(atom: Atom, bindings: dict[str, str]) -> Atom:
        def sub(v: str) -> str:
            return bindings.get(v, v)
        return Atom(sub(atom.subject), atom.predicate, sub(atom.object), atom.negated, atom.confidence, atom.source)
