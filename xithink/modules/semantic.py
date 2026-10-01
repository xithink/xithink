from __future__ import annotations

import re
from collections import OrderedDict
from xithink.models import Atom, Concept, Event, QuestionIntent, Rule, SemanticFrame


class SemanticRecognizer:
    """SemanticFrame V2 deterministic parser.

    The parser intentionally remains inspectable. It extracts facts, rules,
    questions, events, negation, source, modality and time references without
    requiring a foundation model. An LLM adapter can later replace or augment
    this module while preserving the same SemanticFrame contract.
    """

    NEGATED_RELATION = re.compile(
        r"^(.+?)不(间接导致|可能影响|意味着|代表|需要|依赖|导致|带来|造成|限制|约束|影响)(.+)$"
    )
    FACT_PATTERNS = [
        (re.compile(r"^(.+?)(间接导致)(.+)$"), "间接导致", False),
        (re.compile(r"^(.+?)(可能影响)(.+)$"), "可能影响", False),
        (re.compile(r"^(.+?)(影响)(.+)$"), "影响", False),
        (re.compile(r"^(.+?)(意味着|代表)(.+)$"), "意味着", False),
        (re.compile(r"^(.+?)(需要|依赖)(.+)$"), "需要", False),
        (re.compile(r"^(.+?)(导致|带来|造成)(.+)$"), "导致", False),
        (re.compile(r"^(.+?)(限制|约束)(.+)$"), "限制", False),
        (re.compile(r"^(.+?)不是(.+)$"), "是", True),
        (re.compile(r"^(.+?)是(.+)$"), "是", False),
    ]

    EVENT_PATTERN = re.compile(
        r"^(.{1,24}?)(不|没有)?"
        r"(喜欢|支持|反对|选择|保护|伤害|创造|改变|学习|理解|影响|拥有|寻找|认为|相信|记得|忘记|观察|研究)"
        r"(.{0,48})$"
    )

    SOURCE_PATTERNS = [
        re.compile(r"^据(.{1,32}?)(?:说|表示|报道|认为)[,，:： ]*(.+)$"),
        re.compile(r"^根据(.{1,32}?)[,，:： ]+(.+)$"),
    ]

    TIME_PATTERNS = [
        re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
        re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
        re.compile(r"(?:今天|昨天|明天|现在|当前|未来|过去|刚才|随后|之后|之前)"),
    ]

    MODALITY_MARKERS = (
        ("可能", "possible", 0.68),
        ("也许", "possible", 0.62),
        ("或许", "possible", 0.62),
        ("应该", "normative", 0.72),
        ("必须", "obligatory", 0.88),
        ("可以", "permitted", 0.78),
        ("希望", "desired", 0.70),
        ("想要", "desired", 0.70),
    )

    STOP = {
        "一个", "一种", "这个", "那个", "如果", "那么", "可能", "也许", "或许", "是否", "什么",
        "为什么", "如何", "可以", "应该", "必须", "没有", "存在", "进行", "产生", "通过", "当前",
        "今天", "昨天", "明天", "现在", "未来", "过去",
    }

    def parse(self, text: str) -> SemanticFrame:
        normalized = self._normalize(text)
        source, payload = self._extract_source(normalized)
        modality, modal_confidence = self._detect_modality(payload)
        frame = SemanticFrame(raw_text=normalized, source=source, modality=modality)
        frame.time_refs = self._extract_times(payload)

        chunks = [c.strip() for c in re.split(r"[。；;\n]+", payload) if c.strip()]
        for chunk in chunks:
            chunk_source, chunk_payload = self._extract_source(chunk, default=source)
            chunk_modality, chunk_confidence = self._detect_modality(chunk_payload)
            effective_confidence = min(modal_confidence, chunk_confidence)
            time_ref = self._extract_times(chunk_payload)
            selected_time = time_ref[0] if time_ref else (frame.time_refs[0] if frame.time_refs else None)

            if "如果" in chunk_payload and ("那么" in chunk_payload or "则" in chunk_payload):
                rule = self._parse_rule(chunk_payload, source=chunk_source, confidence=effective_confidence)
                if rule:
                    frame.rules.append(rule)
                    frame.concepts.extend(self._atom_concepts(rule.conclusion))
                    for p in rule.premises:
                        frame.concepts.extend(self._atom_concepts(p))
                    continue

            is_question = self._is_question(chunk_payload)
            cleaned = chunk_payload.rstrip("?？")
            if is_question:
                frame.questions.append(cleaned)
                intent = self._parse_question_intent(cleaned)
                if intent:
                    frame.question_intents.append(intent)
                frame.concepts.extend(self._question_concepts(cleaned))

            fact = None if is_question else self._parse_fact(
                cleaned,
                source=chunk_source,
                confidence=effective_confidence,
            )
            if fact:
                frame.facts.append(fact)
                frame.concepts.extend(self._atom_concepts(fact))

            event = None if (is_question or fact is not None) else self._parse_event(
                cleaned,
                source=chunk_source,
                modality=chunk_modality,
                time_ref=selected_time,
                confidence=effective_confidence,
            )
            if event:
                frame.events.append(event)
                frame.concepts.append(event.actor)
                if event.object:
                    frame.concepts.append(event.object)

            if not fact and not event:
                frame.concepts.extend(self._fallback_concepts(cleaned))

        frame.concepts = list(OrderedDict.fromkeys(
            self._clean_concept(c) for c in frame.concepts
            if self._clean_concept(c) and self._clean_concept(c) not in self.STOP
        ))
        frame.concept_nodes = [Concept(canonical_name=c) for c in frame.concepts]
        return frame

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).replace("，", ",")

    @staticmethod
    def _clean_concept(text: str) -> str:
        return text.strip(" ,，。.!！?？:：;；\"'")

    def _extract_source(self, text: str, default: str = "input") -> tuple[str, str]:
        for pattern in self.SOURCE_PATTERNS:
            match = pattern.match(text)
            if match:
                return self._clean_concept(match.group(1)), match.group(2).strip()
        return default, text

    def _extract_times(self, text: str) -> list[str]:
        out: list[str] = []
        for pattern in self.TIME_PATTERNS:
            out.extend(pattern.findall(text))
        return list(OrderedDict.fromkeys(out))

    def _detect_modality(self, text: str) -> tuple[str, float]:
        for marker, modality, confidence in self.MODALITY_MARKERS:
            if marker in text:
                return modality, confidence
        return "asserted", 0.90

    @staticmethod
    def _is_question(text: str) -> bool:
        if text.endswith(("?", "？")):
            return True
        return bool(re.match(r"^(什么|为什么|如何|怎么|是否|谁|哪里|何时|多少)", text))

    def _parse_rule(self, text: str, *, source: str, confidence: float) -> Rule | None:
        match = re.search(r"如果(.+?)(?:,)?(?:那么|则)(.+)", text)
        if not match:
            return None
        left, right = match.group(1).strip(), match.group(2).strip(" ,。")
        premise = self._parse_fact(left, source=source, confidence=confidence)
        conclusion = self._parse_fact(right, source=source, confidence=confidence)
        if not premise or not conclusion:
            return None
        return Rule((premise,), conclusion, name=f"NL:{text[:24]}", confidence=min(0.85, confidence))

    def _parse_fact(self, text: str, *, source: str = "input", confidence: float = 0.90) -> Atom | None:
        text = text.strip(" ,。")
        neg = self.NEGATED_RELATION.match(text)
        if neg:
            subject, predicate_raw, obj = neg.group(1).strip(), neg.group(2), neg.group(3).strip()
            predicate = {
                "代表": "意味着", "依赖": "需要", "带来": "导致", "造成": "导致", "约束": "限制"
            }.get(predicate_raw, predicate_raw)
            if subject and obj:
                return Atom(subject, predicate, obj, negated=True, confidence=confidence, source=source)

        for pattern, predicate, negated in self.FACT_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue
            subject = match.group(1).strip()
            # Relation patterns expose (subject, surface-relation, object),
            # while 是/不是 expose only (subject, object).
            obj_group = 3 if (match.lastindex or 0) >= 3 else 2
            obj = match.group(obj_group).strip()
            if subject and obj and len(subject) <= 40 and len(obj) <= 60:
                return Atom(subject, predicate, obj, negated=negated, confidence=confidence, source=source)
        return None

    def _parse_event(
        self,
        text: str,
        *,
        source: str,
        modality: str,
        time_ref: str | None,
        confidence: float,
    ) -> Event | None:
        cleaned = text
        for marker, _, _ in self.MODALITY_MARKERS:
            cleaned = cleaned.replace(marker, "", 1).strip()
        if time_ref:
            cleaned = cleaned.replace(time_ref, "", 1).strip(" ,，")
        match = self.EVENT_PATTERN.match(cleaned)
        if not match:
            return None
        actor = self._clean_concept(match.group(1))
        negated = bool(match.group(2))
        action = match.group(3)
        obj = self._clean_concept(match.group(4)) or None
        if not actor:
            return None
        return Event(
            actor=actor,
            action=action,
            object=obj,
            time_ref=time_ref,
            negated=negated,
            modality=modality,
            source=source,
            confidence=confidence,
        )

    @staticmethod
    def _atom_concepts(atom: Atom) -> list[str]:
        return [atom.subject, atom.object]

    def _parse_question_intent(self, text: str) -> QuestionIntent | None:
        cleaned = text.strip(" ?？")

        match = re.match(r"^什么是(.+)$", cleaned)
        if match:
            return QuestionIntent("definition", cleaned, subject=self._clean_concept(match.group(1)))
        match = re.match(r"^(.+?)是什么$", cleaned)
        if match:
            return QuestionIntent("definition", cleaned, subject=self._clean_concept(match.group(1)))

        match = re.match(r"^为什么(.+)$", cleaned)
        if match:
            atom = self._parse_fact(match.group(1), source="question", confidence=1.0)
            if atom:
                return QuestionIntent(
                    "why_relation", cleaned, atom.subject, atom.predicate, atom.object, atom.negated
                )

        match = re.match(r"^(.+?)(间接导致|可能影响|意味着|代表|需要|依赖|导致|带来|造成|限制|约束|影响)什么$", cleaned)
        if match:
            predicate = {
                "代表": "意味着", "依赖": "需要", "带来": "导致", "造成": "导致", "约束": "限制"
            }.get(match.group(2), match.group(2))
            return QuestionIntent("relation_object", cleaned, self._clean_concept(match.group(1)), predicate=predicate)

        if cleaned.startswith("是否"):
            atom = self._parse_fact(cleaned[2:], source="question", confidence=1.0)
            if atom:
                return QuestionIntent("yes_no", cleaned, atom.subject, atom.predicate, atom.object, atom.negated)

        # Chinese yes/no questions often place 是否 after the subject.
        if "是否" in cleaned:
            candidate = cleaned.replace("是否", "", 1)
            atom = self._parse_fact(candidate, source="question", confidence=1.0)
            if atom:
                return QuestionIntent("yes_no", cleaned, atom.subject, atom.predicate, atom.object, atom.negated)

        return QuestionIntent("open", cleaned)

    def _question_concepts(self, text: str) -> list[str]:
        cleaned = text.strip(" ?？")
        patterns = [
            r"^什么是(.+)$",
            r"^(.+?)是什么$",
            r"^为什么(.+)$",
            r"^如何(.+)$",
            r"^怎么(.+)$",
            r"^是否(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match:
                target = self._clean_concept(match.group(1))
                return self._fallback_concepts(target) or ([target] if target else [])
        return []

    def _fallback_concepts(self, text: str) -> list[str]:
        parts = re.split(r"[,，:：、\s]|而且|但是|因为|所以|或者|以及|与|和", text)
        concepts: list[str] = []
        for part in parts:
            part = self._clean_concept(part)
            if 1 < len(part) <= 20 and part not in self.STOP:
                concepts.append(part)
        return concepts
