from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from xithink.models import (
    ActiveHypothesisTest, Atom, Concept, Event, Experience, GraphEdge, KnowledgeClaim, LearnedRuleCandidate, EvolvedRuleVariant,
    PersonalSymbol, Rule, RuleGenealogyNode, SubconsciousTrace, Thought, WorldviewBelief,
)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS thoughts (
                id TEXT PRIMARY KEY,
                cycle INTEGER NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                concepts_json TEXT NOT NULL,
                parents_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_thoughts_created ON thoughts(created_at DESC);
            CREATE TABLE IF NOT EXISTS state_kv (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                negated INTEGER NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY(subject, predicate, object, negated)
            );
            CREATE TABLE IF NOT EXISTS rules (
                name TEXT NOT NULL,
                rule_json TEXT NOT NULL,
                PRIMARY KEY(name, rule_json)
            );
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                attributes_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                object TEXT,
                time_ref TEXT,
                negated INTEGER NOT NULL,
                modality TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);
            CREATE TABLE IF NOT EXISTS concept_edges (
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                polarity INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                weight REAL NOT NULL,
                evidence_count INTEGER NOT NULL,
                PRIMARY KEY(source, relation, target, polarity, source_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON concept_edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON concept_edges(target);
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                item_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                concepts_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_created ON memory_embeddings(created_at DESC);
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                concepts_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                outcome TEXT,
                reward REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at DESC);
            CREATE TABLE IF NOT EXISTS subconscious_traces (
                cue TEXT NOT NULL,
                association TEXT NOT NULL,
                strength REAL NOT NULL,
                valence REAL NOT NULL,
                exposures INTEGER NOT NULL,
                source TEXT NOT NULL,
                last_activated REAL NOT NULL,
                PRIMARY KEY(cue, association)
            );
            CREATE INDEX IF NOT EXISTS idx_subconscious_cue ON subconscious_traces(cue);
            CREATE TABLE IF NOT EXISTS external_knowledge_claims (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                negated INTEGER NOT NULL,
                confidence REAL NOT NULL,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_status ON external_knowledge_claims(status);
            CREATE TABLE IF NOT EXISTS learned_rule_candidates (
                signature TEXT PRIMARY KEY,
                premises_json TEXT NOT NULL,
                conclusion_json TEXT NOT NULL,
                support INTEGER NOT NULL,
                counterexamples INTEGER NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                environments_json TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                counterexample_ids_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_learned_rules_status ON learned_rule_candidates(status);
            CREATE TABLE IF NOT EXISTS evolved_rule_variants (
                id TEXT PRIMARY KEY,
                base_signature TEXT NOT NULL,
                premises_json TEXT NOT NULL,
                conclusion_json TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                support INTEGER NOT NULL,
                counterexamples INTEGER NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                environments_json TEXT NOT NULL,
                supporting_ids_json TEXT NOT NULL,
                opposing_ids_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evolved_rules_status ON evolved_rule_variants(status);
            CREATE INDEX IF NOT EXISTS idx_evolved_rules_base ON evolved_rule_variants(base_signature);
            CREATE TABLE IF NOT EXISTS rule_genealogy_nodes (
                id TEXT PRIMARY KEY,
                rule_ref TEXT NOT NULL,
                rule_kind TEXT NOT NULL,
                base_signature TEXT NOT NULL,
                parent_ids_json TEXT NOT NULL,
                generation INTEGER NOT NULL,
                mutation TEXT NOT NULL,
                status TEXT NOT NULL,
                premises_json TEXT NOT NULL,
                conclusion_json TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                support INTEGER NOT NULL,
                counterexamples INTEGER NOT NULL,
                confidence REAL NOT NULL,
                environments_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_genealogy_rule_ref ON rule_genealogy_nodes(rule_ref, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_genealogy_base ON rule_genealogy_nodes(base_signature, generation);
            CREATE TABLE IF NOT EXISTS active_hypothesis_tests (
                id TEXT PRIMARY KEY,
                base_signature TEXT NOT NULL,
                hypothesis_a TEXT NOT NULL,
                hypothesis_b TEXT NOT NULL,
                target TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                question TEXT NOT NULL,
                expected_information_gain REAL NOT NULL,
                priority REAL NOT NULL,
                environment TEXT,
                status TEXT NOT NULL,
                outcome TEXT,
                resolved_experience_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_active_tests_status ON active_hypothesis_tests(status, priority DESC);
            CREATE TABLE IF NOT EXISTS personal_symbols (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                members_json TEXT NOT NULL,
                strength REAL NOT NULL,
                valence REAL NOT NULL,
                exposures INTEGER NOT NULL,
                environments_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_personal_symbols_strength ON personal_symbols(strength DESC);
            CREATE TABLE IF NOT EXISTS worldview_beliefs (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                negated INTEGER NOT NULL,
                environment TEXT NOT NULL,
                support REAL NOT NULL,
                exposures INTEGER NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY(subject, predicate, object, negated, environment)
            );
            CREATE INDEX IF NOT EXISTS idx_worldview_environment ON worldview_beliefs(environment);
            """
        )
        self.conn.commit()

    def add_thought(self, thought: Thought) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO thoughts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                thought.id, thought.cycle, thought.kind, thought.content, thought.confidence,
                json.dumps(thought.concepts, ensure_ascii=False),
                json.dumps(thought.parents, ensure_ascii=False), thought.created_at,
            ),
        )
        self.conn.commit()

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM thoughts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def search_by_concepts(self, concepts: list[str], limit: int = 20) -> list[dict]:
        if not concepts:
            return []
        rows = self.conn.execute(
            "SELECT * FROM thoughts ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        scored = []
        wanted = set(concepts)
        for row in rows:
            item = self._row(row)
            overlap = len(wanted.intersection(item["concepts"]))
            if overlap:
                scored.append((overlap, item))
        scored.sort(key=lambda x: (x[0], x[1]["created_at"]), reverse=True)
        return [item for _, item in scored[:limit]]

    def contains_content(self, content: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM thoughts WHERE content = ? LIMIT 1", (content,)
        ).fetchone()
        return row is not None

    def save_fact(self, atom: Atom) -> None:
        self.conn.execute(
            """
            INSERT INTO facts(subject, predicate, object, negated, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject, predicate, object, negated) DO UPDATE SET
                confidence=MAX(confidence, excluded.confidence),
                source=CASE WHEN excluded.confidence >= confidence THEN excluded.source ELSE source END
            """,
            (atom.subject, atom.predicate, atom.object, int(atom.negated), atom.confidence, atom.source),
        )
        self.conn.commit()

    def load_facts(self) -> list[Atom]:
        rows = self.conn.execute(
            "SELECT subject, predicate, object, negated, confidence, source FROM facts"
        ).fetchall()
        return [
            Atom(r["subject"], r["predicate"], r["object"], bool(r["negated"]), r["confidence"], r["source"])
            for r in rows
        ]

    def save_rule(self, rule: Rule) -> None:
        payload = {
            "premises": [
                {
                    "subject": a.subject, "predicate": a.predicate, "object": a.object,
                    "negated": a.negated, "confidence": a.confidence, "source": a.source,
                }
                for a in rule.premises
            ],
            "conclusion": {
                "subject": rule.conclusion.subject, "predicate": rule.conclusion.predicate,
                "object": rule.conclusion.object, "negated": rule.conclusion.negated,
                "confidence": rule.conclusion.confidence, "source": rule.conclusion.source,
            },
            "confidence": rule.confidence,
        }
        self.conn.execute(
            "INSERT OR IGNORE INTO rules(name, rule_json) VALUES (?, ?)",
            (rule.name, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self.conn.commit()

    def load_rules(self) -> list[Rule]:
        rows = self.conn.execute("SELECT name, rule_json FROM rules").fetchall()
        out: list[Rule] = []
        for row in rows:
            payload = json.loads(row["rule_json"])

            def atom(data: dict) -> Atom:
                return Atom(
                    data["subject"], data["predicate"], data["object"],
                    bool(data.get("negated", False)), float(data.get("confidence", 1.0)),
                    data.get("source", "input"),
                )

            out.append(Rule(
                tuple(atom(x) for x in payload["premises"]),
                atom(payload["conclusion"]),
                name=row["name"],
                confidence=float(payload.get("confidence", 0.9)),
            ))
        return out

    def save_concept(self, concept: Concept) -> None:
        self.conn.execute(
            """
            INSERT INTO concepts(id, canonical_name, kind, aliases_json, attributes_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                canonical_name=excluded.canonical_name,
                kind=excluded.kind,
                aliases_json=excluded.aliases_json,
                attributes_json=excluded.attributes_json
            """,
            (
                concept.id,
                concept.canonical_name,
                concept.kind,
                json.dumps(concept.aliases, ensure_ascii=False),
                json.dumps(concept.attributes, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.conn.commit()

    def load_concepts(self) -> list[Concept]:
        rows = self.conn.execute(
            "SELECT id, canonical_name, kind, aliases_json, attributes_json FROM concepts"
        ).fetchall()
        return [
            Concept(
                id=r["id"],
                canonical_name=r["canonical_name"],
                kind=r["kind"],
                aliases=json.loads(r["aliases_json"]),
                attributes=json.loads(r["attributes_json"]),
            )
            for r in rows
        ]

    def add_event(self, event: Event) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO events(
                id, actor, action, object, time_ref, negated, modality, source, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id, event.actor, event.action, event.object, event.time_ref,
                int(event.negated), event.modality, event.source, event.confidence, event.created_at,
            ),
        )
        self.conn.commit()

    def recent_events(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]



    def add_experience(self, experience: Experience) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiences(
                id, kind, content, concepts_json, context_json, outcome, reward,
                confidence, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experience.id, experience.kind, experience.content,
                json.dumps(experience.concepts, ensure_ascii=False),
                json.dumps(experience.context, ensure_ascii=False, sort_keys=True),
                experience.outcome, experience.reward, experience.confidence,
                experience.source, experience.created_at,
            ),
        )
        self.conn.commit()

    def recent_experiences(self, limit: int = 20) -> list[Experience]:
        rows = self.conn.execute(
            "SELECT * FROM experiences ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Experience(
                id=r["id"], kind=r["kind"], content=r["content"],
                concepts=json.loads(r["concepts_json"]), context=json.loads(r["context_json"]),
                outcome=r["outcome"], reward=float(r["reward"]), confidence=float(r["confidence"]),
                source=r["source"], created_at=float(r["created_at"]),
            )
            for r in rows
        ]

    def experience_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    def save_subconscious_trace(self, trace: SubconsciousTrace) -> None:
        self.conn.execute(
            """
            INSERT INTO subconscious_traces(
                cue, association, strength, valence, exposures, source, last_activated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cue, association) DO UPDATE SET
                strength=excluded.strength, valence=excluded.valence,
                exposures=excluded.exposures, source=excluded.source,
                last_activated=excluded.last_activated
            """,
            (
                trace.cue, trace.association, trace.strength, trace.valence,
                trace.exposures, trace.source, trace.last_activated,
            ),
        )
        self.conn.commit()

    def save_subconscious_traces(self, traces: list[SubconsciousTrace]) -> None:
        if not traces:
            return
        self.conn.executemany(
            """
            INSERT INTO subconscious_traces(
                cue, association, strength, valence, exposures, source, last_activated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cue, association) DO UPDATE SET
                strength=excluded.strength, valence=excluded.valence,
                exposures=excluded.exposures, source=excluded.source,
                last_activated=excluded.last_activated
            """,
            [
                (t.cue, t.association, t.strength, t.valence, t.exposures, t.source, t.last_activated)
                for t in traces
            ],
        )
        self.conn.commit()

    def load_subconscious_traces(self, cue: str | None = None) -> list[SubconsciousTrace]:
        if cue is None:
            rows = self.conn.execute("SELECT * FROM subconscious_traces").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM subconscious_traces WHERE cue = ?", (cue,)
            ).fetchall()
        return [
            SubconsciousTrace(
                cue=r["cue"], association=r["association"], strength=float(r["strength"]),
                valence=float(r["valence"]), exposures=int(r["exposures"]),
                source=r["source"], last_activated=float(r["last_activated"]),
            )
            for r in rows
        ]

    def subconscious_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM subconscious_traces").fetchone()[0])

    def save_knowledge_claim(self, claim: KnowledgeClaim) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO external_knowledge_claims(
                id, query, subject, predicate, object, negated, confidence, provider, status,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id, claim.query, claim.subject, claim.predicate, claim.object,
                int(claim.negated), claim.confidence, claim.provider, claim.status,
                json.dumps(claim.metadata, ensure_ascii=False, sort_keys=True), claim.created_at,
            ),
        )
        self.conn.commit()

    def load_knowledge_claim(self, claim_id: str) -> KnowledgeClaim | None:
        r = self.conn.execute(
            "SELECT * FROM external_knowledge_claims WHERE id = ?", (claim_id,)
        ).fetchone()
        if not r:
            return None
        return KnowledgeClaim(
            id=r["id"], query=r["query"], subject=r["subject"], predicate=r["predicate"],
            object=r["object"], negated=bool(r["negated"]), confidence=float(r["confidence"]),
            provider=r["provider"], status=r["status"], metadata=json.loads(r["metadata_json"]),
            created_at=float(r["created_at"]),
        )

    def list_knowledge_claims(self, status: str | None = None, limit: int = 50) -> list[KnowledgeClaim]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM external_knowledge_claims ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM external_knowledge_claims WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append(KnowledgeClaim(
                id=r["id"], query=r["query"], subject=r["subject"], predicate=r["predicate"],
                object=r["object"], negated=bool(r["negated"]), confidence=float(r["confidence"]),
                provider=r["provider"], status=r["status"], metadata=json.loads(r["metadata_json"]),
                created_at=float(r["created_at"]),
            ))
        return out

    def update_knowledge_claim_status(self, claim_id: str, status: str) -> bool:
        cur = self.conn.execute(
            "UPDATE external_knowledge_claims SET status = ? WHERE id = ?", (status, claim_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def knowledge_claim_count(self, status: str | None = None) -> int:
        if status is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM external_knowledge_claims").fetchone()[0])
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM external_knowledge_claims WHERE status = ?", (status,)
        ).fetchone()[0])

    def save_graph_edge(self, edge: GraphEdge) -> None:
        self.conn.execute(
            """
            INSERT INTO concept_edges(source, relation, target, polarity, source_type, weight, evidence_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, relation, target, polarity, source_type) DO UPDATE SET
                weight=MAX(weight, excluded.weight),
                evidence_count=concept_edges.evidence_count + excluded.evidence_count
            """,
            (edge.source, edge.relation, edge.target, edge.polarity, edge.source_type, edge.weight, edge.evidence_count),
        )
        self.conn.commit()

    def load_graph_edges(self) -> list[GraphEdge]:
        rows = self.conn.execute(
            "SELECT source, relation, target, polarity, source_type, weight, evidence_count FROM concept_edges"
        ).fetchall()
        return [
            GraphEdge(
                source=r["source"], relation=r["relation"], target=r["target"],
                polarity=int(r["polarity"]), source_type=r["source_type"],
                weight=float(r["weight"]), evidence_count=int(r["evidence_count"]),
            )
            for r in rows
        ]

    def save_embedding(
        self, item_id: str, kind: str, text: str, vector: dict[int, float] | list[float],
        concepts: list[str], metadata: dict, created_at: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO memory_embeddings(item_id, kind, text, vector_json, concepts_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                kind=excluded.kind, text=excluded.text, vector_json=excluded.vector_json,
                concepts_json=excluded.concepts_json, metadata_json=excluded.metadata_json,
                created_at=excluded.created_at
            """,
            (
                item_id, kind, text, json.dumps(vector),
                json.dumps(concepts, ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False, sort_keys=True), created_at,
            ),
        )
        self.conn.commit()

    def load_embeddings(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM memory_embeddings ORDER BY created_at DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "item_id": r["item_id"], "kind": r["kind"], "text": r["text"],
                "vector": (lambda v: {int(k): float(x) for k, x in v.items()} if isinstance(v, dict) else v)(json.loads(r["vector_json"])),
                "concepts": json.loads(r["concepts_json"]),
                "metadata": json.loads(r["metadata_json"]),
                "created_at": float(r["created_at"]),
            }
            for r in rows
        ]

    def embedding_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()
        return int(row[0])

    def graph_edge_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM concept_edges").fetchone()
        return int(row[0])

    def save_json(self, key: str, value: dict) -> None:
        self.conn.execute(
            "INSERT INTO state_kv(key, value_json) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def load_json(self, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT value_json FROM state_kv WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "cycle": row["cycle"], "kind": row["kind"],
            "content": row["content"], "confidence": row["confidence"],
            "concepts": json.loads(row["concepts_json"]),
            "parents": json.loads(row["parents_json"]), "created_at": row["created_at"],
        }


    @staticmethod
    def _atom_to_dict(atom: Atom) -> dict:
        return {
            "subject": atom.subject, "predicate": atom.predicate, "object": atom.object,
            "negated": atom.negated, "confidence": atom.confidence, "source": atom.source,
        }

    @staticmethod
    def _atom_from_dict(data: dict) -> Atom:
        return Atom(
            data["subject"], data["predicate"], data["object"],
            bool(data.get("negated", False)), float(data.get("confidence", 1.0)),
            data.get("source", "experience"),
        )

    def save_learned_rule_candidate(self, candidate: LearnedRuleCandidate) -> None:
        self.conn.execute(
            """
            INSERT INTO learned_rule_candidates(
                signature, premises_json, conclusion_json, support, counterexamples,
                confidence, status, environments_json, evidence_ids_json,
                counterexample_ids_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signature) DO UPDATE SET
                premises_json=excluded.premises_json,
                conclusion_json=excluded.conclusion_json,
                support=excluded.support,
                counterexamples=excluded.counterexamples,
                confidence=excluded.confidence,
                status=excluded.status,
                environments_json=excluded.environments_json,
                evidence_ids_json=excluded.evidence_ids_json,
                counterexample_ids_json=excluded.counterexample_ids_json,
                updated_at=excluded.updated_at
            """,
            (
                candidate.signature,
                json.dumps([self._atom_to_dict(a) for a in candidate.premises], ensure_ascii=False, sort_keys=True),
                json.dumps(self._atom_to_dict(candidate.conclusion), ensure_ascii=False, sort_keys=True),
                candidate.support, candidate.counterexamples, candidate.confidence, candidate.status,
                json.dumps(candidate.environments, ensure_ascii=False),
                json.dumps(candidate.evidence_experience_ids, ensure_ascii=False),
                json.dumps(candidate.counterexample_experience_ids, ensure_ascii=False),
                candidate.updated_at,
            ),
        )
        self.conn.commit()

    def load_learned_rule_candidates(self, status: str | None = None) -> list[LearnedRuleCandidate]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM learned_rule_candidates ORDER BY confidence DESC, support DESC").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM learned_rule_candidates WHERE status = ? ORDER BY confidence DESC, support DESC",
                (status,),
            ).fetchall()
        out: list[LearnedRuleCandidate] = []
        for r in rows:
            out.append(LearnedRuleCandidate(
                signature=r["signature"],
                premises=[self._atom_from_dict(x) for x in json.loads(r["premises_json"])],
                conclusion=self._atom_from_dict(json.loads(r["conclusion_json"])),
                support=int(r["support"]),
                counterexamples=int(r["counterexamples"]),
                confidence=float(r["confidence"]),
                status=r["status"],
                environments=json.loads(r["environments_json"]),
                evidence_experience_ids=json.loads(r["evidence_ids_json"]),
                counterexample_experience_ids=json.loads(r["counterexample_ids_json"]),
                updated_at=float(r["updated_at"]),
            ))
        return out

    @staticmethod
    def _atom_payload(atom: Atom) -> dict:
        return {
            "subject": atom.subject, "predicate": atom.predicate, "object": atom.object,
            "negated": atom.negated, "confidence": atom.confidence, "source": atom.source,
        }

    @staticmethod
    def _atom_from_payload(data: dict) -> Atom:
        return Atom(
            str(data["subject"]), str(data["predicate"]), str(data["object"]),
            bool(data.get("negated", False)), float(data.get("confidence", 1.0)),
            str(data.get("source", "experience-evolution")),
        )

    def save_evolved_rule_variant(self, variant: EvolvedRuleVariant) -> None:
        self.conn.execute(
            """
            INSERT INTO evolved_rule_variants(
                id, base_signature, premises_json, conclusion_json, conditions_json,
                support, counterexamples, confidence, status, environments_json,
                supporting_ids_json, opposing_ids_json, rationale, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                base_signature=excluded.base_signature, premises_json=excluded.premises_json,
                conclusion_json=excluded.conclusion_json, conditions_json=excluded.conditions_json,
                support=excluded.support, counterexamples=excluded.counterexamples,
                confidence=excluded.confidence, status=excluded.status,
                environments_json=excluded.environments_json,
                supporting_ids_json=excluded.supporting_ids_json,
                opposing_ids_json=excluded.opposing_ids_json,
                rationale=excluded.rationale, updated_at=excluded.updated_at
            """,
            (
                variant.id, variant.base_signature,
                json.dumps([self._atom_payload(a) for a in variant.premises], ensure_ascii=False, sort_keys=True),
                json.dumps(self._atom_payload(variant.conclusion), ensure_ascii=False, sort_keys=True),
                json.dumps([self._atom_payload(a) for a in variant.conditions], ensure_ascii=False, sort_keys=True),
                variant.support, variant.counterexamples, variant.confidence, variant.status,
                json.dumps(variant.environments, ensure_ascii=False),
                json.dumps(variant.supporting_experience_ids, ensure_ascii=False),
                json.dumps(variant.opposing_experience_ids, ensure_ascii=False),
                variant.rationale, variant.updated_at,
            ),
        )
        self.conn.commit()

    def load_evolved_rule_variants(self, status: str | None = None) -> list[EvolvedRuleVariant]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM evolved_rule_variants ORDER BY confidence DESC, support DESC, id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM evolved_rule_variants WHERE status = ? ORDER BY confidence DESC, support DESC, id",
                (status,),
            ).fetchall()
        out: list[EvolvedRuleVariant] = []
        for r in rows:
            out.append(EvolvedRuleVariant(
                id=r["id"], base_signature=r["base_signature"],
                premises=[self._atom_from_payload(x) for x in json.loads(r["premises_json"])],
                conclusion=self._atom_from_payload(json.loads(r["conclusion_json"])),
                conditions=[self._atom_from_payload(x) for x in json.loads(r["conditions_json"])],
                support=int(r["support"]), counterexamples=int(r["counterexamples"]),
                confidence=float(r["confidence"]), status=r["status"],
                environments=json.loads(r["environments_json"]),
                supporting_experience_ids=json.loads(r["supporting_ids_json"]),
                opposing_experience_ids=json.loads(r["opposing_ids_json"]),
                rationale=r["rationale"], updated_at=float(r["updated_at"]),
            ))
        return out

    def save_rule_genealogy_node(self, node: RuleGenealogyNode) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO rule_genealogy_nodes(
                id, rule_ref, rule_kind, base_signature, parent_ids_json, generation, mutation, status,
                premises_json, conclusion_json, conditions_json, support, counterexamples, confidence,
                environments_json, rationale, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id, node.rule_ref, node.rule_kind, node.base_signature,
                json.dumps(node.parent_ids, ensure_ascii=False), node.generation, node.mutation, node.status,
                json.dumps([self._atom_payload(a) for a in node.premises], ensure_ascii=False, sort_keys=True),
                json.dumps(self._atom_payload(node.conclusion), ensure_ascii=False, sort_keys=True) if node.conclusion else "{}",
                json.dumps([self._atom_payload(a) for a in node.conditions], ensure_ascii=False, sort_keys=True),
                node.support, node.counterexamples, node.confidence,
                json.dumps(node.environments, ensure_ascii=False), node.rationale, node.created_at, node.updated_at,
            ),
        )
        self.conn.commit()

    def load_rule_genealogy_nodes(self, base_signature: str | None = None) -> list[RuleGenealogyNode]:
        if base_signature is None:
            rows = self.conn.execute(
                "SELECT * FROM rule_genealogy_nodes ORDER BY generation, created_at, id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM rule_genealogy_nodes WHERE base_signature = ? ORDER BY generation, created_at, id",
                (base_signature,),
            ).fetchall()
        out: list[RuleGenealogyNode] = []
        for r in rows:
            conclusion_data = json.loads(r["conclusion_json"])
            out.append(RuleGenealogyNode(
                id=r["id"], rule_ref=r["rule_ref"], rule_kind=r["rule_kind"],
                base_signature=r["base_signature"], parent_ids=json.loads(r["parent_ids_json"]),
                generation=int(r["generation"]), mutation=r["mutation"], status=r["status"],
                premises=[self._atom_from_payload(x) for x in json.loads(r["premises_json"])],
                conclusion=self._atom_from_payload(conclusion_data) if conclusion_data else None,
                conditions=[self._atom_from_payload(x) for x in json.loads(r["conditions_json"])],
                support=int(r["support"]), counterexamples=int(r["counterexamples"]),
                confidence=float(r["confidence"]), environments=json.loads(r["environments_json"]),
                rationale=r["rationale"], created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
            ))
        return out

    def latest_rule_genealogy_node(self, rule_ref: str) -> RuleGenealogyNode | None:
        row = self.conn.execute(
            "SELECT id FROM rule_genealogy_nodes WHERE rule_ref = ? ORDER BY updated_at DESC, generation DESC LIMIT 1",
            (rule_ref,),
        ).fetchone()
        if row is None:
            return None
        nodes = [n for n in self.load_rule_genealogy_nodes() if n.id == row["id"]]
        return nodes[0] if nodes else None

    def save_active_hypothesis_test(self, test: ActiveHypothesisTest) -> None:
        self.conn.execute(
            """
            INSERT INTO active_hypothesis_tests(
                id, base_signature, hypothesis_a, hypothesis_b, target, conditions_json, question,
                expected_information_gain, priority, environment, status, outcome, resolved_experience_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                hypothesis_a=excluded.hypothesis_a, hypothesis_b=excluded.hypothesis_b, target=excluded.target,
                conditions_json=excluded.conditions_json, question=excluded.question,
                expected_information_gain=excluded.expected_information_gain, priority=excluded.priority,
                environment=excluded.environment, status=excluded.status, outcome=excluded.outcome,
                resolved_experience_id=excluded.resolved_experience_id, updated_at=excluded.updated_at
            """,
            (
                test.id, test.base_signature, test.hypothesis_a, test.hypothesis_b, test.target,
                json.dumps(test.discriminating_conditions, ensure_ascii=False), test.question,
                test.expected_information_gain, test.priority, test.environment, test.status, test.outcome,
                test.resolved_experience_id, test.created_at, test.updated_at,
            ),
        )
        self.conn.commit()

    def load_active_hypothesis_tests(self, status: str | None = None) -> list[ActiveHypothesisTest]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM active_hypothesis_tests ORDER BY priority DESC, created_at DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM active_hypothesis_tests WHERE status = ? ORDER BY priority DESC, created_at DESC",
                (status,),
            ).fetchall()
        return [ActiveHypothesisTest(
            id=r["id"], base_signature=r["base_signature"], hypothesis_a=r["hypothesis_a"],
            hypothesis_b=r["hypothesis_b"], target=r["target"],
            discriminating_conditions=json.loads(r["conditions_json"]), question=r["question"],
            expected_information_gain=float(r["expected_information_gain"]), priority=float(r["priority"]),
            environment=r["environment"], status=r["status"], outcome=r["outcome"],
            resolved_experience_id=r["resolved_experience_id"], created_at=float(r["created_at"]),
            updated_at=float(r["updated_at"]),
        ) for r in rows]

    def experience_by_id(self, experience_id: str) -> Experience | None:
        row = self.conn.execute(
            "SELECT * FROM experiences WHERE id = ?", (experience_id,)
        ).fetchone()
        if row is None:
            return None
        return Experience(
            id=row["id"], kind=row["kind"], content=row["content"],
            concepts=json.loads(row["concepts_json"]), context=json.loads(row["context_json"]),
            outcome=row["outcome"], reward=float(row["reward"]), confidence=float(row["confidence"]),
            source=row["source"], created_at=float(row["created_at"]),
        )

    def save_personal_symbol(self, symbol: PersonalSymbol) -> None:
        self.conn.execute(
            """
            INSERT INTO personal_symbols(
                id, name, members_json, strength, valence, exposures,
                environments_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, members_json=excluded.members_json,
                strength=excluded.strength, valence=excluded.valence,
                exposures=excluded.exposures, environments_json=excluded.environments_json,
                updated_at=excluded.updated_at
            """,
            (
                symbol.id, symbol.name, json.dumps(symbol.members, ensure_ascii=False),
                symbol.strength, symbol.valence, symbol.exposures,
                json.dumps(symbol.environments, ensure_ascii=False),
                symbol.created_at, symbol.updated_at,
            ),
        )
        self.conn.commit()

    def save_personal_symbols(self, symbols: list[PersonalSymbol]) -> None:
        if not symbols:
            return
        self.conn.executemany(
            """
            INSERT INTO personal_symbols(
                id, name, members_json, strength, valence, exposures,
                environments_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, members_json=excluded.members_json,
                strength=excluded.strength, valence=excluded.valence,
                exposures=excluded.exposures, environments_json=excluded.environments_json,
                updated_at=excluded.updated_at
            """,
            [
                (s.id, s.name, json.dumps(s.members, ensure_ascii=False), s.strength,
                 s.valence, s.exposures, json.dumps(s.environments, ensure_ascii=False),
                 s.created_at, s.updated_at)
                for s in symbols
            ],
        )
        self.conn.commit()

    def load_personal_symbols(self) -> list[PersonalSymbol]:
        rows = self.conn.execute(
            "SELECT * FROM personal_symbols ORDER BY strength DESC, exposures DESC, name"
        ).fetchall()
        return [PersonalSymbol(
            id=r["id"], name=r["name"], members=json.loads(r["members_json"]),
            strength=float(r["strength"]), valence=float(r["valence"]),
            exposures=int(r["exposures"]), environments=json.loads(r["environments_json"]),
            created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
        ) for r in rows]

    def save_worldview_observations(self, observations: list[tuple[Atom, str, float]]) -> None:
        if not observations:
            return
        self.conn.executemany(
            """
            INSERT INTO worldview_beliefs(
                subject, predicate, object, negated, environment, support, exposures, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, 1, strftime('%s','now'))
            ON CONFLICT(subject, predicate, object, negated, environment) DO UPDATE SET
                support=support + excluded.support,
                exposures=exposures + 1,
                last_seen=excluded.last_seen
            """,
            [
                (a.subject, a.predicate, a.object, int(a.negated), env, float(weight))
                for a, env, weight in observations
            ],
        )
        self.conn.commit()

    def save_worldview_observation(self, atom: Atom, environment: str, support: float = 1.0) -> None:
        self.conn.execute(
            """
            INSERT INTO worldview_beliefs(
                subject, predicate, object, negated, environment, support, exposures, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, 1, strftime('%s','now'))
            ON CONFLICT(subject, predicate, object, negated, environment) DO UPDATE SET
                support=support + excluded.support,
                exposures=exposures + 1,
                last_seen=excluded.last_seen
            """,
            (atom.subject, atom.predicate, atom.object, int(atom.negated), environment, float(support)),
        )
        self.conn.commit()

    def load_worldview_beliefs(self, environment: str | None = None) -> list[WorldviewBelief]:
        if environment is None:
            rows = self.conn.execute("SELECT * FROM worldview_beliefs").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM worldview_beliefs WHERE environment = ?", (environment,)
            ).fetchall()
        raw = [dict(r) for r in rows]
        by_key = {
            (r["subject"], r["predicate"], r["object"], bool(r["negated"]), r["environment"]): r
            for r in raw
        }
        out: list[WorldviewBelief] = []
        for key, r in by_key.items():
            opp = by_key.get((key[0], key[1], key[2], not key[3], key[4]))
            support = float(r["support"])
            opposition = float(opp["support"]) if opp else 0.0
            confidence = (support + 1.0) / (support + opposition + 2.0)
            out.append(WorldviewBelief(
                atom=Atom(
                    r["subject"], r["predicate"], r["object"], bool(r["negated"]),
                    confidence, f"worldview:{r['environment']}",
                ),
                environment=r["environment"], support=support, opposition=opposition,
                exposures=int(r["exposures"]), confidence=confidence,
                last_seen=float(r["last_seen"]),
            ))
        out.sort(key=lambda x: (x.confidence, x.support, x.exposures), reverse=True)
        return out

    def worldview_count(self, environment: str | None = None) -> int:
        if environment is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM worldview_beliefs").fetchone()[0])
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM worldview_beliefs WHERE environment = ?", (environment,)
        ).fetchone()[0])

    def close(self) -> None:
        self.conn.close()
