"""Persistent reference lookup, compatible with pilot_contract.ReferenceRepository.

No connection or filesystem access on import. Supply a service-role DSN, set
FACETELLER_DATABASE_URL, or use the private config created by setup_database.py.
Each call owns/closes its connection; DB failures propagate, never become no-match.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import psycopg

from facework.pilot_contract import Passage

LABEL_PATTERN = re.compile(r"[^\s:]+:[^\s:]+\Z")


def local_config_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return base / "FaceTeller" / "postgresql" / "connection.json"


def database_dsn(*, admin: bool = False, config: Path | None = None) -> str:
    name = "FACETELLER_ADMIN_DATABASE_URL" if admin else "FACETELLER_DATABASE_URL"
    if os.environ.get(name):
        return os.environ[name]
    path = config or local_config_path()
    if not path.is_file():
        raise RuntimeError(f"Set {name} or run scripts/release/setup_database.py local-init")
    return json.loads(path.read_text(encoding="utf-8"))["admin_dsn" if admin else "service_dsn"]


def validate_labels(labels) -> list[str]:
    if not isinstance(labels, (set, frozenset, list, tuple)):
        raise ValueError("Labels must be a collection of canonical feature:class strings")
    result = []
    for value in labels:
        if not isinstance(value, str) or not LABEL_PATTERN.fullmatch(value):
            raise ValueError("Every label must be a canonical feature:class string")
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError("Duplicate required label")
    return sorted(result)


CANDIDATES_SQL = """
SELECT p.passage_id, p.original_text, p.translation, p.commentary, p.source,
       p.literature_priority, p.passage_order,
       ARRAY(SELECT lp.label_id FROM face_release.label_passage lp
             WHERE lp.passage_id = p.passage_id ORDER BY lp.label_id) AS required_labels,
       p.annotation_status, p.document->>'display_policy'
FROM face_release.passage p
WHERE p.service_eligible
  AND EXISTS (SELECT 1 FROM face_release.label_passage hit
              WHERE hit.passage_id = p.passage_id AND hit.label_id = ANY(%s::text[]))
  AND NOT EXISTS (SELECT 1 FROM face_release.label_passage missing
                  WHERE missing.passage_id = p.passage_id
                    AND NOT (missing.label_id = ANY(%s::text[])))
ORDER BY p.literature_priority, p.passage_order, p.passage_id COLLATE "C"
"""


def _passage_order(p: Passage):
    # Match PassageSelector's numeric/string ID tie break; never coerce stored IDs.
    key = (0, int(p.passage_id)) if p.passage_id.isdecimal() else (1, p.passage_id)
    return p.literature_priority, p.passage_order, key, p.passage_id


class ReleaseRepository:
    def __init__(self, dsn: str | None = None, *, connect_timeout: int = 5):
        self._dsn = dsn if dsn is not None else database_dsn()
        self._connect_timeout = connect_timeout

    def load_candidates(self, label_ids: set[str]) -> list[Passage]:
        labels = validate_labels(label_ids)
        if not labels:
            return []
        with self.snapshot() as snapshot:
            return snapshot.load_candidates(label_ids)

    def corpus_revision(self) -> str | None:
        """Active canonical JSON SHA256, or None until a corpus is imported."""
        with self.snapshot() as snapshot:
            return snapshot.corpus_revision()

    @contextmanager
    def snapshot(self):
        """Pin candidate text and its revision to one read-only MVCC snapshot."""
        with psycopg.connect(self._dsn, connect_timeout=self._connect_timeout) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            conn.execute("SET LOCAL statement_timeout = '5s'")
            yield _CorpusSnapshot(conn)


class _CorpusSnapshot:
    def __init__(self, connection):
        self.connection=connection

    def corpus_revision(self) -> str | None:
        row=self.connection.execute("SELECT canonical_sha256 FROM face_release.active_corpus WHERE singleton=1").fetchone()
        return row[0] if row else None

    def load_candidates(self, label_ids: set[str]) -> list[Passage]:
        labels=validate_labels(label_ids)
        if not labels:
            return []
        rows=self.connection.execute(CANDIDATES_SQL,(labels,labels)).fetchall()
        # No LIMIT: final confidence ranking and region/global caps belong to selector.
        # The dedicated column is authoritative. Enrich a copy, preserving the
        # archived raw source; never let a conflicting nested tag promote review.
        return sorted([Passage(str(r[0]), frozenset(r[7]), r[1], r[2], r[3],
                               {**r[4], 'annotation_status': r[8],
                                **({'display_policy': r[9]} if r[9] is not None else {})},
                               r[5], r[6]) for r in rows], key=_passage_order)


# Explicit new-module alias for callers accustomed to the old repository class name.
PostgresRepository = ReleaseRepository
