"""Design 4.3 label/passage metadata: frozen-definition sync and additive migration004.

Unit tests read the frozen registry/landmark/depth files only. PostgreSQL tests create
and drop their own randomly named database and login (never the release database),
upgrade 001-003 -> 004 over the real final corpus, and verify the corpus is unchanged.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import unittest
from uuid import uuid4

import numpy as np
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from facework.release_repository import ReleaseRepository, database_dsn, local_config_path
from scripts.release import sync_label_metadata as meta
from scripts.release.setup_database import MIGRATIONS, canonical_json, grant_reader, import_corpus, migrate, status

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'data/release/literature/reference_corpus_v1.json'
READY = ROOT / 'data/release/literature_ready.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class FrozenMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load(meta.INPUTS['training_registry'])
        cls.landmark = load(meta.INPUTS['landmark_labels'])
        cls.depth = load(meta.INPUTS['depth_config'])
        cls.rows, cls.revision, cls.hashes = meta.load_inputs()

    def test_every_registry_class_has_complete_metadata(self):
        expected = {f'{k}:{c}' for src in (self.registry['features'], self.registry['depth_features'])
                    for k, f in src.items() for c in f['classes']}
        self.assertEqual(set(self.rows), expected)
        for label, row in self.rows.items():
            key = label.split(':')[0]
            with self.subTest(label=label):
                self.assertTrue(all(row[c] for c in ('region', 'label_name', 'criteria', 'measurement_method')))
                if key in self.registry['depth_features']:
                    self.assertEqual(row['measurement_method'], meta.DEPTH)
                    self.assertEqual(row['measurement_indicator'], f'{key} (unit: ratio)')
                    self.assertEqual(row['region'], self.registry['depth_features'][key]['part'])
                else:
                    f = self.registry['features'][key]
                    self.assertEqual(row['measurement_method'], meta.CNN)
                    self.assertIsNone(row['measurement_indicator'])
                    self.assertEqual(row['region'], f.get('region', f['part']))
                    self.assertTrue(row['label_name'].startswith(f['ko'] + ': '))

    def test_corpus_labels_are_all_covered(self):
        corpus = load(CORPUS)
        used = {l for p in corpus['passages'] for l in p['required_labels']}
        self.assertLessEqual(used, set(self.rows))

    def test_boundaries_are_exact_and_follow_runtime_side_convention(self):
        for key, f in self.registry['features'].items():
            t = f.get('thresholds') or {}
            cuts = t.get('cuts') or ([t['lower'], t['upper']] if t.get('status') == 'fitted' else None)
            if not cuts:
                continue
            for cut in cuts:
                # Value exactly at a cut belongs to the upper class (searchsorted side='right').
                index = int(np.searchsorted(cuts, cut, side='right'))
                text = self.rows[f"{key}:{f['classes'][index]}"]['criteria']
                with self.subTest(key=key, cut=cut):
                    self.assertIn(f'{repr(float(cut))} ≤', text) if index < len(cuts) else self.assertIn(f'≥ {repr(float(cut))}', text)
        for key, c in self.depth['features'].items():
            classes = self.registry['depth_features'][key]['classes']
            self.assertIn(f"r < {repr(float(c['lower']))}", self.rows[f'{key}:{classes[0]}']['criteria'])
            self.assertIn(f"r ≥ {repr(float(c['upper']))}", self.rows[f'{key}:{classes[2]}']['criteria'])

    def test_visual_consensus_labels_do_not_get_invented_physical_boundaries(self):
        visual = [k for k, f in self.registry['features'].items() if not f.get('thresholds')]
        self.assertTrue(visual)
        for key in visual:
            for cls in self.registry['features'][key]['classes']:
                text = self.rows[f'{key}:{cls}']['criteria']
                with self.subTest(label=f'{key}:{cls}'):
                    self.assertIn('수치 물리 경계 없음', text)
                    self.assertNotIn(' < ', text)
                    self.assertNotIn('≥', text)

    def test_disagreeing_or_reviewed_definitions_fail(self):
        landmark = copy.deepcopy(self.landmark)
        landmark['features']['eye_size']['cuts'][0] += 1e-6
        with self.assertRaisesRegex(ValueError, 'landmark_labels'):
            meta.build_metadata(self.registry, landmark, self.depth)
        depth = copy.deepcopy(self.depth)
        depth['features'].pop('chin_projection')
        with self.assertRaisesRegex(ValueError, 'depth feature set'):
            meta.build_metadata(self.registry, self.landmark, depth)
        registry = copy.deepcopy(self.registry)
        registry['human_review_count'] = 1
        with self.assertRaisesRegex(ValueError, 'human_review_count'):
            meta.build_metadata(registry, self.landmark, self.depth)

    def test_revision_is_deterministic_and_bound_to_inputs(self):
        self.assertEqual(meta.load_inputs()[1], self.revision)
        with tempfile.TemporaryDirectory() as directory:
            copy_path = Path(directory) / 'depth.json'
            copy_path.write_text(json.dumps(self.depth, indent=3), encoding='utf-8')  # same content, other bytes
            rows, revision, hashes = meta.load_inputs({'depth_config': copy_path})
        self.assertEqual(rows, self.rows)
        self.assertNotEqual(revision, self.revision)


@unittest.skipUnless(os.environ.get('FACETELLER_ADMIN_DATABASE_URL') or local_config_path().exists(),
                     'Actual PostgreSQL required: run setup_database.py local-init')
class MigrationAndSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin = database_dsn(admin=True)
        token = uuid4().hex[:16]
        cls.db, cls.role = 'faceteller_test_' + token, 'faceteller_test_reader_' + token
        cls.password = secrets.token_urlsafe(24)
        with psycopg.connect(cls.admin, dbname='postgres', autocommit=True) as conn:
            conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(cls.db)))
        cls.addClassCleanup(cls.cleanup_database)
        cls.dsn = make_conninfo(cls.admin, dbname=cls.db)
        # Production path: an existing 001-003 database with the final corpus, then 004.
        with tempfile.TemporaryDirectory() as directory:
            for path in sorted(MIGRATIONS.glob('00[1-3]_*.sql')):
                shutil.copyfile(path, Path(directory) / path.name)
            migrate(cls.dsn, Path(directory))
        grant_reader(cls.dsn, cls.role, cls.password)
        cls.reader = make_conninfo(cls.dsn, user=cls.role, password=cls.password)
        cls.corpus = load(CORPUS)
        cls.digest = hashlib.sha256(canonical_json(cls.corpus).encode('utf-8')).hexdigest()
        cls.imported = import_corpus(cls.dsn, cls.corpus)
        cls.before = status(cls.dsn)
        cls.before_rows = cls.passage_rows()
        cls.applied = migrate(cls.dsn)
        grant_reader(cls.dsn, cls.role)  # production re-provision after migration
        cls.rows, cls.revision, _ = meta.load_inputs()

    @classmethod
    def cleanup_database(cls):
        with psycopg.connect(cls.admin, dbname='postgres', autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(cls.db)))
            conn.execute(sql.SQL('DROP ROLE IF EXISTS {}').format(sql.Identifier(cls.role)))

    @classmethod
    def passage_rows(cls):
        with psycopg.connect(cls.dsn) as conn:
            return {'passage': conn.execute("""SELECT md5(string_agg(passage_id||document::text||source::text||
                        service_eligible::text||translation||commentary, '|' ORDER BY passage_id)) FROM face_release.passage""").fetchone()[0],
                    'mapping': conn.execute("""SELECT md5(string_agg(passage_id||'>'||label_id, '|' ORDER BY passage_id,label_id))
                        FROM face_release.label_passage""").fetchone()[0],
                    'labels': {r[0] for r in conn.execute('SELECT label_id FROM face_release.label')},
                    'eligible': {r[0] for r in conn.execute('SELECT passage_id FROM face_release.passage WHERE service_eligible')}}

    def assert_corpus_unchanged(self):
        after = status(self.dsn)
        for key in ('corpus_sha256', 'sources', 'passages', 'source_sections', 'eligible', 'imports'):
            self.assertEqual(after[key], self.before[key], key)
        rows = self.passage_rows()
        for key in ('passage', 'mapping', 'eligible'):
            self.assertEqual(rows[key], self.before_rows[key], key)
        self.assertLessEqual(self.before_rows['labels'], rows['labels'])

    def setUp(self):
        meta.sync(self.dsn, self.rows, self.revision)

    def test_upgrade_is_additive_and_keeps_final_corpus_identity(self):
        self.assertEqual(self.applied, ['004_label_passage_metadata.sql'])
        self.assertEqual(self.digest, load(READY)['corpus_sha256'])
        self.assertEqual(self.before['corpus_sha256'], self.digest)
        self.assertEqual((self.before['eligible'], self.before['passages']), (73, 5304))
        self.assert_corpus_unchanged()
        self.assertEqual(ReleaseRepository(self.reader).corpus_revision(), self.digest)
        self.assertEqual(migrate(self.dsn), [])

    def test_generated_chapter_and_passage_no_mirror_existing_source(self):
        with psycopg.connect(self.dsn) as conn:
            rows = conn.execute('SELECT passage_id,chapter,passage_no FROM face_release.passage').fetchall()
            self.assertEqual(len(rows), 5304)
            expected = {p['passage_id']: (p['source'].get('chapter'), p['source'].get('locator')) for p in self.corpus['passages']}
            self.assertEqual({r[0]: r[1:] for r in rows}, expected)
            with self.assertRaises(psycopg.errors.GeneratedAlways):
                conn.execute("UPDATE face_release.passage SET passage_no='1' WHERE passage_id=%s", (rows[0][0],))

    def test_sync_is_idempotent_and_reimport_preserves_metadata(self):
        again = meta.sync(self.dsn, self.rows, self.revision)
        self.assertEqual((again['inserted'], again['updated'], again['stale_cleared']), (0, 0, 0))
        self.assertEqual(again['unchanged'], len(self.rows))
        self.assertEqual(again['unmapped_db_labels'], [])
        self.assertEqual(status(self.dsn)['labels_with_metadata'], len(self.rows))
        self.assertEqual(import_corpus(self.dsn, self.corpus), self.imported)
        self.assertTrue(meta.check(self.reader, self.rows, self.revision)['matches'])
        self.assert_corpus_unchanged()

    def test_retrieval_for_every_eligible_condition_set_is_unchanged(self):
        repository = ReleaseRepository(self.reader)
        for passage in (p for p in self.corpus['passages'] if p['service_eligible']):
            hits = repository.load_candidates(set(passage['required_labels']))
            with self.subTest(passage=passage['passage_id']):
                self.assertIn(passage['passage_id'], [h.passage_id for h in hits])

    def test_stale_metadata_is_cleared_and_missing_eligible_coverage_rolls_back(self):
        extra = dict(self.rows)
        extra['legacy_feature:old'] = {**self.rows['eye_size:small'], 'label_name': 'legacy'}
        meta.sync(self.dsn, extra, self.revision)
        result = meta.sync(self.dsn, self.rows, self.revision)
        self.assertEqual((result['stale_cleared'], result['unmapped_db_labels']), (1, ['legacy_feature:old']))
        with psycopg.connect(self.dsn) as conn:
            self.assertIsNone(conn.execute("SELECT criteria FROM face_release.label WHERE label_id='legacy_feature:old'").fetchone()[0])
            conn.execute("DELETE FROM face_release.label WHERE label_id='legacy_feature:old'")
        eligible = next(l for p in self.corpus['passages'] if p['service_eligible'] for l in p['required_labels'])
        partial = {k: v for k, v in self.rows.items() if k != eligible}
        before = meta.check(self.dsn, self.rows, self.revision)
        with psycopg.connect(self.dsn) as conn:
            conn.execute('UPDATE face_release.label SET ' + ','.join(f'{c}=NULL' for c in (*meta.COLUMNS, 'metadata_revision'))
                         + ' WHERE label_id=%s', (eligible,))
        with self.assertRaisesRegex(ValueError, 'lack frozen metadata'):
            meta.sync(self.dsn, partial, self.revision)
        self.assertEqual(meta.check(self.dsn, self.rows, self.revision)['eligible_without_metadata'], [eligible])
        self.assertTrue(before['matches'])

    def test_constraints_reject_partial_or_inconsistent_metadata(self):
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            for statement in ("UPDATE face_release.label SET criteria=NULL WHERE label_id='eye_size:small'",
                              "UPDATE face_release.label SET measurement_method='guess' WHERE label_id='eye_size:small'",
                              "UPDATE face_release.label SET measurement_indicator='x' WHERE label_id='eye_size:small'",
                              "UPDATE face_release.label SET measurement_indicator=NULL WHERE label_id='nose_projection:high'",
                              "UPDATE face_release.label SET region=' ' WHERE label_id='eye_size:small'"):
                with self.subTest(statement=statement):
                    with self.assertRaises(psycopg.errors.CheckViolation):
                        conn.execute(statement)

    def test_reader_sees_metadata_but_cannot_write_even_after_disabling_readonly(self):
        with psycopg.connect(self.reader, autocommit=True) as conn:
            self.assertEqual(conn.execute('SHOW default_transaction_read_only').fetchone()[0], 'on')
            row = conn.execute("SELECT region,label_name,measurement_method FROM face_release.label WHERE label_id='eye_size:small'").fetchone()
            self.assertEqual(row, ('eye', self.rows['eye_size:small']['label_name'], meta.CNN))
            self.assertEqual(conn.execute('SELECT count(*) FROM face_release.passage WHERE passage_no IS NOT NULL').fetchone()[0], 5304)
            conn.execute('SET default_transaction_read_only=off')
            for statement in ("UPDATE face_release.label SET criteria='forged' WHERE label_id='eye_size:small'",
                              "INSERT INTO face_release.label (label_id) VALUES ('forged:x')",
                              "DELETE FROM face_release.label WHERE label_id='eye_size:small'",
                              'ALTER TABLE face_release.label DROP COLUMN criteria',
                              'SELECT * FROM face_release.corpus_import', 'SELECT * FROM face_release.source_section'):
                with self.subTest(statement=statement):
                    with self.assertRaises(psycopg.errors.InsufficientPrivilege):
                        conn.execute(statement)


if __name__ == '__main__':
    unittest.main()
