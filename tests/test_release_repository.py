"""Real PostgreSQL integration tests; synthetic reference text only, no models.

Uses FACETELLER_ADMIN_DATABASE_URL or local-init's private config. Requires a
dedicated administrative role with CREATEDB/CREATEROLE. Creates and removes only
its own randomly named database/login, never truncates the release database.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from facework.pilot_contract import Passage, PassageSelector, Decision
from facework.release_repository import ReleaseRepository, database_dsn, local_config_path
from scripts.release.setup_database import (
    MIGRATIONS, grant_reader, import_corpus, migrate, status, validate_corpus, canonical_json,
)


def corpus():
    source = {"title":"Synthetic reference", "url":"https://example.invalid/reference",
              "chapter":"Synthetic chapter", "locator":"paragraph 1", "extra":"preserve me"}
    def passage(key, labels, priority=0, order=0, eligible=True):
        return {"passage_id":key,"literature_id":"synthetic-book",
                "original_text":"Synthetic original","translation":"Synthetic translation",
                "commentary":"Synthetic commentary","source":dict(source),
                "required_labels":labels,"literature_priority":priority,"passage_order":order,
                "annotation_status":"ai_annotated_unreviewed","service_eligible":eligible,
                "extra":{"provenance":"synthetic fixture"}}
    return {"schema_version":1,"extra_metadata":{"preserved":True},
            "sources":[{"literature_id":"synthetic-book","title":source["title"],"extra":[1,2]}],
            "passages":[passage("10",["eye_shape:round"]),
                        passage("2",["eye_shape:round"]),
                        passage("canonical:A",["eye_shape:round","nose_width:wide"]),
                        passage("priority",["eye_shape:round"],-1,99),
                        passage("later",["eye_shape:round"],0,1),
                        passage("excluded",["eye_shape:round"],eligible=False),
                        passage("archive-without-labels",[],eligible=False)]}


def source_section(index=1,text='古籍原文\nUntranslated source text'):
    return {'section_id':f'page:1:{index}','literature_id':'synthetic-book',
            'page_id':'page','heading':'Historical heading','occurrence':1,
            'original_text':text,'text_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'source':{'title':'Synthetic reference','url':'https://example.invalid/revision/1',
                      'chapter':'Chapter','snapshot_file':'snapshots/page.html',
                      'snapshot_sha256':'a'*64,'wikisource_revid':1,'scan_page':None},
            'annotation_status':'source_archive_unreviewed','service_eligible':False}


class CorpusValidationTests(unittest.TestCase):
    def test_eligible_requires_labels_and_complete_text(self):
        for field, value in [("required_labels",[]),("translation",""),("service_eligible",1),
                             ("annotation_status",""),("passage_order",True)]:
            with self.subTest(field=field):
                doc=corpus(); doc["passages"][0][field]=value
                with self.assertRaises(ValueError): validate_corpus(doc)

    def test_duplicate_keys_missing_foreign_key_and_bad_conditions(self):
        changes = [("passage_id","2"),("literature_id","unknown"),
                   ("required_labels",["eye_shape:round","eye_shape:round"]),
                   ("required_labels",["eye_shape:round","eye_shape:thin"]),
                   ("required_labels",["123"]),("required_labels",["eye_shape:round:extra"])]
        for field,value in changes:
            with self.subTest(value=value):
                doc=corpus(); doc["passages"][0][field]=value
                with self.assertRaises(ValueError): validate_corpus(doc)

    def test_excluded_incomplete_annotation_is_preserved(self):
        doc=corpus(); doc["passages"][-1]["translation"]=""; doc["passages"][-1]["source"]={}
        sources, passages=validate_corpus(doc)
        self.assertEqual(len(passages),7)
        self.assertEqual(passages[-1]["translation"],"")

    def test_empty_query_does_not_open_connection(self):
        self.assertEqual(ReleaseRepository("invalid connection").load_candidates(set()),[])

    def test_optional_archive_strict_fields_hash_and_no_promotion(self):
        validate_corpus(corpus())  # legacy snapshots remain valid
        valid=corpus();valid['source_sections']=[source_section()];validate_corpus(valid)
        for key,value in [('text_sha256','0'*64),('original_text','changed'),('original_text',' '),
                          ('service_eligible',True),('service_eligible',0),
                          ('annotation_status','human_reviewed'),('literature_id','absent'),
                          ('occurrence',True),('occurrence',0),('section_id','page:2:1'),
                          ('section_id','page:1:01'),('page_id',''),('source',{})]:
            with self.subTest(key=key,value=value):
                bad=copy.deepcopy(valid);bad['source_sections'][0][key]=value
                with self.assertRaises(ValueError):validate_corpus(bad)
        for sections in (None,{},[None],[source_section(),source_section()]):
            with self.subTest(sections=sections):
                bad=corpus();bad['source_sections']=sections
                with self.assertRaises(ValueError):validate_corpus(bad)

    def test_archive_scan_requires_transcription_provenance_and_valid_page_range(self):
        doc=corpus();doc['source_sections']=[source_section()]
        source=doc['source_sections'][0]['source']
        source.update(wikisource_revid=None,scan_source_id='scan-a',scan_page='pdf_page:1',
                      transcription_sha256='b'*64,transcription_files=['transcriptions/one.json'],
                      pdf_page=1,pdf_page_end=2)
        validate_corpus(doc)
        for key,value in [('transcription_sha256','bad'),('transcription_files',[]),
                          ('pdf_page',0),('pdf_page_end',0),('wikisource_revid',True)]:
            with self.subTest(key=key):
                bad=copy.deepcopy(doc);bad['source_sections'][0]['source'][key]=value
                with self.assertRaises(ValueError):validate_corpus(bad)


@unittest.skipUnless(os.environ.get("FACETELLER_ADMIN_DATABASE_URL") or local_config_path().exists(),
                     "Actual PostgreSQL required: run setup_database.py local-init")
class PostgreSQLRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin=database_dsn(admin=True)
        token=uuid4().hex[:16]
        cls.db="faceteller_test_"+token
        cls.role="faceteller_test_reader_"+token
        cls.password=secrets.token_urlsafe(24)
        with psycopg.connect(cls.admin,dbname="postgres",autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(cls.db)))
        cls.addClassCleanup(cls.cleanup_database)
        cls.dsn=make_conninfo(cls.admin,dbname=cls.db)
        migrate(cls.dsn)
        grant_reader(cls.dsn,cls.role,cls.password)
        cls.reader=make_conninfo(cls.dsn,user=cls.role,password=cls.password)
        cls.repo=ReleaseRepository(cls.reader)

    @classmethod
    def cleanup_database(cls):
        with psycopg.connect(cls.admin,dbname="postgres",autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(cls.db)))
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(cls.role)))

    def setUp(self):
        with psycopg.connect(self.dsn) as conn:
            conn.execute("TRUNCATE face_release.source_section, face_release.active_corpus, face_release.corpus_import, face_release.label_passage, "
                         "face_release.label, face_release.passage, face_release.literature")
        self.document=corpus()
        import_corpus(self.dsn,self.document)

    def test_and_matching_retains_all_conditions(self):
        single=self.repo.load_candidates({"eye_shape:round"})
        self.assertEqual([p.passage_id for p in single],["priority","2","10","later"])
        both=self.repo.load_candidates({"eye_shape:round","nose_width:wide"})
        self.assertEqual([p.passage_id for p in both],["priority","2","10","canonical:A","later"])
        multi=next(p for p in both if p.passage_id=="canonical:A")
        self.assertIsInstance(multi,Passage)
        self.assertEqual(multi.required_labels,frozenset({"eye_shape:round","nose_width:wide"}))
        self.assertEqual(multi.source,{**self.document["passages"][2]["source"],
                                      "annotation_status":"ai_annotated_unreviewed"})
        self.assertEqual(self.repo.load_candidates({"nose_width:wide"}),[])
        self.assertEqual(self.repo.load_candidates({"unknown:unknown"}),[])

    def test_repeat_import_and_migration_are_idempotent(self):
        before=status(self.dsn)
        first=import_corpus(self.dsn,self.document)
        second=import_corpus(self.dsn,json.loads(json.dumps(self.document)))
        self.assertEqual(first,second)
        self.assertEqual(status(self.dsn),before)
        self.assertEqual(before["passages"],7)
        self.assertEqual(before["eligible"],5)
        self.assertEqual(before["imports"],1)
        self.assertEqual(before['corpus_sha256'],first['sha256'])
        self.assertEqual(self.repo.corpus_revision(),first['sha256'])
        self.assertEqual(migrate(self.dsn),[])
        grant_reader(self.dsn,self.role)  # Repeated privilege setup preserves login password.
        self.assertEqual(len(self.repo.load_candidates({"eye_shape:round"})),4)
        with psycopg.connect(self.dsn) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM face_release.label_passage").fetchone()[0],7)

    def test_full_archival_payload_and_unreviewed_status_survive(self):
        with psycopg.connect(self.dsn) as conn:
            self.assertEqual(conn.execute("SELECT document FROM face_release.corpus_import").fetchone()[0],self.document)
            for raw in self.document["passages"]:
                row=conn.execute("SELECT document,annotation_status FROM face_release.passage WHERE passage_id=%s",
                                 (raw["passage_id"],)).fetchone()
                self.assertEqual(row,(raw,"ai_annotated_unreviewed"))
            self.assertEqual(conn.execute("SELECT document FROM face_release.literature").fetchone()[0],self.document["sources"][0])

    def test_source_archive_preserved_private_idempotent_and_never_candidates(self):
        doc=copy.deepcopy(self.document);doc['source_sections']=[source_section(),source_section(2,'未翻譯全文')]
        candidates=self.repo.load_candidates({'eye_shape:round'})
        first=import_corpus(self.dsn,doc);before=status(self.dsn)
        self.assertEqual(first['source_sections'],2)
        self.assertEqual(import_corpus(self.dsn,doc),first)
        self.assertEqual(status(self.dsn),before)
        self.assertEqual(self.repo.load_candidates({'eye_shape:round'}),candidates)
        with psycopg.connect(self.dsn) as conn:
            rows=conn.execute('SELECT section_id,document,original_text,text_sha256,corpus_sha256,service_eligible FROM face_release.source_section ORDER BY section_id').fetchall()
            self.assertEqual(rows,[(s['section_id'],s,s['original_text'],s['text_sha256'],first['sha256'],False) for s in doc['source_sections']])
            self.assertEqual(conn.execute('SELECT document FROM face_release.corpus_import WHERE sha256=%s',(first['sha256'],)).fetchone()[0],doc)
        grant_reader(self.dsn,self.role)  # repeated grant cannot accidentally publish archive
        with psycopg.connect(self.reader,autocommit=True) as conn:
            conn.execute('SET default_transaction_read_only=off')
            for statement in ('SELECT * FROM face_release.source_section',
                              "UPDATE face_release.source_section SET original_text='changed'",
                              'DELETE FROM face_release.source_section','TRUNCATE face_release.source_section'):
                with self.subTest(statement=statement):
                    with self.assertRaises(psycopg.errors.InsufficientPrivilege):conn.execute(statement)

    def test_source_archive_replaces_revision_restores_old_and_legacy_clears(self):
        old=copy.deepcopy(self.document);old['source_sections']=[source_section(),source_section(2)]
        old_hash=import_corpus(self.dsn,old)['sha256']
        new=copy.deepcopy(old);new['source_sections']=[source_section(3,'New untranslated source')]
        new_hash=import_corpus(self.dsn,new)['sha256']
        self.assertNotEqual(old_hash,new_hash)
        with psycopg.connect(self.dsn) as conn:
            self.assertEqual(conn.execute('SELECT section_id,corpus_sha256 FROM face_release.source_section').fetchall(),[('page:1:3',new_hash)])
        import_corpus(self.dsn,old)
        self.assertEqual(self.repo.corpus_revision(),old_hash)
        self.assertEqual(status(self.dsn)['source_sections'],2)
        import_corpus(self.dsn,self.document)
        self.assertEqual(status(self.dsn)['source_sections'],0)
        with psycopg.connect(self.dsn) as conn:
            self.assertEqual(conn.execute('SELECT document FROM face_release.corpus_import WHERE sha256=%s',(old_hash,)).fetchone()[0],old)

    def test_archive_hash_database_constraint_and_late_failure_roll_back_revision(self):
        old=copy.deepcopy(self.document);old['source_sections']=[source_section()]
        old_hash=import_corpus(self.dsn,old)['sha256'];before=status(self.dsn)
        with psycopg.connect(self.dsn,autocommit=True) as conn:
            with self.assertRaises(psycopg.errors.CheckViolation):
                conn.execute("UPDATE face_release.source_section SET text_sha256=%s",('0'*64,))
            conn.execute("ALTER TABLE face_release.source_section ADD CONSTRAINT reject_test CHECK (original_text<>'reject')")
        try:
            broken=copy.deepcopy(old);broken['source_sections']=[source_section(2,'reject')]
            broken['passages'][0]['translation']='must roll back'
            with self.assertRaises(psycopg.errors.CheckViolation):import_corpus(self.dsn,broken)
            self.assertEqual(status(self.dsn),before)
            self.assertEqual(self.repo.corpus_revision(),old_hash)
            with psycopg.connect(self.dsn) as conn:
                self.assertEqual(conn.execute('SELECT document FROM face_release.source_section').fetchone()[0],old['source_sections'][0])
                self.assertEqual(conn.execute("SELECT translation FROM face_release.passage WHERE passage_id='10'").fetchone()[0],old['passages'][0]['translation'])
        finally:
            with psycopg.connect(self.dsn) as conn:conn.execute('ALTER TABLE face_release.source_section DROP CONSTRAINT reject_test')

    def test_api_annotation_status_uses_column_without_promoting_human_review(self):
        self.document['passages'][0]['source']['annotation_status']='human_reviewed'
        import_corpus(self.dsn,self.document)
        passage=next(p for p in self.repo.load_candidates({'eye_shape:round'}) if p.passage_id=='10')
        self.assertEqual(passage.source['annotation_status'],'ai_annotated_unreviewed')
        with psycopg.connect(self.dsn) as conn:
            raw=conn.execute("SELECT source FROM face_release.passage WHERE passage_id='10'").fetchone()[0]
            self.assertEqual(raw['annotation_status'],'human_reviewed')  # raw input retained, not endorsed

    def test_api_preserves_historical_display_policy_without_rewriting_source(self):
        raw=self.document['passages'][0]
        raw['display_policy']='historical_text_only_not_a_personal_prediction'
        raw['source']['display_policy']='conflicting_nested_value'
        import_corpus(self.dsn,self.document)
        passage=next(p for p in self.repo.load_candidates({'eye_shape:round'}) if p.passage_id=='10')
        self.assertEqual(passage.source['display_policy'],raw['display_policy'])
        self.assertEqual(passage.source['annotation_status'],'ai_annotated_unreviewed')
        with psycopg.connect(self.dsn) as conn:
            saved=conn.execute("SELECT source,document FROM face_release.passage WHERE passage_id='10'").fetchone()
            self.assertEqual(saved,(raw['source'],raw))

    def test_updated_conditions_replace_mapping_and_absent_records_retire(self):
        new=copy.deepcopy(self.document)
        new["passages"]=[new["passages"][0]]
        new["passages"][0]["required_labels"]=["nose_width:narrow"]
        import_corpus(self.dsn,new)
        self.assertEqual(self.repo.load_candidates({"eye_shape:round"}),[])
        self.assertEqual([p.passage_id for p in self.repo.load_candidates({"nose_width:narrow"})],["10"])
        self.assertEqual(status(self.dsn)["passages"],7)
        self.assertEqual(status(self.dsn)["imports"],2)
        import_corpus(self.dsn,self.document)
        self.assertEqual(len(self.repo.load_candidates({"eye_shape:round"})),4)
        self.assertEqual(status(self.dsn)["imports"],2)

    def test_old_revision_reimport_sets_active_hash_not_newest_audit_time(self):
        old_hash=hashlib.sha256(canonical_json(self.document).encode('utf-8')).hexdigest()
        newer=copy.deepcopy(self.document);newer['schema_version']=2
        newer['passages'][0]['translation']='New revision text'
        new_hash=import_corpus(self.dsn,newer)['sha256']
        self.assertNotEqual(old_hash,new_hash)
        self.assertEqual(self.repo.corpus_revision(),new_hash)
        import_corpus(self.dsn,self.document)
        self.assertEqual(self.repo.corpus_revision(),old_hash)
        with psycopg.connect(self.reader) as conn:
            self.assertEqual(conn.execute('SELECT singleton,version,canonical_sha256 FROM face_release.active_corpus').fetchall(),[(1,'1',old_hash)])
        with psycopg.connect(self.dsn) as conn:
            self.assertEqual(conn.execute('SELECT sha256 FROM face_release.corpus_import ORDER BY imported_at DESC LIMIT 1').fetchone()[0],new_hash)
            self.assertEqual(conn.execute('SELECT count(*) FROM face_release.corpus_import').fetchone()[0],2)

    def test_candidates_and_revision_share_snapshot_during_import(self):
        old_hash=self.repo.corpus_revision()
        newer=copy.deepcopy(self.document);newer['passages'][0]['translation']='Updated while reading'
        with self.repo.snapshot() as snapshot:
            self.assertEqual(snapshot.corpus_revision(),old_hash)
            new_hash=import_corpus(self.dsn,newer)['sha256']
            self.assertEqual(snapshot.corpus_revision(),old_hash)
            candidate=next(p for p in snapshot.load_candidates({'eye_shape:round'}) if p.passage_id=='10')
            self.assertEqual(candidate.translation,'Synthetic translation')
        self.assertEqual(self.repo.corpus_revision(),new_hash)
        candidate=next(p for p in self.repo.load_candidates({'eye_shape:round'}) if p.passage_id=='10')
        self.assertEqual(candidate.translation,'Updated while reading')

    def test_revision_is_unknown_until_explicit_import(self):
        with psycopg.connect(self.dsn) as conn:
            conn.execute('DELETE FROM face_release.active_corpus')
        self.assertIsNone(self.repo.corpus_revision())
        # Historical documents alone never imply which revision is active.
        self.assertEqual(migrate(self.dsn),[])
        self.assertIsNone(self.repo.corpus_revision())
        result=import_corpus(self.dsn,self.document)
        self.assertEqual(self.repo.corpus_revision(),result['sha256'])

    def test_invalid_snapshot_does_not_modify_database(self):
        before=status(self.dsn)
        broken=copy.deepcopy(self.document)
        broken["passages"][0]["translation"]=""
        with self.assertRaises(ValueError): import_corpus(self.dsn,broken)
        self.assertEqual(status(self.dsn),before)

    def test_database_failure_rolls_back_entire_import(self):
        before=status(self.dsn)
        with psycopg.connect(self.dsn) as conn:
            conn.execute("ALTER TABLE face_release.passage ADD CONSTRAINT test_reject CHECK (original_text <> 'reject')")
        try:
            broken=copy.deepcopy(self.document)
            broken["sources"][0]["title"]="Should roll back"
            broken["passages"][-1]["original_text"]="reject"
            with self.assertRaises(psycopg.errors.CheckViolation): import_corpus(self.dsn,broken)
            self.assertEqual(status(self.dsn),before)
            with psycopg.connect(self.dsn) as conn:
                self.assertEqual(conn.execute("SELECT title FROM face_release.literature").fetchone()[0],"Synthetic reference")
            self.assertEqual(len(self.repo.load_candidates({"eye_shape:round"})),4)
        finally:
            with psycopg.connect(self.dsn) as conn:
                conn.execute("ALTER TABLE face_release.passage DROP CONSTRAINT test_reject")

    def test_reader_acl_blocks_writes_even_after_readonly_setting_disabled(self):
        forbidden=["INSERT INTO face_release.label VALUES ('eye_shape:other')",
                   "UPDATE face_release.passage SET translation='changed'",
                   "DELETE FROM face_release.label_passage", "TRUNCATE face_release.passage CASCADE",
                   "CREATE TABLE face_release.evil (id int)","CREATE TABLE public.evil (id int)",
                   "CREATE SCHEMA evil", "CREATE TEMP TABLE evil (id int)",
                   "ALTER TABLE face_release.passage ADD COLUMN evil int",
                   "SELECT * FROM face_release.corpus_import", "SELECT * FROM face_release.schema_migration",
                   "UPDATE face_release.active_corpus SET version='forged'",
                   "DELETE FROM face_release.active_corpus", "TRUNCATE face_release.active_corpus"]
        with psycopg.connect(self.reader,autocommit=True) as conn:
            self.assertEqual(conn.execute("SHOW default_transaction_read_only").fetchone()[0],"on")
            self.assertEqual(conn.execute("SELECT count(*) FROM face_release.passage").fetchone()[0],7)
            self.assertEqual(conn.execute('SELECT count(*) FROM face_release.active_corpus').fetchone()[0],1)
            conn.execute("SET default_transaction_read_only=off")
            for statement in forbidden:
                with self.subTest(statement=statement):
                    with self.assertRaises(psycopg.errors.InsufficientPrivilege): conn.execute(statement)

    def test_reader_cannot_be_an_admin_or_owner(self):
        with psycopg.connect(self.dsn) as conn:
            owner=conn.execute("SELECT current_user").fetchone()[0]
        with self.assertRaisesRegex(ValueError,"unprivileged"):
            grant_reader(self.dsn,owner)

    def test_migration_checksum_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in MIGRATIONS.glob("*.sql"):
                (Path(directory)/path.name).write_bytes(path.read_bytes()+b"\n-- changed\n")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                migrate(self.dsn,Path(directory))
        self.assertEqual(migrate(self.dsn),[])

    def test_selector_compatibility_and_no_repository_prelimit(self):
        doc=corpus()
        for n in range(12):
            p=copy.deepcopy(doc["passages"][0]); p["passage_id"]=f"extra-{n}"; doc["passages"].append(p)
        import_corpus(self.dsn,doc)
        candidates=self.repo.load_candidates({"eye_shape:round"})
        self.assertEqual(len(candidates),16)
        decision=Decision("eye_shape","eye","CONFIRMED","eye_shape:round","round",.8,None,None,
                          "synthetic",(),"classification")
        selected=PassageSelector().select([decision],candidates)
        self.assertEqual(len(selected),3)

    def test_database_failure_is_not_silenced_as_empty_result(self):
        with self.assertRaises(psycopg.OperationalError):
            ReleaseRepository(make_conninfo(self.reader,port=1),connect_timeout=1).load_candidates({"eye_shape:round"})


if __name__ == "__main__":
    unittest.main()
