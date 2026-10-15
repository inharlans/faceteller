"""CPU-only final corpus coordinator. Start ONLY after the main annotation freeze.

Wait for automatic label readiness, build/test the corpus, import twice, verify
the actual read-only repository, then publish the compact-canonical DB revision.
No models, photographs, network fetching, annotation rewriting or human review.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath,PureWindowsPath
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.release_repository import ReleaseRepository,database_dsn
from scripts.release import setup_database as db

OUT=ROOT/'outputs/release'
SPLITS={'train':270,'selection':34,'calibration':34,'test':62}
OVERLAY='data/release/local_marks/merged_mark_labels_vlm_agreement.json'
RECORDS='data/release/local_marks/qwen3-vl-2b-v2/records'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(document):return hashlib.sha256(db.canonical_json(document).encode('utf-8')).hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)


def write_atomic(path,value,*,exclusive=False,raw=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=value if raw else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    try:
        if exclusive:
            # Atomic no-replace publication, including on Windows NTFS. Readers
            # cannot observe a partially written ready marker or backup.
            os.link(temporary,path)
        else:temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def state(stage,status,**extra):
    value={'stage':stage,'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    write_atomic(OUT/'literature-finalization-state.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


@contextmanager
def operation_lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'literature-finalization.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def require_no_marker():
    path=ROOT/'data/release/literature_ready.json'
    require(not path.exists() and not path.is_symlink(),'Existing literature_ready.json preserved; finalization already published or needs inspection')


def source_path_key(name):
    require(isinstance(name,str) and bool(name),'Readiness source key must be a nonempty relative path')
    path=PurePosixPath(name.replace('\\','/'))
    require(not path.is_absolute() and not PureWindowsPath(name).drive and '..' not in path.parts
            and path.as_posix()!='.','Readiness source key must stay within the project')
    return path.as_posix()


def normalized_sources(sources):
    """Normalize only the lookup view; frozen marker/map bytes remain authoritative."""
    result={};seen=set()
    for name,digest in sources.items():
        key=source_path_key(name)
        # Windows also aliases case variants. Reject aliases even when hashes agree.
        require(key.casefold() not in seen,'Duplicate readiness source path aliases')
        seen.add(key.casefold());result[key]=digest
    return result


def checked_source(name,digest):
    name=source_path_key(name)
    path=(ROOT/name).resolve()
    require(path.is_relative_to(ROOT.resolve()),'Readiness source escapes the project')
    require(path.is_file() and sha(path)==digest,f'Readiness source hash mismatch: {name}')


def validate_readiness():
    marker_path=ROOT/'data/release/extended_ready.json'
    marker=load(marker_path);registry_path=ROOT/'data/release/training_registry.json'
    registry=load(registry_path);overlay_path=ROOT/OVERLAY;overlay=load(overlay_path)
    require(marker.get('ready') is True,'Automatic label readiness is not true')
    require(marker.get('registry_sha256')==sha(registry_path),'Readiness registry hash mismatch')
    require(marker.get('overlay_sha256')==sha(overlay_path),'Readiness overlay hash mismatch')
    sources=registry.get('source_hashes')
    require(isinstance(sources,dict) and sources and marker.get('source_hashes')==sources,'Readiness source map mismatch')
    lookup=normalized_sources(sources)
    require(lookup.get(OVERLAY)==sha(overlay_path),'Registry does not bind the mark overlay')
    for name,digest in lookup.items():checked_source(name,digest)
    protocol_path='models/selection-20261014/protocol.json'
    groups=load(ROOT/protocol_path)['groups']
    require(set(groups)==set(SPLITS) and all(len(groups[k])==v for k,v in SPLITS.items()),'Expected 270/34/34/62 split sizes')
    ids=[sid for values in groups.values() for sid in values]
    require(all(isinstance(sid,str) and sid and '/' not in sid and '\\' not in sid for sid in ids)
            and len(set(ids))==400,'Expected 400 disjoint participant IDs')
    require(overlay.get('groups')==groups,'Overlay split mismatch')
    require(set(registry['subjects'])==set(ids)==set(overlay['subjects']),'Registry/overlay participant mismatch')
    require(registry['features']['pigmented_marks']['classes']==['absent','present'],'Unexpected mark class order')
    require(overlay.get('preprocess_version')==registry['preprocess_version'],'Overlay preprocessing mismatch')
    require(all(x.get('human_review_count')==0 for x in (marker,registry,overlay)),'Automatic labels must retain human_review_count=0')
    source_map=overlay['source_hashes']
    for key,path in [('protocol',protocol_path),('original_extended_labels','data/release/extended_labels.json'),
                     ('merge_implementation','scripts/release/merge_mark_labels.py')]:
        checked_source(path,source_map[key])
    records=source_map['records']
    require(set(records)==set(ids),'All 400 annotation record hashes required')
    for sid,digest in records.items():checked_source(f'{RECORDS}/{sid}.json',digest)
    counts={}
    for split,people in groups.items():
        counts[split]={'absent':0,'present':0,'unknown':0}
        for sid in people:
            person=registry['subjects'][sid];label=person['labels']['pigmented_marks']
            require(person['split']==split and label==overlay['subjects'][sid]['pigmented_marks'],'Registry mark target/split differs from overlay')
            require(label in (None,'absent','present'),'Unexpected mark target')
            counts[split]['unknown' if label is None else label]+=1
        require(overlay['class_support'][split]==counts[split],'Overlay support differs from targets')
        support=registry['class_support'][split]['pigmented_marks']
        require(support=={'counts':[counts[split]['absent'],counts[split]['present']],
                          'unknown':counts[split]['unknown']},'Registry support differs from targets')
    require(min(counts['train']['absent'],counts['train']['present'])>=2,'Insufficient training marks: at least 2 per class required')
    require(all(counts[s]['absent']+counts[s]['present']>=2 for s in ('selection','calibration')),'Insufficient selection/calibration mark targets')
    validation=marker['validation']
    require(validation.get('class_support')==counts and validation.get('train_per_class_minimum')==2
            and validation.get('selection_known_minimum')==2 and validation.get('calibration_known_minimum')==2
            and validation.get('test_used_for_readiness') is False
            and validation.get('independent_accuracy_verified') is False,'Readiness validation proof mismatch')
    return {'extended_ready_sha256':sha(marker_path),'registry_sha256':sha(registry_path),
            'overlay_sha256':sha(overlay_path),'validation':validation}


def promote_pending(proof):
    path=ROOT/'data/release/literature/pending_labels.json'
    backup=OUT/'literature-pending-labels.original.json'
    current=path.read_bytes()
    if not backup.exists():
        require('pigmented_marks' in json.loads(current)['features'],'Pending marks already removed without a finalizer backup')
        write_atomic(backup,current,exclusive=True,raw=True)
    original=load(backup);expected=deepcopy(original)
    require('pigmented_marks' in expected['features'],'Original pending backup lacks mark labels')
    del expected['features']['pigmented_marks']
    expected['status']='pending_local_annotation' if expected['features'] else 'ready_automatic_labels'
    expected['status_note']='Automatic weak-label support verified; historical source note is superseded for pigmented_marks. Upper-face visible dark marks only; absence does not mean whole-face absence. No medical identity, human review or model accuracy approval. Annotation rules are unchanged.'
    expected['ready_proof']=proof
    require(current==backup.read_bytes() or json.loads(current)==expected,'Pending labels changed outside this finalization; original backup preserved')
    if json.loads(current)!=expected:write_atomic(path,expected)
    return sha(path)


def literature_inputs():
    base=ROOT/'data/release/literature'
    paths=[base/'source_registry.json',base/'snapshots/snapshot_manifest.json',
           ROOT/'scripts/release/build_reference_corpus.py',ROOT/'tests/test_release_literature.py']
    paths+=list((base/'annotations').glob('*.json'))+list((base/'transcriptions').rglob('*.json'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def run_stage(stage,*arguments,environment=None):
    state(stage,'running')
    with (OUT/f'literature-{stage}.log').open('ab',buffering=0) as log:
        subprocess.run([sys.executable,'-X','utf8',*arguments],cwd=ROOT,env=environment,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def run_isolated_repository_tests(admin):
    # Audited setUpClass creates faceteller_test_<uuid> and derives cls.dsn from
    # that database. All fixture TRUNCATE/imports use cls.dsn, never the active DB.
    # Fail on skipped tests too: no DB-less success may publish readiness.
    program="""import sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_release_repository')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1)
"""
    environment=os.environ.copy();environment['FACETELLER_ADMIN_DATABASE_URL']=admin
    run_stage('repository-tests','-c',program,environment=environment)


def verify_all_candidates(snapshot,document):
    eligible=[p for p in document['passages'] if p['service_eligible']]
    require(eligible,'Final corpus must have eligible passages')
    def order(p):
        pid=p['passage_id'];return p['literature_priority'],p['passage_order'],(0,int(pid)) if pid.isdecimal() else (1,pid),pid
    checked={};counts={'eligible_condition_sets':len(eligible),'missing_condition_checks':0,'wrong_class_checks':0,'extra_condition_checks':0}
    def check(labels):
        key=frozenset(labels)
        if key in checked:return checked[key]
        expected=sorted([p for p in eligible if set(p['required_labels'])<=key],key=order)
        actual=snapshot.load_candidates(set(key))
        require([p.passage_id for p in actual]==[p['passage_id'] for p in expected],'Runtime AND matching/order differs from full eligible corpus')
        for got,want in zip(actual,expected):
            source={**want['source'],'annotation_status':want['annotation_status']}
            if want.get('display_policy') is not None:source['display_policy']=want['display_policy']
            require((got.original_text,got.translation,got.commentary,got.required_labels,got.source,
                     got.literature_priority,got.passage_order)==
                    (want['original_text'],want['translation'],want['commentary'],frozenset(want['required_labels']),source,
                     want['literature_priority'],want['passage_order']),'Runtime passage/provenance differs from corpus')
        checked[key]={p.passage_id for p in actual};return checked[key]
    for passage in eligible:
        labels=set(passage['required_labels']);pid=passage['passage_id']
        require(pid in check(labels),'Eligible passage missing when all conditions match')
        check(labels|{'__finalizer_unrelated__:never'});counts['extra_condition_checks']+=1
        for label in labels:
            remainder=labels-{label}
            require(pid not in check(remainder),'Passage matched with missing condition')
            feature=label.split(':',1)[0]
            require(pid not in check(remainder|{feature+':__finalizer_wrong_class__'}),'Passage matched with wrong class')
            counts['missing_condition_checks']+=1;counts['wrong_class_checks']+=1
    check(set());counts['unique_queries']=len(checked)
    return counts


def verify_database(document,digest,admin,reader):
    import psycopg
    expected_passages={p['passage_id']:p for p in document['passages']}
    expected_sources={s['literature_id']:s for s in document['sources']}
    expected_sections={s['section_id']:s for s in document.get('source_sections',[])}
    # Administrative SELECT is needed only for private archival tables. Never
    # broaden the service role to inspect them.
    with psycopg.connect(admin) as connection:
        connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        stored=connection.execute('SELECT document FROM face_release.corpus_import WHERE sha256=%s',(digest,)).fetchone()
        require(stored is not None and stored[0]==document,'Imported complete document differs from final corpus')
        sources=dict(connection.execute('SELECT literature_id,document FROM face_release.literature').fetchall())
        passages=dict(connection.execute('SELECT passage_id,document FROM face_release.passage').fetchall())
        require(all(sources.get(k)==v for k,v in expected_sources.items()),'DB source documents differ from corpus')
        require(all(passages.get(k)==v for k,v in expected_passages.items()),'DB passage documents differ from corpus')
        sections=connection.execute('SELECT section_id,document,corpus_sha256,text_sha256,original_text FROM face_release.source_section').fetchall()
        require({r[0]:r[1] for r in sections}==expected_sections,'Private source archive differs from corpus')
        require(all(r[2]==digest and r[3]==hashlib.sha256(r[4].encode()).hexdigest()
                    and (r[3],r[4])==(expected_sections[r[0]]['text_sha256'],expected_sections[r[0]]['original_text'])
                    for r in sections),'Private source archive revision/text hash mismatch')
        retained={'sources':len(sources),'passages':len(passages)}
    repository=ReleaseRepository(reader)
    with repository.snapshot() as snapshot:
        conn=snapshot.connection
        require(snapshot.corpus_revision()==digest,'Runtime active canonical corpus differs from import')
        role=conn.execute('SELECT rolsuper,rolcreatedb,rolcreaterole,rolbypassrls FROM pg_roles WHERE rolname=current_user').fetchone()
        require(role is not None and not any(role),'Runtime reader has administrative privileges')
        require(conn.execute('SHOW default_transaction_read_only').fetchone()[0]=='on','Runtime reader is not read-only by default')
        safe=('literature','passage','label','label_passage','active_corpus')
        private=('source_section','corpus_import','schema_migration')
        for table in safe+private:
            name='face_release.'+table
            require(not conn.execute('SELECT has_table_privilege(current_user,%s,%s)',(name,'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER')).fetchone()[0],'Runtime reader has table write privileges')
            can_read=conn.execute('SELECT has_table_privilege(current_user,%s,%s)',(name,'SELECT')).fetchone()[0]
            require(bool(can_read)==(table in safe),'Runtime reader archive/metadata SELECT permission mismatch')
        active_ids={r[0] for r in conn.execute('SELECT passage_id FROM face_release.passage WHERE service_eligible').fetchall()}
        require(active_ids=={p['passage_id'] for p in document['passages'] if p['service_eligible']},'Runtime eligible set differs from final corpus')
        and_checks=verify_all_candidates(snapshot,document)
    return {'corpus_sha256':digest,'sources':len(expected_sources),'passages':len(expected_passages),
            'source_sections':len(expected_sections),'eligible':len(active_ids),'retained_db_rows':retained,
            'readonly_role_verified':True,'private_archive_hidden_from_reader':True,
            'archive_document_and_text_hashes_verified':True,'and_checks':and_checks}


def finalize(*,wait=False):
    require_no_marker()
    state('automatic-labels','waiting')
    while not (ROOT/'data/release/extended_ready.json').exists():
        if not wait:raise RuntimeError('extended_ready.json is absent; no pending/corpus/DB changes made')
        failed=OUT/'input_preparation.json'
        if failed.exists() and load(failed).get('status')=='failed':raise RuntimeError('Input preparation failed; finalization stopped')
        time.sleep(10);require_no_marker()
    proof=validate_readiness();inputs=literature_inputs()
    pending_sha=promote_pending(proof)
    def unchanged():
        require_no_marker()
        require(validate_readiness()==proof,'Readiness changed during finalization')
        require(literature_inputs()==inputs,'Frozen literature inputs changed during finalization')
        require(sha(ROOT/'data/release/literature/pending_labels.json')==pending_sha,'Pending labels changed during finalization')
    run_stage('build','scripts/release/build_reference_corpus.py','build')
    run_stage('literature-tests','-m','unittest','tests.test_release_literature','-v')
    corpus_path=ROOT/'data/release/literature/reference_corpus_v1.json'
    document=load(corpus_path);db.validate_corpus(document);digest=canonical_hash(document)
    admin=database_dsn(admin=True);reader=database_dsn()
    run_isolated_repository_tests(admin)
    unchanged()
    require(canonical_hash(load(corpus_path))==digest,'Corpus changed after literature tests')
    state('postgres-import','running')
    db.migrate(admin)
    first=db.import_corpus(admin,document);before=db.status(admin)
    second=db.import_corpus(admin,document);after=db.status(admin)
    require(first==second and before==after and first['sha256']==digest,'Import canonical identity/idempotence failed')
    verified=verify_database(document,digest,admin,reader)
    unchanged()
    require(canonical_hash(load(corpus_path))==digest,'Corpus changed after actual DB verification')
    report={'status':'complete','corpus_sha256':digest,'import':first,'idempotent_reimport':True,
            'database':verified,'ready_proof':proof,'literature_input_hashes':inputs,
            'pending_original_sha256':sha(OUT/'literature-pending-labels.original.json'),
            'pending_promoted_sha256':pending_sha,'literature_tests_passed':True,
            'repository_tests_passed':True,'repository_tests_database':'isolated random faceteller_test_<uuid>; active fixtures never replaced',
            'human_review_count':0,'meaning':'Historical morphology-based literature retrieval; not personality, health or fortune prediction. Automatic weak labels are not independent human ground truth.'}
    report_path=OUT/'literature-finalization.json';write_atomic(report_path,report)
    require_no_marker()
    # Only the actual import's compact-canonical SHA is a readiness identity.
    write_atomic(ROOT/'data/release/literature_ready.json',
                 {'ready':True,'corpus_sha256':first['sha256'],'import_sha256':first['sha256'],
                  'registry_sha256':proof['registry_sha256'],'extended_ready_sha256':proof['extended_ready_sha256'],
                  'counts':{k:first[k] for k in ('sources','passages','source_sections','eligible')},
                  'finalization_report_sha256':sha(report_path),'human_review_count':0,
                  'decision':'Automated corpus, canonical DB revision, read-only permissions and AND retrieval checks passed; no human review'},exclusive=True)
    state('literature-finalization','complete',corpus_sha256=digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    with operation_lock():
        try:finalize(wait=args.wait)
        except Exception as exc:
            # DB errors can contain connection credentials; never serialize them.
            state('literature-finalization','failed',error_type=type(exc).__name__)
            if isinstance(exc,db.psycopg.Error):
                raise RuntimeError(f'PostgreSQL finalization failed: {type(exc).__name__}; check private configuration') from None
            raise


if __name__=='__main__':main()
