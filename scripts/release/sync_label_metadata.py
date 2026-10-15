"""Fill face_release.label design metadata from frozen label definitions. CPU/DB only.

Every registry classification class and depth class becomes one label row with
region, label_name, criteria, measurement_indicator and measurement_method.
Criteria text is assembled only from values recorded in the frozen files
(registry thresholds, landmark targets, depth config); visual AI-consensus labels
have no physical boundary and say so. Repeatable: unchanged rows are not rewritten.

  .venv/Scripts/python.exe -B -X utf8 scripts/release/sync_label_metadata.py preview
  .venv/Scripts/python.exe -B -X utf8 scripts/release/sync_label_metadata.py sync
  .venv/Scripts/python.exe -B -X utf8 scripts/release/sync_label_metadata.py check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg

from facework.release_repository import database_dsn
from scripts.release.setup_database import LOCK_ID, canonical_json

INPUTS = {'training_registry': ROOT / 'data/release/training_registry.json',
          'landmark_labels': ROOT / 'data/release/landmark_labels.json',
          'depth_config': ROOT / 'data/release/depth_config.json'}
COLUMNS = ('region', 'label_name', 'criteria', 'measurement_indicator', 'measurement_method')
CNN = 'cnn_image_classification'
DEPTH = 'vggt_multiview_depth_measurement'
SERVICE_CNN = ' 서비스 판정: 이 참조 레이블로 미세조정한 부위 영상 CNN의 보정 신뢰도가 헤드별 기준값 이상일 때만 확정한다.'


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and value == value and abs(value) != float('inf'), 'non-finite boundary')
    return repr(float(value))


def _interval(metric, cuts, index):
    """Matches searchsorted(side='right') / value<lower: the left boundary is inclusive."""
    values = [_number(c) for c in cuts]
    if index == 0:
        return f'{metric} < {values[0]}'
    if index == len(values):
        return f'{metric} ≥ {values[-1]}'
    return f'{values[index - 1]} ≤ {metric} < {values[index]}'


def _classification_criteria(key, feature, landmark):
    thresholds = feature.get('thresholds')
    classes = feature['classes']
    provenance = feature.get('annotation_provenance', '')
    if not thresholds:
        # AI visual consensus: no measured boundary exists in the frozen files.
        note = f" 참고: {feature['note']}" if feature.get('note') else ''
        if feature.get('rater_fleiss_kappa') is not None:
            note += f" AI 판정자 간 Fleiss kappa {_number(feature['rater_fleiss_kappa'])}."
        return [f"시각 참조 레이블(수치 물리 경계 없음). 근거: {provenance}.{note}" + SERVICE_CNN for _ in classes]
    if 'cuts' in thresholds:
        _require(feature.get('source') == 'landmark', f'{key}: cut thresholds outside landmark source')
        reference = landmark.get('features', {}).get(key)
        _require(reference == thresholds, f'{key}: registry thresholds differ from landmark_labels.json')
        cuts = thresholds['cuts']
        _require(len(cuts) == len(classes) - 1 and thresholds.get('classes') == classes, f'{key}: cut/class count mismatch')
        if 'physical_centre' in thresholds:
            basis = (f"경계는 물리 중심 {_number(thresholds['physical_centre'])} ± 학습군 {thresholds['train_subjects']}명 "
                     f"반복 허용폭 {_number(thresholds['single_capture_tolerance'])}")
        else:
            basis = f"경계는 학습군 {thresholds['train_subjects']}명의 분위수 {thresholds['quantiles']}"
        metric = feature['metric']
        return [f"자동 랜드마크 측정 참조 레이블: {_interval(metric, cuts, i)}. {basis}. "
                f"경계 가드: {thresholds['boundary_guard']}. 근거: {provenance}." + SERVICE_CNN
                for i in range(len(classes))]
    if thresholds.get('status') == 'fitted':
        _require(len(classes) == 3, f'{key}: lower/upper thresholds need three classes')
        cuts = [thresholds['lower'], thresholds['upper']]
        _require(cuts[0] < cuts[1], f'{key}: degenerate thresholds')
        model = feature.get('provenance', {}).get('model', '기록 없음')
        return [f"자동 분할 측정 참조 레이블: {_interval('r', cuts, i)}, r = {feature['metric']}. "
                f"경계는 학습군 유효 {thresholds['eligible_train_n']}명의 분위수 {thresholds['quantiles']}. "
                f"분할 모델: {model}. 근거: {provenance}." + SERVICE_CNN for i in range(len(classes))]
    if thresholds.get('status') == 'fixed_algorithm':
        policy = feature.get('annotation_policy') or {}
        _require(classes == ['absent', 'present'] and policy.get('positive') and policy.get('negative'),
                 f'{key}: fixed algorithm policy incomplete')
        scope = f"범위: {thresholds.get('scope')}. 상충 시: {policy.get('conflict')}. 의미: {policy.get('meaning')}."
        return [f"고정 알고리즘 참조 레이블(음성 규칙): {policy['negative']}. {scope}" + SERVICE_CNN,
                f"고정 알고리즘 참조 레이블(양성 규칙): {policy['positive']}. {scope}" + SERVICE_CNN]
    raise ValueError(f'{key}: unsupported threshold definition')


def build_metadata(registry, landmark, depth_config):
    """Return {label_id: row} for every registry class; raises if definitions disagree."""
    _require(registry.get('human_review_count') == 0, 'registry human_review_count must remain 0')
    rows = {}
    for key, feature in registry['features'].items():
        _require(feature.get('kind') == 'classification', f'{key}: non-classification feature in registry.features')
        criteria = _classification_criteria(key, feature, landmark)
        for cls, name, text in zip(feature['classes'], feature['classes_ko'], criteria, strict=True):
            rows[f'{key}:{cls}'] = {'region': feature.get('region', feature['part']),
                                   'label_name': f"{feature['ko']}: {name}", 'criteria': text,
                                   'measurement_indicator': None, 'measurement_method': CNN}
    depth = depth_config['features']
    _require(set(depth) == set(registry['depth_features']), 'depth_config/registry depth feature set mismatch')
    quality = depth_config['quality']
    for key, feature in registry['depth_features'].items():
        c = depth[key]
        cuts = [c['lower'], c['upper']]
        _require(cuts[0] < cuts[1] and len(feature['classes']) == 3, f'{key}: invalid depth boundaries')
        for i, (cls, name) in enumerate(zip(feature['classes'], feature['classes_ko'], strict=True)):
            rows[f'{key}:{cls}'] = {
                'region': feature['part'], 'label_name': f"{feature['ko']}: {name}",
                'criteria': (f"VGGT 3장 복원 상대 비율 r({key}): {_interval('r', cuts, i)}. 경계는 {c['n']}명 분포의 분위수 "
                             f"{c['quantiles']}({depth_config['boundary_scope']}). 경계점수는 Phi((r-b)/sigma), "
                             f"sigma={_number(c['sigma_repeat_train'])}(학습 반복 {c['repeat_train_people']}명)이며 정답확률이 아니다. "
                             f"품질: 깊이 신뢰도 ≥ {_number(quality['min_confidence'])}, 시점 간 오차 ≤ "
                             f"{_number(quality['max_cross_view_error'])}({quality['fit_scope']}). 실제 길이 단위가 아니다."),
                'measurement_indicator': f'{key} (unit: ratio)', 'measurement_method': DEPTH}
    return rows


def load_inputs(paths=None):
    paths = {**INPUTS, **(paths or {})}
    documents, hashes = {}, {}
    for name, path in paths.items():
        raw = Path(path).read_bytes()
        documents[name] = json.loads(raw.decode('utf-8-sig'))
        hashes[name] = hashlib.sha256(raw).hexdigest()
    rows = build_metadata(documents['training_registry'], documents['landmark_labels'], documents['depth_config'])
    revision = hashlib.sha256(canonical_json({'inputs': hashes, 'labels': rows}).encode('utf-8')).hexdigest()
    return rows, revision, hashes


def _eligible_without_metadata(conn):
    return [r[0] for r in conn.execute("""SELECT DISTINCT lp.label_id FROM face_release.label_passage lp
        JOIN face_release.passage p USING (passage_id) JOIN face_release.label l USING (label_id)
        WHERE p.service_eligible AND l.metadata_revision IS NULL ORDER BY 1""").fetchall()]


def sync(dsn, rows, revision):
    """One transaction: upsert changed rows, clear stale metadata, require eligible coverage."""
    with psycopg.connect(dsn) as conn:
        conn.execute('SELECT pg_advisory_xact_lock(%s)', (LOCK_ID,))
        current = {r[0]: r[1:] for r in conn.execute(
            'SELECT label_id,' + ','.join(COLUMNS) + ',metadata_revision FROM face_release.label').fetchall()}
        wanted = {k: tuple(v[c] for c in COLUMNS) + (revision,) for k, v in rows.items()}
        changed = {k: v for k, v in wanted.items() if current.get(k) != v}
        with conn.cursor() as cursor:
            cursor.executemany(
                'INSERT INTO face_release.label (label_id,' + ','.join(COLUMNS) + ',metadata_revision) '
                'VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (label_id) DO UPDATE SET '
                + ','.join(f'{c}=EXCLUDED.{c}' for c in (*COLUMNS, 'metadata_revision')),
                [(k, *v) for k, v in sorted(changed.items())])
        stale = sorted(k for k, v in current.items() if k not in wanted and v[-1] is not None)
        if stale:
            conn.execute('UPDATE face_release.label SET ' + ','.join(f'{c}=NULL' for c in (*COLUMNS, 'metadata_revision'))
                         + ' WHERE label_id = ANY(%s)', (stale,))
        missing = _eligible_without_metadata(conn)
        _require(not missing, 'Service-eligible labels lack frozen metadata: ' + ', '.join(missing))
        unmapped = sorted(set(current) - set(wanted))
    return {'metadata_revision': revision, 'registry_labels': len(rows),
            'inserted': sum(k not in current for k in changed), 'updated': sum(k in current for k in changed),
            'unchanged': len(rows) - len(changed), 'stale_cleared': len(stale), 'unmapped_db_labels': unmapped}


def check(dsn, rows, revision):
    """Read-only comparison; works with the service reader account."""
    with psycopg.connect(dsn) as conn:
        conn.execute('SET TRANSACTION READ ONLY')
        current = {r[0]: r[1:] for r in conn.execute(
            'SELECT label_id,' + ','.join(COLUMNS) + ',metadata_revision FROM face_release.label').fetchall()}
        missing = _eligible_without_metadata(conn)
    wanted = {k: tuple(v[c] for c in COLUMNS) + (revision,) for k, v in rows.items()}
    differing = sorted(k for k, v in wanted.items() if current.get(k) != v)
    return {'metadata_revision': revision, 'matches': not differing and not missing,
            'differing_labels': differing, 'eligible_without_metadata': missing}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('preview', 'sync', 'check'))
    for name in INPUTS:
        parser.add_argument('--' + name.replace('_', '-'), type=Path)
    args = parser.parse_args(argv)
    rows, revision, hashes = load_inputs({k: getattr(args, k) for k in INPUTS if getattr(args, k)})
    try:
        if args.command == 'preview':
            methods = {m: sum(r['measurement_method'] == m for r in rows.values()) for m in (CNN, DEPTH)}
            result = {'metadata_revision': revision, 'inputs_sha256': hashes, 'labels': len(rows), 'methods': methods}
        elif args.command == 'sync':
            result = {**sync(database_dsn(admin=True), rows, revision), 'inputs_sha256': hashes}
        else:
            result = check(database_dsn(), rows, revision)
    except psycopg.Error as exc:
        print(f'PostgreSQL operation failed: {type(exc).__name__} (SQLSTATE {exc.sqlstate}); check configuration', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get('matches', True) else 3


if __name__ == '__main__':
    raise SystemExit(main())
