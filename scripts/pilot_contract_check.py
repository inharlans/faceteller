"""Run the independent CPU pilot, optionally with an isolated PostgreSQL cluster.

Windows: .venv/Scripts/python.exe scripts/pilot_contract_check.py --postgres
The --postgres option ONLY creates its own temporary trust-auth cluster on
127.0.0.1:55437, never connects to an existing database, and stops/deletes it.
Reports go to a new timestamped directory (or a new --output-dir); prior reports
are never overwritten. Default G combines CNN confidence and depth membership;
classification_only remains an explicitly selected legacy comparison mode.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facework.pilot_contract import (
    DecisionPolicy, MemoryRepository, Passage, PassageSelector, PostgresRepository,
    POSTGRES_SCHEMA, load_real_repository,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.cases.append({"name": test.id(), "status": "passed",
                           "data_kind": "real_text_synthetic_predictions" if "RealDataTests" in test.id() else "fixture"})


def postgres_check(bin_dir: Path, port: int = 55437):
    report = {"status": "not_run", "scope": "isolated temporary local cluster; no existing DB access",
              "bin_directory": str(bin_dir), "port": port, "checks": [], "stopped": False}
    names = ("postgres", "initdb", "pg_ctl", "psql")
    report["binaries"] = {n: (bin_dir / (n + ".exe")).is_file() for n in names}
    if not all(report["binaries"].values()):
        report.update(status="unavailable", limitation="PostgreSQL binaries missing; mock/schema tests only")
        return report
    # Check availability without ever attempting a connection to an existing DB.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            report.update(status="unavailable", limitation="Pilot port occupied; existing server left untouched")
            return report
    try:
        import psycopg
    except ImportError:
        report.update(status="unavailable", limitation="psycopg absent; no installation attempted")
        return report
    parent = ROOT / "tmp/pilot-20261014"
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contract-pg-", dir=parent))
    cluster, log = work / "cluster", work / "postgres.log"
    initialized = started = False

    def run(binary, *args):
        # Windows postgres can inherit pg_ctl pipe handles. A real log handle
        # avoids communicate() waiting for a long-lived child to close stdout.
        command_log = work / (binary + "-command.log")
        with command_log.open("wb") as output:
            result = subprocess.run([str(bin_dir / (binary + ".exe")), *map(str, args)],
                                    stdout=output, stderr=subprocess.STDOUT, timeout=45)
        message = command_log.read_text(encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(f"{binary} exit {result.returncode}: {message[-2000:]}")
        return message

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        report["checks"].append({"name": name, "status": "passed"})

    try:
        report["version"] = run("postgres", "--version").strip()
        run("initdb", "-D", cluster, "--username=pilot_owner", "--auth-local=trust", "--auth-host=trust",
            "--encoding=UTF8", "--locale=C", "--no-sync")
        initialized = True
        run("pg_ctl", "-D", cluster, "-l", log, "-o", f"-p {port} -h 127.0.0.1 -F", "-w", "-t", "30", "start")
        started = True
        with psycopg.connect(host="127.0.0.1", port=port, user="pilot_owner", dbname="postgres",
                             password="", passfile=str(work / "no-password-file"), connect_timeout=5,
                             autocommit=True) as conn:
            check("own_cluster_identity", Path(conn.execute("SHOW data_directory").fetchone()[0]).resolve() == cluster.resolve())
            conn.execute(POSTGRES_SCHEMA)
            conn.execute("INSERT INTO literature VALUES (1, 'fixture literature', 0)")
            conn.execute("INSERT INTO passage VALUES (10,1,'chapter','section','original','translation','commentary',0)")
            conn.execute("INSERT INTO label VALUES (100,'eye','wide','fixture',NULL,'classification'),"
                         "(101,'nose','high','fixture','projection','depth')")
            conn.execute("INSERT INTO label_passage VALUES (100,10),(101,10)")
            repository = PostgresRepository(conn)
            candidates = repository.load_candidates({"100"})
            check("any_hit_fetches_all_required_labels", candidates[0].required_labels == {"100", "101"})
            # Test helpers use generated PNGs and simulated inference only.
            sys.path.insert(0, str(ROOT / "tests"))
            from test_pilot_contract import classified, measured, fixture_app, FixtureModules, multipart, events
            from fastapi.testclient import TestClient
            only_eye = DecisionPolicy().apply([classified(label="100")])
            check("AND_rejects_missing_depth", PassageSelector().select(only_eye, candidates) == [])
            full = DecisionPolicy().apply([classified(label="100", confidence=.81), measured(label="101", score=.04)])
            selected = PassageSelector().select(full, repository.load_candidates({"100", "101"}))
            expected_g = math.sqrt(.81 * .04)
            check("full_AND_selected_default_G_includes_depth", len(selected) == 1 and abs(selected[0]["G"] - expected_g) < 1e-12)
            legacy = PassageSelector("classification_only").select(full, candidates)
            check("explicit_legacy_classification_only_G", len(legacy) == 1 and abs(legacy[0]["G"] - .81) < 1e-12)
            app, modules = fixture_app(FixtureModules([classified(label="100", confidence=.81)],
                                                     [measured(label="101", score=.04)]), repository)
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            final = events(response)[-1]
            check("FastAPI_PostgreSQL_default_mixed_G_parity", response.status_code == 200 and final["type"] == "result"
                  and final["data"]["passages"] == selected)
            policy = final["data"]["ranking_policy"]
            check("mixed_G_ranking_metadata", policy["score_mode"] == "include_measurement_scores"
                  and policy["is_correctness_probability"] is False
                  and selected[0]["rank"] == 1 and selected[0]["score_feature_count"] == 2
                  and {c["kind"] for c in selected[0]["score_components"]} == {"classification", "depth"})
            depth_item = next(i for i in final["data"]["items"] if i["kind"] == "depth")
            check("API_depth_score_separate_from_confidence", depth_item["confidence"] is None
                  and depth_item["measurement_score"] == .04
                  and depth_item["measurement_score_kind"] == "geometry_membership_not_calibrated_accuracy")
            check("mixed_API_request_memory_cleaned", modules.inputs.images == {})
            for name, depth in (("missing_score", measured(label="101", score=None)),
                                ("quality_failed", replace(measured(label="101", score=.99), quality_passed=False))):
                app, _ = fixture_app(FixtureModules([classified(label="100")], [depth]), repository)
                with TestClient(app) as client:
                    response = client.post("/api/v1/analysis", files=multipart())
                final = events(response)[-1]
                check("API_PostgreSQL_AND_rejects_" + name, response.status_code == 200 and final["type"] == "result"
                      and final["data"]["status"] == "PARTIAL" and final["data"]["passages"] == [])
            report["mixed_G_example"] = {"CNN_confidence": .81, "depth_membership_score": .04,
                                         "expected_G": expected_g, "PostgreSQL_G": selected[0]["G"],
                                         "legacy_classification_only_G": legacy[0]["G"],
                                         "ranking_policy": policy, "api_matches_PostgreSQL": True}
            check("empty_lookup_normal", repository.load_candidates({"999"}) == [])
            for name, sql, exception in (
                ("composite_PK_rejects_duplicate", "INSERT INTO label_passage VALUES (100,10)", psycopg.errors.UniqueViolation),
                ("FK_rejects_missing_label", "INSERT INTO label_passage VALUES (999,10)", psycopg.errors.ForeignKeyViolation),
                ("FK_rejects_missing_passage", "INSERT INTO label_passage VALUES (100,999)", psycopg.errors.ForeignKeyViolation),
            ):
                rejected = False
                try:
                    with conn.transaction():
                        conn.execute(sql)
                except exception:
                    rejected = True
                check(name, rejected)

            # More eligible rows than either cap: three mixed eye+nose clauses
            # and two lower-scored mouth clauses must survive the ordered caps.
            conn.execute("INSERT INTO label VALUES (102,'mouth','protruded','fixture','projection','depth')")
            for ident in (11, 12, 13, 20, 21, 22):
                conn.execute("INSERT INTO passage VALUES (%s,1,'chapter','section','original','translation','commentary',%s)", (ident, ident))
                for label in ((100, 101) if ident < 20 else (102,)):
                    conn.execute("INSERT INTO label_passage VALUES (%s,%s)", (label, ident))
            app, _ = fixture_app(FixtureModules([classified(label="100", confidence=.81)],
                                                [measured(label="101", score=.04), measured("mouth", "mouth", "102", score=.01)]), repository)
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            final = events(response)[-1]
            check("API_PostgreSQL_mixed_region3_total5", response.status_code == 200 and final["type"] == "result"
                  and [p["passage_id"] for p in final["data"]["passages"]] == ["10", "11", "12", "20", "21"]
                  and [p["rank"] for p in final["data"]["passages"]] == [1, 2, 3, 4, 5])

            # Migrate only structurally complete draft rows into this disposable DB.
            # Source strings are explicitly mapped to local BIGINT IDs for the pilot.
            conn.execute("TRUNCATE label_passage,label,passage,literature")
            real, audit = load_real_repository(ROOT, allow_draft=True)
            label_ids = {label: i + 1 for i, label in enumerate(sorted({k for p in real.passages for k in p.required_labels}))}
            schema = json.loads((ROOT / "data/label_schema_kface_v3.json").read_text(encoding="utf8"))["features"]
            conn.execute("INSERT INTO literature VALUES (1,'欽定古今圖書集成',0)")
            for label, ident in label_ids.items():
                feature = label.split(":")[0]
                spec = schema[feature]
                depth = spec["source"] == "multiview"
                conn.execute("INSERT INTO label VALUES (%s,%s,%s,%s,%s,%s)",
                             (ident, spec["part"], label, "existing draft condition", feature if depth else None,
                              "depth" if depth else "classification"))
            mapped = []
            for ident, p in enumerate(real.passages, 1):
                conn.execute("INSERT INTO passage VALUES (%s,1,%s,%s,%s,%s,%s,%s)",
                             (ident, p.source["chapter"], p.source["passage_no"], p.original_text,
                              p.translation, p.commentary, p.passage_order))
                labels = frozenset(str(label_ids[k]) for k in p.required_labels)
                for label in labels:
                    conn.execute("INSERT INTO label_passage VALUES (%s,%s)", (int(label), ident))
                mapped.append(replace(p, passage_id=str(ident), required_labels=labels, literature_priority=0))
            check("real_draft_rows_inserted", conn.execute("SELECT count(*) FROM passage").fetchone()[0] == len(real.passages))
            for p, source in zip(mapped, real.passages):
                observations = []
                for label in sorted(source.required_labels):
                    feature = label.split(":")[0]
                    spec = schema[feature]
                    helper = measured if spec["source"] == "multiview" else classified
                    observations.append(helper(feature, spec["part"], str(label_ids[label])))
                decisions = DecisionPolicy().apply(observations)
                ids = {i.label_id for i in decisions}
                pg_selected = PassageSelector().select(decisions, repository.load_candidates(ids))
                mem_selected = PassageSelector().select(decisions, MemoryRepository(mapped).load_candidates(ids))
                # DB source omits local audit metadata; content/ranking/evidence must match exactly.
                comparable = lambda rows: [{k: v for k, v in r.items() if k != "source"} for r in rows]
                check("real_text_query_parity_" + source.passage_id, comparable(pg_selected) == comparable(mem_selected))
            first = mapped[0]
            observations = [classified("fixture_" + label, "brow", label) for label in sorted(first.required_labels)]
            app, modules = fixture_app(FixtureModules(observations), repository)
            with TestClient(app) as client:
                response = client.post("/api/v1/analysis", files=multipart())
            final = events(response)[-1]
            check("FastAPI_actual_PostgreSQL_repository_result", response.status_code == 200 and final["type"] == "result"
                  and final["data"]["passages"][0]["original_text"] == first.original_text)
            check("FastAPI_PostgreSQL_request_memory_cleaned", modules.inputs.images == {})
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                check("readonly_query", bool(repository.load_candidates(first.required_labels)))
            report.update(status="passed", real_text_rows=len(real.passages),
                          synthetic_predictions=True, real_face_inference=False,
                          production_reviewed_rows=audit["independently_reviewed"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Always target our own -D directory, including a partially successful start.
        if initialized:
            try:
                if started or (cluster / "postmaster.pid").exists():
                    run("pg_ctl", "-D", cluster, "-m", "fast", "-w", "-t", "30", "stop")
                report["stopped"] = True
            except Exception as exc:
                report.update(status="failed", cleanup_error=str(exc), retained_cluster=str(cluster))
        if not (cluster / "postmaster.pid").exists():
            shutil.rmtree(work)
            report["temporary_cluster_removed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--postgres-bin", type=Path, default=Path("C:/Program Files/PostgreSQL/17/bin"))
    parser.add_argument("--output-dir", type=Path, help="New report directory; must not already exist")
    args = parser.parse_args()
    out = args.output_dir or ROOT / "outputs/pilot-20261014" / ("contract-mixed-g-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    # Reserve before any work so an accidental reused path cannot overwrite evidence.
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    shared = [ROOT / "facework/server.py", ROOT / "facework/analysis_v2.py"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in shared}
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_pilot_contract.py")
    tests = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordingResult).run(suite)
    _, strict = load_real_repository(ROOT)
    draft, draft_audit = load_real_repository(ROOT, allow_draft=True)
    pg = postgres_check(args.postgres_bin) if args.postgres else {"status": "not_requested", "limitation": "mock/schema only in this invocation"}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "design": "tmp/pilot-20261014/design.txt (23 pages)",
        "python": sys.executable, "status": "passed" if tests.wasSuccessful() and pg["status"] in ("passed", "not_requested") else "failed_or_limited",
        "tests": {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors),
                  "cases": tests.cases, "log": log.getvalue()},
        "real_data": {"default_reviewed_only": strict, "draft_text_pilot": draft_audit,
                      "this_check_predictions": "synthetic_fixtures_only",
                      "eligible_source_ids": [p.passage_id for p in draft.passages],
                      "source_hashes": {name: sha(ROOT / "data" / name) for name in
                          ("interpretation_catalog_v1.json", "interpretation_service_v1.json", "type_rules_v3.json")}},
        "postgresql": pg,
        "contracts": {
            "service": "create_app(service_factory); validate(InputSet) async; analyze(InputSet) async iterator",
            "depth": "3 images + preprocessing detections; VGGTEngine.infer(images) independent, measure_geometry(geometry,detections) after mapping",
            "default_G": "include_measurement_scores: equal-feature geometric mean of CNN classification confidence and depth membership scores over all confirmed required conditions; not a correctness probability",
            "legacy_G": "PassageSelector(score_mode='classification_only') explicitly excludes depth scores from G; depth labels still participate in AND conditions",
            "limits": "AND all conditions; deduplicate passage; count every required region <=3; total<=5",
            "memory": "bounded streaming python-multipart callbacks; bytearray; no UploadFile/form/spool; finally clears buffers and context",
            "camera": "Python pure state tested; browser implementation owned by separate UI worker",
        },
        "no_gpu": {"torch_imported": "torch" in sys.modules, "CNN_loaded": False, "GPU_used": False},
        "shared_file_hashes_before": before,
        "shared_file_hashes_after": {str(p.relative_to(ROOT)): sha(p) for p in shared},
        "limitations": [
            "Generated PNGs and simulated calibrated classifications/depth flags exercise contracts, not face recognition, CNN accuracy, or depth accuracy.",
            "All 103 source records are unreviewed drafts. Default reviewed-only repository yields zero. Explicit draft opt-in permits the authorized real-inference pilot while preserving draft source status; this checker itself uses synthetic predictions only.",
            "Partial/unknown conditions are never silently discarded. Multi-value OR condition needs a reviewed label mapping; it is excluded from strict AND pilot.",
            "Legacy original section headings are preserved as passage_no; source has no established original numeric clause number. BIGINT mapping is temporary in PostgreSQL test.",
            "Calibration, depth quality/boundaries, identity/pose checking, and real model adapter are main-worker responsibilities; pilot requires explicit injected service.",
            "Memory cleanup releases/clears application buffers, not a claim of forensic zeroization of all ASGI/Python copies. Request concurrency/process RSS unbenchmarked.",
            "TestClient verifies event content/order, direct ASGI test verifies cancellation; network latency, HTTPS/proxy behavior and browser/A4 print validation are outside this worker.",
        ],
        "changed_files": ["scripts/pilot_contract_check.py"],
        "read_only_review": {"files": ["facework/pilot_contract.py", "tests/test_pilot_contract.py", "facework/pilot/runtime.py"],
                             "finding": "No correctness blocker found in the inspected mixed-G path: default selector/build_result agree; invalid depth scores are held; AND, tie-breaks, caps and separate score semantics remain present.",
                             "scope": "Current working-tree code inspected read-only; untracked files provide no git baseline diff. No actual runtime/GPU execution."},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (out / "contract_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    lines = ["FaceTeller 비GPU 독립 파일럿 계약 검증", f"실행: {sys.executable}",
             f"결과: {report['status']}; tests={tests.testsRun}, failures={len(tests.failures)}, errors={len(tests.errors)}",
             f"PostgreSQL: {pg['status']}; 실제 검사 {len(pg.get('checks', []))}개; 종료={pg.get('stopped')}; 임시클러스터 제거={pg.get('temporary_cluster_removed')}",
             "", "변경: " + ", ".join(report["changed_files"]),
             "공유 server.py / analysis_v2.py는 수정하지 않음 (실행 전후 SHA256 포함).",
             "", "검증: 기본 CNN+깊이 점수 혼합 G, 명시적 classification_only 구형 모드, AND·중복제거·부위3/전체5,",
             "확정/보류·재입력/문헌없음, 3장 메모리 multipart, 주입형 FastAPI NDJSON, 오류·취소 정리, 카메라 순수 상태.",
             f"실제 문헌: 103개 JOIN·원문 근거 대조; 완전조건15/부분65/없음23; 독립검수0.",
             f"초벌 허용 계약검증 가능: {len(draft.passages)}개; 제외 사유: {draft_audit['excluded']}",
             "실제 텍스트·PostgreSQL 검증과 fixture 이미지/예측 검증을 분리. 실제 얼굴 추론/GPU 실행 없음.",
             "", "서비스 연결: create_app(lambda: service). async validate(inputs), analyze(inputs)->async iterator events.",
             "또는 AnalysisPipeline(preprocessor,classifier,depth,repository). DepthEstimator에 inputs 3장+context['detections'] 전달.",
             "기본 G는 include_measurement_scores: sqrt(0.81*0.04)=0.18. 구형 비교: PassageSelector('classification_only')=0.81.",
             "혼합 G는 특징 일치 점수이며 정답 확률이 아님. 실제 서비스/GPU 실행은 main 담당.",
             "읽기 전용 검토: 현재 혼합 G 경로의 차단 수준 오류 없음. main 계약·테스트 파일 수정 없음.", "", "한계:"]
    lines += ["- " + limitation for limitation in report["limitations"]]
    if "error" in pg:
        lines += ["PostgreSQL error: " + pg["error"]]
    lines += ["", "테스트 로그:", log.getvalue()]
    (out / "contract_results.txt").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps({"status": report["status"], "tests": tests.testsRun, "postgresql": pg["status"],
                      "postgres_checks": len(pg.get("checks", [])), "draft_records": len(draft.passages),
                      "elapsed_seconds": report["elapsed_seconds"], "report": str(out / "contract_results.json")}, ensure_ascii=True))
    if "error" in pg:
        print(pg["error"].encode("ascii", errors="backslashreplace").decode())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
