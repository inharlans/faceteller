"""PostgreSQL reference DB administration. No inference, GPU, or face data.

Windows quick start:
  .venv/Scripts/python.exe scripts/release/setup_database.py local-init
  .venv/Scripts/python.exe scripts/release/setup_database.py import --corpus FILE
  .venv/Scripts/python.exe scripts/release/setup_database.py sync-labels
  .venv/Scripts/python.exe scripts/release/setup_database.py status

Existing managed server: set FACETELLER_ADMIN_DATABASE_URL (created database,
owner connection), FACETELLER_SERVICE_PASSWORD; run provision, migrate, import.
Passwords are never printed. local-init reuses installed binaries, creates a
separate loopback-only cluster, and is also the restart command after reboot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo, conninfo_to_dict
from psycopg.types.json import Jsonb

from facework.release_repository import database_dsn, local_config_path, validate_labels

MIGRATIONS = ROOT / "data" / "release" / "migrations"
SERVICE_ROLE = "faceteller_reference_reader"
LOCK_ID = 202610140431


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _string(value, name, *, blank=False):
    if not isinstance(value, str) or (not blank and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if blank else 'nonempty '}string")
    return value


def _sha256(value, name):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256 hex digest')
    return value


def validate_source_sections(sections, source_ids):
    """Strict optional archive contract; no translation or runtime eligibility added."""
    if not isinstance(sections,list):
        raise ValueError('source_sections must be an array when present')
    seen=set()
    for section in sections:
        if not isinstance(section,dict):
            raise ValueError('Every source section must be an object')
        key=_string(section.get('section_id'),'section_id')
        if key in seen: raise ValueError(f'Duplicate section_id: {key}')
        seen.add(key)
        literature=_string(section.get('literature_id'),f'{key}.literature_id')
        if literature not in source_ids: raise ValueError(f'{key}: unknown literature_id')
        page=_string(section.get('page_id'),f'{key}.page_id')
        _string(section.get('heading'),f'{key}.heading',blank=True)
        occurrence=section.get('occurrence')
        if type(occurrence) is not int or not 1<=occurrence<2**31:
            raise ValueError(f'{key}.occurrence must be a positive int32')
        prefix=f'{page}:{occurrence}:'
        index=key[len(prefix):] if key.startswith(prefix) else ''
        if not index.isascii() or not index.isdecimal() or int(index)<1 or str(int(index))!=index:
            raise ValueError(f'{key}: section_id must be page_id:occurrence:positive_section_index')
        text=_string(section.get('original_text'),f'{key}.original_text')
        digest=_sha256(section.get('text_sha256'),f'{key}.text_sha256')
        if digest!=hashlib.sha256(text.encode('utf-8')).hexdigest():
            raise ValueError(f'{key}: original_text SHA256 mismatch')
        if section.get('annotation_status')!='source_archive_unreviewed' or section.get('service_eligible') is not False:
            raise ValueError(f'{key}: archive must be unreviewed and service_eligible=false')
        source=section.get('source')
        if not isinstance(source,dict): raise ValueError(f'{key}: source must be an object')
        for field in ('title','url','chapter','snapshot_file'):
            _string(source.get(field),f'{key}.source.{field}')
        _sha256(source.get('snapshot_sha256'),f'{key}.source.snapshot_sha256')
        revision=source.get('wikisource_revid')
        if revision is not None:
            if type(revision) is not int or revision<=0:
                raise ValueError(f'{key}: wikisource_revid must be a positive integer')
            if source.get('scan_page') is not None:
                _string(source['scan_page'],f'{key}.source.scan_page')
        else:
            _string(source.get('scan_page'),f'{key}.source.scan_page')
            _string(source.get('scan_source_id'),f'{key}.source.scan_source_id')
            _sha256(source.get('transcription_sha256'),f'{key}.source.transcription_sha256')
            files=source.get('transcription_files')
            if not isinstance(files,list) or not files:
                raise ValueError(f'{key}: transcription_files must be a nonempty array')
            for file in files: _string(file,f'{key}.source.transcription_files')
            first,last=source.get('pdf_page'),source.get('pdf_page_end')
            if type(first) is not int or type(last) is not int or not 1<=first<=last:
                raise ValueError(f'{key}: invalid scan PDF page range')


def validate_corpus(document: dict) -> tuple[list[dict], list[dict]]:
    if not isinstance(document, dict):
        raise ValueError("Corpus must be a JSON object")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, (str, int)) or not str(version).strip():
        raise ValueError("schema_version must be a nonempty string or integer")
    sources, passages = document.get("sources"), document.get("passages")
    if not isinstance(sources, list) or not isinstance(passages, list):
        raise ValueError("sources and passages must be arrays")
    source_ids, passage_ids, prepared = set(), set(), []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every source must be an object")
        key = _string(source.get("literature_id"), "source.literature_id")
        _string(source.get("title"), "source.title")
        if key in source_ids:
            raise ValueError(f"Duplicate literature_id: {key}")
        source_ids.add(key)
    for raw in passages:
        if not isinstance(raw, dict):
            raise ValueError("Every passage must be an object")
        p = dict(raw)
        key = _string(p.get("passage_id"), "passage_id")
        if key in passage_ids:
            raise ValueError(f"Duplicate passage_id: {key}")
        passage_ids.add(key)
        literature = _string(p.get("literature_id"), f"{key}.literature_id")
        if literature not in source_ids:
            raise ValueError(f"{key}: literature_id is not present in sources")
        eligible = p.get("service_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(f"{key}: service_eligible must be boolean")
        _string(p.get("annotation_status"), f"{key}.annotation_status")
        for field in ("original_text", "translation", "commentary"):
            p[field] = _string(p.get(field, ""), f"{key}.{field}", blank=not eligible)
        if not isinstance(p.get("source"), dict):
            raise ValueError(f"{key}: source must be an object")
        if eligible:
            for field in ("title", "url", "chapter", "locator"):
                _string(p["source"].get(field), f"{key}.source.{field}")
        labels = p.get("required_labels")
        if not isinstance(labels, list):
            raise ValueError(f"{key}: required_labels must be an array")
        p["required_labels"] = validate_labels(labels)
        if eligible and not labels:
            raise ValueError(f"{key}: an eligible passage needs at least one required label")
        features = [label.split(":", 1)[0] for label in labels]
        if eligible and len(set(features)) != len(features):
            raise ValueError(f"{key}: eligible AND rule has contradictory classes for one feature")
        for field in ("literature_priority", "passage_order"):
            n = p.get(field)
            if isinstance(n, bool) or not isinstance(n, int) or not -(2**31) <= n < 2**31:
                raise ValueError(f"{key}.{field} must be an int32")
        prepared.append(p)
    validate_source_sections(document.get('source_sections',[]),source_ids)
    canonical_json(document)  # Reject non-finite / non-JSON metadata before opening DB.
    return sources, prepared


def migrate(dsn: str, directory: Path = MIGRATIONS) -> list[str]:
    files = sorted(directory.glob("[0-9]*_*.sql"))
    if not files:
        raise ValueError("No migrations found")
    applied = []
    with psycopg.connect(dsn) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        conn.execute("CREATE SCHEMA IF NOT EXISTS face_release")
        conn.execute("REVOKE ALL ON SCHEMA face_release FROM PUBLIC")
        conn.execute("""CREATE TABLE IF NOT EXISTS face_release.schema_migration (
            filename TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        existing = dict(conn.execute("SELECT filename, sha256 FROM face_release.schema_migration"))
        if set(existing) - {p.name for p in files}:
            raise ValueError("Database contains migrations missing from this checkout")
        for path in files:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if path.name in existing:
                if existing[path.name] != digest:
                    raise ValueError(f"Migration checksum mismatch: {path.name}")
                continue
            if existing and path.name < max(existing):
                raise ValueError(f"Out-of-order migration: {path.name}")
            conn.execute(raw.decode("utf-8"))
            conn.execute("INSERT INTO face_release.schema_migration(filename,sha256) VALUES (%s,%s)",
                         (path.name, digest))
            applied.append(path.name)
    return applied


def grant_reader(dsn: str, role: str = SERVICE_ROLE, password: str | None = None):
    """Provision a dedicated, non-owner login; ACLs enforce read-only even if GUC reset."""
    with psycopg.connect(dsn) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone()
        if not exists:
            if not password:
                raise ValueError("A service password is required to create the reader role")
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
        # Never silently strip memberships/ownership from a pre-existing shared account.
        unsafe = conn.execute("""SELECT r.rolsuper OR r.rolcreatedb OR r.rolcreaterole
            OR r.rolreplication OR r.rolbypassrls
            OR EXISTS(SELECT 1 FROM pg_auth_members m WHERE m.member=r.oid)
            OR EXISTS(SELECT 1 FROM pg_class c WHERE c.relowner=r.oid)
            OR EXISTS(SELECT 1 FROM pg_namespace n WHERE n.nspowner=r.oid)
            OR EXISTS(SELECT 1 FROM pg_database d WHERE d.datdba=r.oid)
            FROM pg_roles r WHERE r.rolname=%s""", (role,)).fetchone()[0]
        if unsafe:
            raise ValueError("Reader role must be dedicated, unprivileged, and own no objects")
        if password:
            conn.execute(sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
        conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(role)))
        db = conn.execute("SELECT current_database()").fetchone()[0]
        # This is a dedicated application DB; public TEMP/CREATE would break read-only.
        conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(db)))
        conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM {}").format(sql.Identifier(db), sql.Identifier(role)))
        conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db), sql.Identifier(role)))
        conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        for schema in ("public", "face_release"):
            conn.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM {}").format(sql.Identifier(schema), sql.Identifier(role)))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA face_release TO {}").format(sql.Identifier(role)))
        conn.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA face_release FROM {}, PUBLIC").format(sql.Identifier(role)))
        conn.execute(sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA face_release FROM {}, PUBLIC").format(sql.Identifier(role)))
        for table in ("literature", "passage", "label", "label_passage", "active_corpus"):
            conn.execute(sql.SQL("GRANT SELECT ON face_release.{} TO {}").format(sql.Identifier(table), sql.Identifier(role)))


def import_corpus(dsn: str, document: dict) -> dict:
    """Atomic complete-corpus snapshot. Keep absent records but retire their eligibility.

    Original inputs/revisions remain in corpus_import. Reimporting the same JSON
    restores its live projection without adding audit rows or duplicate mappings.
    """
    sources, passages = validate_corpus(document)
    sections = document.get('source_sections',[])
    digest = hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()
    with psycopg.connect(dsn) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        for s in sources:
            conn.execute("""INSERT INTO face_release.literature (literature_id,title,document) VALUES (%s,%s,%s)
                ON CONFLICT (literature_id) DO UPDATE SET title=EXCLUDED.title, document=EXCLUDED.document""",
                         (s["literature_id"], s["title"], Jsonb(s)))
        conn.execute("UPDATE face_release.passage SET service_eligible=false WHERE service_eligible")
        for p, raw in zip(passages, document["passages"]):
            conn.execute("""INSERT INTO face_release.passage
                (passage_id,literature_id,original_text,translation,commentary,source,
                 literature_priority,passage_order,annotation_status,service_eligible,document)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (passage_id) DO UPDATE SET literature_id=EXCLUDED.literature_id,
                original_text=EXCLUDED.original_text,translation=EXCLUDED.translation,
                commentary=EXCLUDED.commentary,source=EXCLUDED.source,
                literature_priority=EXCLUDED.literature_priority,passage_order=EXCLUDED.passage_order,
                annotation_status=EXCLUDED.annotation_status,service_eligible=EXCLUDED.service_eligible,
                document=EXCLUDED.document""",
                         (p["passage_id"],p["literature_id"],p["original_text"],p["translation"],p["commentary"],
                          Jsonb(p["source"]),p["literature_priority"],p["passage_order"],p["annotation_status"],
                          p["service_eligible"],Jsonb(raw)))
            conn.execute("DELETE FROM face_release.label_passage WHERE passage_id=%s", (p["passage_id"],))
            for label in p["required_labels"]:
                # Explicit columns: metadata (migration004) is owned by sync_label_metadata.py
                # and must survive corpus re-imports untouched.
                conn.execute("INSERT INTO face_release.label (label_id) VALUES (%s) ON CONFLICT DO NOTHING", (label,))
                conn.execute("INSERT INTO face_release.label_passage (passage_id,label_id) VALUES (%s,%s)",
                             (p["passage_id"],label))
        conn.execute("""INSERT INTO face_release.corpus_import(sha256,schema_version,document)
                     VALUES (%s,%s,%s) ON CONFLICT DO NOTHING""", (digest,str(document["schema_version"]),Jsonb(document)))
        # This private table is the active revision only. An older revision, or a
        # legacy snapshot lacking source_sections, must remove obsolete live rows.
        # The immutable corpus_import JSON retains every previously imported text.
        conn.execute('DELETE FROM face_release.source_section')
        with conn.cursor() as cursor:
            cursor.executemany("""INSERT INTO face_release.source_section
                (section_id,literature_id,page_id,heading,occurrence,original_text,text_sha256,
                 source,annotation_status,service_eligible,corpus_sha256,document)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(s['section_id'],s['literature_id'],s['page_id'],s['heading'],s['occurrence'],
                  s['original_text'],s['text_sha256'],Jsonb(s['source']),s['annotation_status'],
                  s['service_eligible'],digest,Jsonb(s)) for s in sections])
        conn.execute("""INSERT INTO face_release.active_corpus(singleton,version,canonical_sha256)
                     VALUES (1,%s,%s) ON CONFLICT (singleton) DO UPDATE
                     SET version=EXCLUDED.version, canonical_sha256=EXCLUDED.canonical_sha256""",
                     (str(document['schema_version']),digest))
    return {"sha256": digest, "sources": len(sources), "passages": len(passages),
            "source_sections": len(sections),
            "eligible": sum(p["service_eligible"] for p in passages)}


def _labels_with_metadata(conn):
    # Present after migration004; None on an older schema rather than failing status.
    if not conn.execute("""SELECT 1 FROM information_schema.columns WHERE table_schema='face_release'
            AND table_name='label' AND column_name='metadata_revision'""").fetchone():
        return None
    return conn.execute("SELECT count(*) FROM face_release.label WHERE metadata_revision IS NOT NULL").fetchone()[0]


def status(dsn: str) -> dict:
    with psycopg.connect(dsn) as conn:
        active=conn.execute("SELECT canonical_sha256 FROM face_release.active_corpus WHERE singleton=1").fetchone()
        return {"database": conn.execute("SELECT current_database()").fetchone()[0],
                "corpus_sha256": active[0] if active else None,
                "postgresql": conn.info.server_version,
                "sources": conn.execute("SELECT count(*) FROM face_release.literature").fetchone()[0],
                "passages": conn.execute("SELECT count(*) FROM face_release.passage").fetchone()[0],
                "source_sections": conn.execute("SELECT count(*) FROM face_release.source_section").fetchone()[0],
                "eligible": conn.execute("SELECT count(*) FROM face_release.passage WHERE service_eligible").fetchone()[0],
                "labels": conn.execute("SELECT count(*) FROM face_release.label").fetchone()[0],
                "labels_with_metadata": _labels_with_metadata(conn),
                "imports": conn.execute("SELECT count(*) FROM face_release.corpus_import").fetchone()[0]}


def _run(args):
    # pg_ctl starts a persistent child on Windows: PIPEs can remain open in that
    # child and hang communicate() after pg_ctl exits. Use file handles instead.
    with tempfile.TemporaryFile() as output:
        result = subprocess.run([str(a) for a in args], stdin=subprocess.DEVNULL,
                                stdout=output, stderr=output, timeout=45)
        if result.returncode:
            output.seek(0)
            raise RuntimeError(f"{Path(str(args[0])).name} failed: " + output.read().decode(errors="replace"))


def local_init(bin_dir: Path | None = None) -> dict:
    if os.name != "nt":
        raise RuntimeError("local-init currently supports Windows; use an existing PostgreSQL DSN on other systems")
    path = local_config_path()
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    # Private user directory, before writing credentials. No credentials in repo.
    identity = subprocess.check_output(["whoami.exe"], text=True).strip()
    _run(["icacls.exe",root,"/inheritance:r","/grant:r",f"{identity}:(OI)(CI)F","SYSTEM:(OI)(CI)F"])
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        binaries = Path(config["bin_dir"])
    else:
        candidates = list(Path("C:/Program Files/PostgreSQL").glob("*/bin"))
        binaries = bin_dir or (max(candidates, key=lambda p: tuple(int(n) for n in p.parent.name.split('.'))) if candidates else None)
        if binaries is None or not (binaries / "initdb.exe").exists():
            raise RuntimeError("Install PostgreSQL or supply --bin-dir")
        admin_password, reader_password = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        common = dict(host="127.0.0.1",port=55432,dbname="faceteller_release",connect_timeout=5)
        config = {"bin_dir":str(binaries),"data_dir":str(root / "data"),
                  "admin_dsn":make_conninfo(**common,user="faceteller_admin",password=admin_password),
                  "service_dsn":make_conninfo(**common,user=SERVICE_ROLE,password=reader_password)}
        path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    data = Path(config["data_dir"])
    if not (data / "PG_VERSION").exists():
        pwfile = root / "init-password.tmp"
        pwfile.write_text(conninfo_to_dict(config["admin_dsn"])["password"] + "\n",encoding="utf-8")
        try:
            _run([binaries/"initdb.exe","-D",data,"-U","faceteller_admin","--encoding=UTF8",
                  "--locale=C","--auth=scram-sha-256",f"--pwfile={pwfile}"])
        finally:
            pwfile.unlink(missing_ok=True)
        with (data / "postgresql.conf").open("a",encoding="utf-8") as f:
            f.write("\n# FaceTeller dedicated local reference database\nlisten_addresses='127.0.0.1'\n"
                    "port=55432\nshared_buffers='32MB'\nmax_connections=20\n"
                    "log_statement='none'\nlog_min_error_statement='panic'\n")
    running = subprocess.run([str(binaries/"pg_ctl.exe"),"-D",str(data),"status"],capture_output=True,timeout=10)
    if running.returncode:
        _run([binaries/"pg_ctl.exe","-D",data,"-l",root/"postgres.log","-w","start"])
    with psycopg.connect(config["admin_dsn"],dbname="postgres",autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname='faceteller_release'").fetchone():
            conn.execute("CREATE DATABASE faceteller_release")
    migrate(config["admin_dsn"])
    grant_reader(config["admin_dsn"],password=conninfo_to_dict(config["service_dsn"])["password"])
    return {"config_path":str(path),"port":55432,**status(config["admin_dsn"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("local-init","migrate","provision","import","sync-labels","status"))
    parser.add_argument("--corpus",type=Path)
    parser.add_argument("--bin-dir",type=Path)
    args = parser.parse_args()
    try:
        if args.command == "local-init":
            result = local_init(args.bin_dir)
        else:
            dsn = database_dsn(admin=True)
            if args.command == "migrate":
                result = {"applied":migrate(dsn)}
            elif args.command == "provision":
                migrate(dsn)
                grant_reader(dsn,password=os.environ.get("FACETELLER_SERVICE_PASSWORD"))
                result = {"reader_role":SERVICE_ROLE,"read_only":True}
            elif args.command == "sync-labels":
                from scripts.release.sync_label_metadata import load_inputs, sync
                rows, revision, hashes = load_inputs()
                result = {**sync(dsn, rows, revision), "inputs_sha256": hashes}
            elif args.command == "import":
                if args.corpus is None:
                    parser.error("import requires --corpus (one complete corpus snapshot)")
                result = import_corpus(dsn,json.loads(args.corpus.read_text(encoding="utf-8-sig")))
            else:
                result = status(dsn)
        print(json.dumps(result,ensure_ascii=True,indent=2))
    except psycopg.Error as exc:
        # Connection failures can include credentials from user-supplied DSNs.
        print(f"PostgreSQL operation failed: {type(exc).__name__} (SQLSTATE {exc.sqlstate}); check configuration",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
