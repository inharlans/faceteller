"""Import the supplied grouped archive without modifying trained models or old sources."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg
from psycopg.types.json import Jsonb
from facework.release_gwansang import GwansangCatalog
from facework.release_repository import ReleaseRepository, database_dsn
from scripts.release.setup_database import migrate, grant_reader, MIGRATIONS, LOCK_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    catalog = GwansangCatalog.load(ROOT)
    if not args.check:
        dsn = database_dsn(admin=True)
        migrate(dsn, MIGRATIONS)
        with psycopg.connect(dsn) as conn:
            conn.execute('SELECT pg_advisory_xact_lock(%s)', (LOCK_ID,))
            conn.execute('''INSERT INTO face_release.gwansang_catalog(singleton,canonical_sha256,document)
                VALUES (1,%s,%s) ON CONFLICT(singleton) DO UPDATE SET
                canonical_sha256=EXCLUDED.canonical_sha256, document=EXCLUDED.document, imported_at=now()''',
                (catalog.identity['gwansang_catalog_sha256'], Jsonb(catalog.document)))
        grant_reader(dsn)
    with ReleaseRepository().snapshot() as snapshot:
        catalog.verify_snapshot(snapshot)
    print(json.dumps({'verified': True, **catalog.identity}, ensure_ascii=False))


if __name__ == '__main__':
    main()
