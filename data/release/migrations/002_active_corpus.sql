-- Active projection, not the newest historical import timestamp. Re-importing an
-- older document must point back to its hash. Only safe reference metadata here.
CREATE TABLE face_release.active_corpus (
    singleton SMALLINT PRIMARY KEY CHECK (singleton = 1),
    version TEXT NOT NULL CHECK (btrim(version) <> ''),
    canonical_sha256 TEXT NOT NULL REFERENCES face_release.corpus_import(sha256)
        CHECK (canonical_sha256 ~ '^[0-9a-f]{64}$')
);
-- Do not infer an active revision from historical timestamps on an existing DB.
-- Re-import the intended complete corpus once to establish verified provenance.
