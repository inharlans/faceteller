CREATE TABLE face_release.gwansang_catalog (
    singleton SMALLINT PRIMARY KEY CHECK (singleton = 1),
    canonical_sha256 TEXT NOT NULL CHECK (canonical_sha256 ~ '^[0-9a-f]{64}$'),
    document JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
