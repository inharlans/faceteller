-- Reference material only: no users, photographs, measurements or analysis history.
CREATE TABLE face_release.literature (
    literature_id TEXT PRIMARY KEY CHECK (btrim(literature_id) <> ''),
    title TEXT NOT NULL CHECK (btrim(title) <> ''),
    document JSONB NOT NULL CHECK (jsonb_typeof(document) = 'object')
);
CREATE TABLE face_release.passage (
    passage_id TEXT PRIMARY KEY CHECK (btrim(passage_id) <> ''),
    literature_id TEXT NOT NULL REFERENCES face_release.literature(literature_id),
    original_text TEXT NOT NULL,
    translation TEXT NOT NULL,
    commentary TEXT NOT NULL,
    source JSONB NOT NULL CHECK (jsonb_typeof(source) = 'object'),
    literature_priority INTEGER NOT NULL,
    passage_order INTEGER NOT NULL,
    annotation_status TEXT NOT NULL CHECK (btrim(annotation_status) <> ''),
    service_eligible BOOLEAN NOT NULL,
    document JSONB NOT NULL CHECK (jsonb_typeof(document) = 'object'),
    CHECK (NOT service_eligible OR (
        btrim(original_text) <> '' AND btrim(translation) <> '' AND btrim(commentary) <> ''
        AND COALESCE(btrim(source->>'title'), '') <> ''
        AND COALESCE(btrim(source->>'url'), '') <> ''
        AND COALESCE(btrim(source->>'chapter'), '') <> ''
        AND COALESCE(btrim(source->>'locator'), '') <> ''))
);
CREATE TABLE face_release.label (
    label_id TEXT PRIMARY KEY CHECK (label_id ~ '^[^[:space:]:]+:[^[:space:]:]+$')
);
CREATE TABLE face_release.label_passage (
    passage_id TEXT NOT NULL REFERENCES face_release.passage(passage_id),
    label_id TEXT NOT NULL REFERENCES face_release.label(label_id),
    PRIMARY KEY (passage_id, label_id)
);
CREATE INDEX label_passage_label_idx ON face_release.label_passage(label_id, passage_id);
CREATE INDEX passage_eligible_idx ON face_release.passage(literature_priority, passage_order, passage_id)
    WHERE service_eligible;
-- Preserve original complete input, including excluded records and extra metadata.
-- One immutable document per canonical JSON hash, not one row per re-import.
CREATE TABLE face_release.corpus_import (
    sha256 TEXT PRIMARY KEY CHECK (length(sha256) = 64),
    schema_version TEXT NOT NULL,
    document JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
