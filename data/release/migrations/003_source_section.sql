-- Private source-text archive for the active complete corpus revision. These rows
-- are never runtime candidates; previous revisions remain in corpus_import.document.
CREATE TABLE face_release.source_section (
    section_id TEXT PRIMARY KEY CHECK (btrim(section_id) <> ''),
    literature_id TEXT NOT NULL REFERENCES face_release.literature(literature_id),
    page_id TEXT NOT NULL CHECK (btrim(page_id) <> ''),
    heading TEXT NOT NULL,
    occurrence INTEGER NOT NULL CHECK (occurrence > 0),
    original_text TEXT NOT NULL CHECK (btrim(original_text) <> ''),
    text_sha256 TEXT NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$')
        CHECK (text_sha256 = encode(sha256(convert_to(original_text, 'UTF8')), 'hex')),
    source JSONB NOT NULL CHECK (jsonb_typeof(source) = 'object'),
    annotation_status TEXT NOT NULL CHECK (annotation_status = 'source_archive_unreviewed'),
    service_eligible BOOLEAN NOT NULL CHECK (service_eligible = false),
    corpus_sha256 TEXT NOT NULL REFERENCES face_release.corpus_import(sha256),
    document JSONB NOT NULL CHECK (jsonb_typeof(document) = 'object')
);
REVOKE ALL ON face_release.source_section FROM PUBLIC;
-- grant_reader uses an explicit allowlist and does NOT grant access to this table.
