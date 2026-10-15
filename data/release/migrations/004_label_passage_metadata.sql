-- Additive design 4.3 columns. Existing natural TEXT identifiers are intentional:
-- label_id is the canonical feature:class key used by runtime/API/corpus, and
-- passage_id is the stable corpus key. Nothing is renumbered to BIGINT.
-- Label metadata is filled by scripts/release/sync_label_metadata.py from the frozen
-- training registry/landmark targets/depth config. Columns stay NULL until synced.
ALTER TABLE face_release.label
    ADD COLUMN region TEXT CHECK (region IS NULL OR btrim(region) <> ''),
    ADD COLUMN label_name TEXT CHECK (label_name IS NULL OR btrim(label_name) <> ''),
    ADD COLUMN criteria TEXT CHECK (criteria IS NULL OR btrim(criteria) <> ''),
    ADD COLUMN measurement_indicator TEXT CHECK (measurement_indicator IS NULL OR btrim(measurement_indicator) <> ''),
    ADD COLUMN measurement_method TEXT CHECK (measurement_method IS NULL OR measurement_method IN
        ('cnn_image_classification', 'vggt_multiview_depth_measurement')),
    ADD COLUMN metadata_revision TEXT CHECK (metadata_revision IS NULL OR metadata_revision ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT label_metadata_all_or_none CHECK (
        (region IS NULL AND label_name IS NULL AND criteria IS NULL AND measurement_indicator IS NULL
         AND measurement_method IS NULL AND metadata_revision IS NULL)
        OR (region IS NOT NULL AND label_name IS NOT NULL AND criteria IS NOT NULL
            AND measurement_method IS NOT NULL AND metadata_revision IS NOT NULL
            AND (measurement_method = 'vggt_multiview_depth_measurement') = (measurement_indicator IS NOT NULL)));
COMMENT ON COLUMN face_release.label.region IS 'Observation region key returned by the API (eye, brow, nose, ...)';
COMMENT ON COLUMN face_release.label.measurement_indicator IS 'Depth measurement indicator (ratio); NULL for CNN classification labels';
COMMENT ON COLUMN face_release.label.metadata_revision IS 'SHA256 of the synced metadata and its frozen input file hashes';

-- Derived read-only views of the archived source JSON; importer never writes them.
-- passage_no is the existing precise locator (section/char span/scan page). The
-- historical texts have no uniform original article numbers, so none are invented.
ALTER TABLE face_release.passage
    ADD COLUMN chapter TEXT GENERATED ALWAYS AS (source->>'chapter') STORED,
    ADD COLUMN passage_no TEXT GENERATED ALWAYS AS (source->>'locator') STORED;
COMMENT ON COLUMN face_release.passage.passage_no IS 'Generated from source.locator; not a renumbered article number';
