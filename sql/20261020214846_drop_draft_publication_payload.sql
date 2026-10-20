-- The site reads the section columns, so the combined payload and its
-- agreement check retire; dropping the column drops the check with it.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE cfb.draft_publications
    DROP CONSTRAINT draft_publications_payload_matches_sections,
    DROP COLUMN payload;
NOTIFY pgrst, 'reload schema';
COMMIT;
