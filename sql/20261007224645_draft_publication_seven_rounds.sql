-- Accept the seven-round contract while preserving the current edition.
-- Deploy the frontend reader for both versions before publishing version 2.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE cfb.draft_publications
    DROP CONSTRAINT draft_publications_payload_check,
    ADD CONSTRAINT draft_publications_payload_check
        CHECK (COALESCE(payload->>'schema_version' IN ('1', '2'), false));
COMMIT;
