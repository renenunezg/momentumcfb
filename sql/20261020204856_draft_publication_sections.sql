-- Each section gets its own column so a reader detoasts only what it selects.
-- payload stays until the frontend reads the columns; the check keeps both equal,
-- so a publisher that writes only payload fails instead of leaving stale sections.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE cfb.draft_publications
    ADD COLUMN schema_version integer,
    ADD COLUMN board jsonb,
    ADD COLUMN meta jsonb,
    ADD COLUMN players jsonb,
    ADD COLUMN roster jsonb,
    ADD COLUMN history jsonb,
    ADD COLUMN catalog jsonb,
    ALTER COLUMN payload DROP NOT NULL;
UPDATE cfb.draft_publications SET
    schema_version = (payload->>'schema_version')::integer,
    board = payload->'board',
    meta = payload->'meta',
    players = payload->'players',
    roster = payload->'roster',
    history = payload->'history',
    catalog = payload->'catalog';
ALTER TABLE cfb.draft_publications
    ALTER COLUMN schema_version SET NOT NULL,
    ALTER COLUMN board SET NOT NULL,
    ALTER COLUMN meta SET NOT NULL,
    ALTER COLUMN players SET NOT NULL,
    ALTER COLUMN roster SET NOT NULL,
    ALTER COLUMN history SET NOT NULL,
    ALTER COLUMN catalog SET NOT NULL,
    DROP CONSTRAINT draft_publications_payload_check,
    ADD CONSTRAINT draft_publications_schema_version_check
        CHECK (schema_version IN (1, 2)),
    ADD CONSTRAINT draft_publications_sections_check CHECK (
        jsonb_typeof(board) = 'object' AND jsonb_typeof(meta) = 'object'
        AND jsonb_typeof(players) = 'array' AND jsonb_typeof(roster) = 'array'
        AND jsonb_typeof(history) = 'array' AND jsonb_typeof(catalog) = 'array'
    ),
    ADD CONSTRAINT draft_publications_payload_matches_sections CHECK (
        payload IS NULL OR payload = jsonb_build_object(
            'schema_version', schema_version, 'board', board, 'meta', meta,
            'players', players, 'roster', roster, 'history', history,
            'catalog', catalog
        )
    );
NOTIFY pgrst, 'reload schema';
COMMIT;
