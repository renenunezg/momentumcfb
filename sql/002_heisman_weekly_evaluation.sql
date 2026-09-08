-- Apply before publishing cfb_heisman_share_v2 artifacts.
-- A winner outside the contemporaneous candidate pool has no predicted rank.
BEGIN;

ALTER TABLE cfb.heisman_history
    ALTER COLUMN actual_winner_predicted_rank DROP NOT NULL;

ALTER TABLE cfb.player_model_meta
    ADD COLUMN IF NOT EXISTS heisman_evaluation_kind text,
    ADD COLUMN IF NOT EXISTS heisman_evaluation_week integer,
    ADD COLUMN IF NOT EXISTS heisman_evaluation_seasons integer,
    ADD COLUMN IF NOT EXISTS heisman_winner_pool_coverage double precision,
    ADD COLUMN IF NOT EXISTS heisman_ballot_share_covered double precision;

COMMIT;
