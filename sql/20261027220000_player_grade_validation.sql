begin;
-- How the served position grade ranks later production against the baselines it
-- must beat: one row per grade model, input value model, evaluation, seasons and
-- position group. Written by `python -m backend publish-grade-validation` from
-- backend/players/traits/validation.py; never by the weekly runner.
--
-- eval_kind: 'next_season' (final-week grade of season t against shrunk_per_game at
-- the final week of t + 1) or 'rest_of_season' (grade at weeks 5 to 9 against
-- adjusted EPA credit per game over the player's later games, mean of the weeks).
-- Every spearman_* column is on the same players; spearman_qbr is null unless
-- every player in the row has a QBR. diff_* is grade minus EPA credit with its
-- 95 percent paired bootstrap interval. holdout_status says what the seasons were
-- to the model ('2025 inspected during development', 'pre-registered, untouched').
create table if not exists cfb.player_grade_validation (
    grade_model_version text not null,
    model_sha text not null,
    input_value_model_version text not null,
    eval_kind text not null check (eval_kind in ('next_season', 'rest_of_season')),
    seasons text not null,
    position_group text not null,
    players integer not null,
    spearman_grade double precision not null,
    spearman_epa_credit double precision not null,
    spearman_games double precision,
    spearman_value_above_replacement double precision,
    spearman_qbr double precision,
    diff_grade_minus_epa double precision not null,
    diff_lo_95 double precision,
    diff_hi_95 double precision,
    holdout_status text not null,
    computed_at timestamptz not null,
    primary key (
        grade_model_version,
        input_value_model_version,
        eval_kind,
        seasons,
        position_group
    )
);
alter table cfb.player_grade_validation enable row level security;
grant select on cfb.player_grade_validation to anon, authenticated;
create policy player_grade_validation_read on cfb.player_grade_validation
    for select to anon, authenticated using (true);
-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql. A publish
-- replaces one evaluation's rows in one transaction, so the site refreshes once.
drop trigger if exists site_revalidate on cfb.player_grade_validation;
create trigger site_revalidate
    after insert or delete or update or truncate on cfb.player_grade_validation
    for each statement execute function public.site_revalidate();
commit;
