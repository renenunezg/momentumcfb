begin;
-- Publication is stamped by the database, not by the caller. Every insert or
-- rewrite of a projection sets published_at to the transaction's clock, so an
-- outsider can check that the row graded was published before kickoff. Rows
-- published before this migration keep a null published_at; their evidence is
-- as_of and, from week 2, the matching cfb.recommendations.published_at.
alter table cfb.game_projections add column if not exists published_at timestamptz;
alter table cfb.game_projections
    add constraint game_projection_pregame check (as_of < start_date);

create or replace function cfb.stamp_game_projection() returns trigger
language plpgsql set search_path = pg_catalog, cfb as $$
begin
  new.published_at := clock_timestamp();
  if new.published_at >= new.start_date then
    raise exception 'A projection cannot be published after kickoff';
  end if;
  return new;
end
$$;
drop trigger if exists stamp_game_projection on cfb.game_projections;
create trigger stamp_game_projection
    before insert or update on cfb.game_projections
    for each row execute function cfb.stamp_game_projection();

-- A graded game is a frozen record: written once, never updated or deleted.
-- forecast_published_at copies the projection's published_at at grading.
alter table cfb.graded_games add column if not exists forecast_published_at timestamptz;
alter table cfb.graded_games
    alter column model_version set not null,
    alter column forecast_as_of set not null,
    add constraint graded_game_pregame check (
        forecast_as_of < start_date
        and (forecast_published_at is null or forecast_published_at < start_date));

create or replace function cfb.protect_graded_game() returns trigger
language plpgsql set search_path = pg_catalog, cfb as $$
begin
  raise exception 'Graded games are frozen records';
end
$$;
drop trigger if exists protect_graded_game on cfb.graded_games;
create trigger protect_graded_game
    before update or delete on cfb.graded_games
    for each row execute function cfb.protect_graded_game();
drop trigger if exists protect_graded_game_truncate on cfb.graded_games;
create trigger protect_graded_game_truncate
    before truncate on cfb.graded_games
    for each statement execute function cfb.protect_graded_game();
commit;
