begin;
-- One raw season stat line per published athlete. ``stats`` holds the values in
-- the key order and scale ``backend/players/season_stats.py`` (STAT_KEYS) defines
-- per position group; momentumweb decodes it. The weekly publish replaces the
-- season's rows with one delete and one insert.
create table if not exists cfb.player_season_stats (
    season smallint not null,
    athlete_id text not null,
    position_group text not null,
    week smallint not null,
    stats smallint[] not null,
    updated_at timestamptz not null default now(),
    primary key (season, athlete_id)
);
alter table cfb.player_season_stats enable row level security;
grant select on cfb.player_season_stats to anon, authenticated;
create policy player_season_stats_read on cfb.player_season_stats
    for select to anon, authenticated using (true);
commit;
