begin;
-- Latest weekly position trait snapshot per season; the parquet keeps the history.
create table if not exists cfb.position_traits (
    season integer not null,
    week integer not null,
    as_of timestamptz not null,
    model_version text not null,
    athlete_id text not null,
    athlete_name text,
    team text,
    team_id integer,
    position_group text not null,
    games double precision,
    trait text not null,
    value double precision,
    percentile double precision,
    kind text,
    in_grade boolean not null,
    uncertainty double precision,
    points_above_replacement double precision,
    primary key (season, athlete_id, trait)
);
create index if not exists position_traits_group_trait_idx
    on cfb.position_traits (season, position_group, trait);
alter table cfb.position_traits enable row level security;
grant select on cfb.position_traits to anon, authenticated;
create policy position_traits_read on cfb.position_traits
    for select to anon, authenticated using (true);
commit;
