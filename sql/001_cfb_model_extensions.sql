begin;

create table if not exists cfb.team_unit_ratings (
  season int not null,
  week int not null,
  as_of timestamptz not null,
  model_version text not null,
  source_season int,
  team_id bigint not null,
  team text not null,
  classification text,
  unit_history_missing boolean not null default false,
  rush_offense float8,
  pass_offense float8,
  rush_defense float8,
  pass_defense float8,
  pass_block float8,
  run_block float8,
  primary key (season, week, team_id)
);

alter table cfb.game_projections
  add column if not exists pure_home_margin float8,
  add column if not exists pure_home_spread float8,
  add column if not exists market_home_spread float8,
  add column if not exists market_weight float8,
  add column if not exists market_informed_home_margin float8,
  add column if not exists market_informed_home_spread float8;

alter table cfb.team_unit_ratings enable row level security;

do $$
begin
  if not exists (
    select 1
    from pg_policies
    where schemaname = 'cfb'
      and tablename = 'team_unit_ratings'
      and policyname = 'public read'
  ) then
    execute 'create policy "public read" '
      'on cfb.team_unit_ratings '
      'for select to anon, authenticated '
      'using (true)';
  end if;
end
$$;

grant select on cfb.team_unit_ratings to anon, authenticated;
grant all on cfb.team_unit_ratings to service_role;

commit;
