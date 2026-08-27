begin;

create schema if not exists cfb;

create table if not exists cfb.teams (
  team_id integer primary key,
  team text not null,
  color text,
  alternate_color text,
  logo_light text,
  logo_dark text
);

create table if not exists cfb.team_ratings (
  season integer not null,
  week integer not null,
  as_of timestamptz not null,
  model_version text not null,
  team_id bigint not null,
  team text not null,
  conference text,
  classification text,
  offense_points double precision,
  defense_points double precision,
  power_rating double precision not null,
  scoring_environment double precision,
  expected_possessions double precision,
  power_rating_sd double precision,
  missing_input_count integer,
  primary key (season, week, team_id)
);

create table if not exists cfb.team_unit_ratings (
  season integer not null,
  week integer not null,
  as_of timestamptz not null,
  model_version text not null,
  source_season integer,
  team_id bigint not null,
  team text not null,
  classification text,
  unit_history_missing boolean not null default false,
  rush_offense double precision,
  pass_offense double precision,
  rush_defense double precision,
  pass_defense double precision,
  pass_block double precision,
  run_block double precision,
  primary key (season, week, team_id)
);

create table if not exists cfb.game_projections (
  game_id bigint primary key,
  season integer not null,
  week integer not null,
  as_of timestamptz not null,
  model_version text not null,
  start_date timestamptz,
  home_team_id bigint,
  home_team text not null,
  away_team_id bigint,
  away_team text not null,
  neutral_site boolean,
  home_field_points double precision,
  expected_home_points double precision,
  expected_away_points double precision,
  home_margin double precision,
  home_spread double precision,
  model_total double precision,
  margin_sd double precision,
  total_sd double precision,
  margin_total_correlation double precision,
  distribution text,
  degrees_of_freedom double precision,
  home_classification text,
  away_classification text,
  home_missing_input_count integer,
  away_missing_input_count integer,
  conference_game boolean,
  pure_home_margin double precision,
  pure_home_spread double precision,
  market_home_spread double precision,
  market_weight double precision,
  market_informed_home_margin double precision,
  market_informed_home_spread double precision
);

create table if not exists cfb.market_comparisons (
  game_id bigint primary key,
  start_date timestamptz,
  home_team text,
  away_team text,
  model_home_spread double precision,
  model_total double precision,
  margin_sd double precision,
  total_sd double precision,
  model_as_of timestamptz,
  market_available boolean,
  priced_offer_available boolean,
  executable_offer_available boolean,
  review_status text,
  recommendation_status text,
  best_offer_market text,
  best_offer_selection text,
  best_offer_point double precision,
  best_offer_price double precision,
  best_offer_provider text,
  best_offer_provider_key text,
  best_offer_provider_last_update timestamptz,
  best_offer_event_link text,
  best_offer_market_link text,
  best_offer_bet_link text,
  best_offer_edge_points double precision,
  best_offer_edge_standardized double precision,
  best_offer_model_cover_probability double precision,
  best_offer_model_fair_price double precision,
  best_offer_expected_value_per_unit double precision
);

create table if not exists cfb.backtest_predictions (
  game_id bigint primary key,
  season integer not null,
  week integer not null,
  week_index integer,
  season_type text,
  home_team text not null,
  away_team text not null,
  neutral_site boolean,
  home_points integer,
  away_points integer,
  margin double precision,
  closing_spread double precision,
  model_margin double precision,
  actual_margin double precision
);

create table if not exists cfb.serving_anchors (
  season integer not null,
  anchor_week integer not null,
  game_id bigint not null,
  model_week integer not null,
  home_margin double precision not null,
  margin_sd double precision not null,
  closing_spread double precision,
  n_spread_offers integer,
  margin_sd_method text,
  market_anchor_source text,
  closing_snapshot_id text,
  closing_fetched_at timestamptz,
  latest_provider_update timestamptz,
  published_at timestamptz not null default now(),
  primary key (season, anchor_week, game_id)
);

create index if not exists game_projections_season_week
  on cfb.game_projections (season, week);
create index if not exists backtest_predictions_season_week
  on cfb.backtest_predictions (season, week);

alter table cfb.teams enable row level security;
alter table cfb.team_ratings enable row level security;
alter table cfb.team_unit_ratings enable row level security;
alter table cfb.game_projections enable row level security;
alter table cfb.market_comparisons enable row level security;
alter table cfb.backtest_predictions enable row level security;
alter table cfb.serving_anchors enable row level security;

do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'teams',
    'team_ratings',
    'team_unit_ratings',
    'game_projections',
    'market_comparisons',
    'backtest_predictions',
    'serving_anchors'
  ]
  loop
    if not exists (
      select 1
      from pg_policies
      where schemaname = 'cfb'
        and tablename = table_name
        and policyname = 'public_read'
    ) then
      execute format(
        'create policy public_read on cfb.%I '
        'for select to anon, authenticated using (true)',
        table_name
      );
    end if;
  end loop;
end
$$;

grant usage on schema cfb to anon, authenticated, service_role;
grant select on all tables in schema cfb to anon, authenticated;
grant all on all tables in schema cfb to service_role;

commit;
