begin;

create table if not exists cfb.recommendations (
  game_id bigint not null,
  market text not null check (market in ('spreads', 'totals')),
  season integer not null,
  week integer not null,
  start_date timestamptz not null,
  home_team text not null,
  away_team text not null,
  model_version text not null,
  forecast_as_of timestamptz not null,
  home_missing_input_count integer,
  away_missing_input_count integer,
  policy_version text not null,
  decision_at timestamptz not null,
  published_at timestamptz not null default clock_timestamp(),
  status text not null check (status in ('recommended', 'no_play')),
  reason text not null,
  selection text,
  side text check (side in ('home', 'away', 'over', 'under')),
  point double precision,
  price double precision,
  provider text,
  provider_key text,
  market_fetched_at timestamptz,
  odds_api_event_id text,
  provider_start_date timestamptz,
  provider_last_update timestamptz,
  match_score double precision,
  win_probability double precision,
  push_probability double precision,
  probability_edge double precision,
  expected_value_per_unit double precision,
  stake_units double precision not null,
  model_home_margin double precision not null,
  model_total double precision not null,
  margin_sd double precision not null,
  total_sd double precision not null,
  degrees_of_freedom double precision,
  outcome text not null default 'pending'
    check (outcome in ('pending', 'win', 'loss', 'push', 'void', 'no_play')),
  home_points integer,
  away_points integer,
  profit_units double precision,
  graded_at timestamptz,
  primary key (game_id, market),
  check (forecast_as_of <= decision_at and decision_at <= published_at),
  check (published_at < start_date),
  check ((status = 'no_play' and stake_units = 0) or
    (status = 'recommended' and stake_units = 1 and
     home_missing_input_count = 0 and away_missing_input_count = 0 and
     match_score >= 0.95 and match_score <= 1 and
     selection is not null and side is not null and point is not null and
     ((market = 'spreads' and side in ('home','away')) or
      (market = 'totals' and side in ('over','under'))) and
     selection = case side when 'home' then home_team when 'away' then away_team
                           when 'over' then 'Over' else 'Under' end and
     abs(point) < 'Infinity'::float8 and point * 2 = round(point * 2) and
     abs(price) >= 100 and abs(price) < 'Infinity'::float8 and
     provider is not null and provider_key is not null and
     odds_api_event_id is not null and provider_start_date = start_date and
     provider_last_update <= market_fetched_at and market_fetched_at <= decision_at and
     published_at - provider_last_update <= interval '1 hour' and
     published_at - forecast_as_of <= interval '7 days' and
     abs(model_home_margin) < 'Infinity'::float8 and
     model_total >= 0 and model_total < 'Infinity'::float8 and
     margin_sd > 0 and margin_sd < 'Infinity'::float8 and
     total_sd > 0 and total_sd < 'Infinity'::float8 and
     (degrees_of_freedom is null or degrees_of_freedom > 2) and
     win_probability > 0 and win_probability < 1 and
     push_probability >= 0 and win_probability + push_probability <= 1 and
     probability_edge >= 0.045 and probability_edge < 1 and
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true),
  check ((outcome = 'pending' and graded_at is null and profit_units is null) or
    (outcome <> 'pending' and graded_at is not null and profit_units is not null))
);

create index if not exists recommendations_season_start
  on cfb.recommendations (season, start_date desc, game_id, market);

create or replace function cfb.protect_recommendation() returns trigger
language plpgsql set search_path = pg_catalog, cfb as $$
declare
  balance float8;
  expected_outcome text;
  expected_profit float8;
begin
  if TG_OP = 'DELETE' then
    raise exception 'Recommendation history cannot be deleted';
  end if;
  if TG_OP = 'INSERT' then
    -- Caller-supplied historical timestamps must never admit a past pick.
    new.published_at := clock_timestamp();
    if new.outcome <> 'pending' then
      raise exception 'New recommendations must be pending';
    end if;
  end if;
  if TG_OP = 'UPDATE' then
    if old.outcome <> 'pending' and new is distinct from old then
      raise exception 'Settled recommendations are immutable';
    end if;
    if old.status = 'recommended' or old.start_date <= clock_timestamp() then
      if (to_jsonb(new) - array['outcome','home_points','away_points','profit_units','graded_at'])
        is distinct from
         (to_jsonb(old) - array['outcome','home_points','away_points','profit_units','graded_at']) then
        raise exception 'Published picks and started game decisions are frozen';
      end if;
    end if;
    if old.status = 'no_play' and old.outcome = 'pending' and new.outcome = 'pending'
       and new is distinct from old then
      new.published_at := clock_timestamp();
    end if;
  end if;
  if new.outcome = 'pending' then
    if new.home_points is not null or new.away_points is not null then
      raise exception 'Pending recommendations cannot contain final scores';
    end if;
  else
    if not (new.graded_at >= new.published_at and new.graded_at <= clock_timestamp()) then
      raise exception 'Invalid settlement timestamp';
    end if;
    if new.outcome = 'void' then
      expected_profit := 0;
    elsif new.outcome = 'no_play' and new.status = 'no_play' then
      if new.graded_at < new.start_date then
        raise exception 'Cannot settle before kickoff';
      end if;
      expected_profit := 0;
    elsif new.status = 'recommended' and new.outcome in ('win','loss','push') then
      if not (new.graded_at >= new.start_date and new.home_points >= 0
              and new.away_points >= 0) is true then
        raise exception 'Settlement requires a started game and final scores';
      end if;
      balance := case when new.market = 'spreads' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end + new.point
        else (new.home_points::float8 + new.away_points - new.point) * case new.side when 'over' then 1 else -1 end end;
      expected_outcome := case when balance > 0 then 'win' when balance < 0 then 'loss' else 'push' end;
      if new.outcome <> expected_outcome then
        raise exception 'Outcome disagrees with the recorded line and final score';
      end if;
      expected_profit := case new.outcome when 'win' then
        case when new.price > 0 then new.price / 100 else 100 / abs(new.price) end
        when 'loss' then -1 else 0 end;
    else
      raise exception 'Outcome is incompatible with the recorded decision';
    end if;
    if not (abs(new.profit_units - expected_profit) < 1e-10) is true then
      raise exception 'Profit disagrees with the recorded price and outcome';
    end if;
  end if;
  return new;
end
$$;

drop trigger if exists protect_recommendation on cfb.recommendations;
create trigger protect_recommendation before insert or update or delete on cfb.recommendations
  for each row execute function cfb.protect_recommendation();

create or replace view cfb.recommendation_performance
with (security_invoker = true) as
with segments as (
  select r.*, s.kind, s.label
  from cfb.recommendations r
  cross join lateral (values
    ('overall', 'All picks'), ('market', r.market), ('week', r.week::text),
    ('policy', r.policy_version),
    ('edge', case when r.probability_edge < 0.075 then '4.5-7.5 pp'
                  when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end)
  ) s(kind, label)
), aggregates as (
  select season, kind as segment_kind, label as segment,
    count(*) filter (where status = 'recommended') as picks,
    count(*) filter (where status = 'no_play') as no_plays,
    count(*) filter (where status = 'recommended' and outcome = 'pending') as pending,
    count(*) filter (where outcome = 'win') as wins,
    count(*) filter (where outcome = 'loss') as losses,
    count(*) filter (where outcome = 'push') as pushes,
    count(*) filter (where status = 'recommended' and outcome = 'void') as voids,
    coalesce(sum(stake_units) filter (where outcome in ('win','loss','push')), 0) as staked_units,
    coalesce(sum(profit_units) filter (where outcome in ('win','loss','push')), 0) as profit_units,
    avg(expected_value_per_unit) filter (where status = 'recommended') as average_ev,
    min(decision_at) as first_decision_at,
    max(decision_at) as last_decision_at,
    max(graded_at) as last_graded_at
  from segments group by season, kind, label
)
select *, profit_units / nullif(staked_units, 0) as roi,
  wins::float8 / nullif(wins + losses, 0) as win_rate,
  wins + losses + pushes < 30 as thin_sample
from aggregates;

alter table cfb.recommendations enable row level security;
drop policy if exists public_read on cfb.recommendations;
create policy public_read on cfb.recommendations for select to anon, authenticated using (true);
grant usage on schema cfb to anon, authenticated, service_role;
grant select on cfb.recommendations, cfb.recommendation_performance to anon, authenticated;
revoke all on cfb.recommendations from service_role;
grant select, insert, update on cfb.recommendations to service_role;
grant select on cfb.recommendation_performance to service_role;
notify pgrst, 'reload schema';
commit;
