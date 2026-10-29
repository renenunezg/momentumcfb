begin;

-- Closing line value against the market, not against the shopped price.
-- clv_points (kept for readers that still use it) compares the line taken, the
-- best of every book in the feed, with the CFBD median close, so shopping alone
-- makes it positive. Each decision now records the median market on its side:
-- decision_market_point, the median posted line for spreads and totals, and
-- decision_market_probability, the vig-free median moneyline probability. At
-- settlement clv_market_points is that median line minus the close (side
-- signed, positive when the market moved toward the pick; a total decided
-- before this migration uses its recorded median posted total, market_total),
-- closing_probability is the vig-free CFBD median closing probability of the
-- pick's side, and clv_market_probability is closing_probability minus the
-- decision-time median. Settled picks receive each new field once.
alter table cfb.recommendations
    add column if not exists decision_market_point double precision,
    add column if not exists decision_market_probability double precision,
    add column if not exists closing_probability double precision,
    add column if not exists clv_market_points double precision,
    add column if not exists clv_market_probability double precision;
alter table cfb.recommendation_decisions
    add column if not exists decision_market_point double precision,
    add column if not exists decision_market_probability double precision,
    add column if not exists closing_probability double precision,
    add column if not exists clv_market_points double precision,
    add column if not exists clv_market_probability double precision;

alter table cfb.recommendations add constraint recommendation_decision_market
check (((decision_market_point is null
   or (market in ('spreads', 'totals') and decision_market_point > '-Infinity'::float8
       and decision_market_point < 'Infinity'::float8))
  and (decision_market_probability is null
   or (market = 'h2h' and decision_market_probability > 0
       and decision_market_probability < 1))) is true);
alter table cfb.recommendation_decisions add constraint recommendation_decision_market
check (((decision_market_point is null
   or (market in ('spreads', 'totals') and decision_market_point > '-Infinity'::float8
       and decision_market_point < 'Infinity'::float8))
  and (decision_market_probability is null
   or (market = 'h2h' and decision_market_probability > 0
       and decision_market_probability < 1))) is true);
alter table cfb.recommendations add constraint recommendation_market_clv
check (((closing_probability is null and clv_market_points is null
  and clv_market_probability is null)
 or (closing_source is not null and status = 'recommended'
  and outcome in ('win', 'loss', 'push') and
  ((market = 'h2h' and clv_market_points is null
    and (closing_probability is null
         or (closing_probability > 0 and closing_probability < 1))
    and (clv_market_probability is null or closing_probability is not null))
   or (market in ('spreads', 'totals') and closing_probability is null
    and clv_market_probability is null
    and (clv_market_points is null or abs(clv_market_points - case
      when market = 'spreads' then decision_market_point - closing_point
      when side = 'over' then closing_point - coalesce(decision_market_point, market_total)
      else coalesce(decision_market_point, market_total) - closing_point end) < 1e-9))))) is true);
alter table cfb.recommendation_decisions add constraint recommendation_market_clv
check (((closing_probability is null and clv_market_points is null
  and clv_market_probability is null)
 or (closing_source is not null and status = 'recommended'
  and outcome in ('win', 'loss', 'push') and
  ((market = 'h2h' and clv_market_points is null
    and (closing_probability is null
         or (closing_probability > 0 and closing_probability < 1))
    and (clv_market_probability is null or closing_probability is not null))
   or (market in ('spreads', 'totals') and closing_probability is null
    and clv_market_probability is null
    and (clv_market_points is null or abs(clv_market_points - case
      when market = 'spreads' then decision_market_point - closing_point
      when side = 'over' then closing_point - coalesce(decision_market_point, market_total)
      else coalesce(decision_market_point, market_total) - closing_point end) < 1e-9))))) is true);

create or replace function cfb.protect_recommendation() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path TO 'pg_catalog', 'cfb'
    AS $$
declare
  balance float8;
  expected_outcome text;
  expected_profit float8;
  settlement_fields text[] := array['outcome','home_points','away_points','profit_units',
    'graded_at','settlement_reason','closing_point','closing_price','closing_source','clv_points',
    'closing_probability','clv_market_points','clv_market_probability'];
  closing_fields text[] := array['closing_point','closing_price','closing_source','clv_points',
    'closing_probability','clv_market_points','clv_market_probability'];
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
    if new.closing_source is not null then
      raise exception 'Closing lines are recorded at settlement';
    end if;
  end if;
  if TG_OP = 'UPDATE' then
    if old.outcome <> 'pending' and new is distinct from old then
      -- A settled pick may receive each closing field once: a recorded field
      -- never changes, nothing outside the closing fields changes, and every
      -- closing field needs the closing source.
      if new.closing_source is null
         or (to_jsonb(new) - closing_fields) is distinct from (to_jsonb(old) - closing_fields)
         or exists (
           select 1 from unnest(closing_fields) as field
           where (to_jsonb(old) -> field) <> 'null'::jsonb
             and (to_jsonb(new) -> field) is distinct from (to_jsonb(old) -> field)) then
        raise exception 'Settled recommendations are immutable';
      end if;
      return new;
    end if;
    if old.status = 'recommended' or old.start_date <= clock_timestamp() then
      if (to_jsonb(new) - settlement_fields) is distinct from (to_jsonb(old) - settlement_fields) then
        -- Before kickoff a newer model version may replace its own open pick.
        if not (old.status = 'recommended' and new.outcome = 'pending'
                and old.start_date > clock_timestamp()
                and new.model_version is distinct from old.model_version) is true then
          raise exception 'Published picks and started game decisions are frozen';
        end if;
        new.published_at := clock_timestamp();
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
    if new.closing_source is not null then
      raise exception 'Closing lines are recorded at settlement';
    end if;
  else
    if not (new.graded_at >= new.published_at and new.graded_at <= clock_timestamp()) then
      raise exception 'Invalid settlement timestamp';
    end if;
    if new.outcome = 'void' then
      -- A policy withdrawal settles a published pick before kickoff. Every
      -- other void keeps the schedule-change and moneyline-tie semantics.
      if new.settlement_reason = 'policy_withdrawn'
         and not (TG_OP = 'UPDATE' and old.outcome = 'pending'
                  and new.status = 'recommended'
                  and new.graded_at < new.start_date) is true then
        raise exception 'A withdrawal must settle a pending pick before kickoff';
      end if;
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
      balance := case when new.market = 'h2h' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end
        when new.market = 'spreads' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end + new.point
        else (new.home_points::float8 + new.away_points - new.point) * case new.side when 'over' then 1 else -1 end end;
      if new.market = 'h2h' and balance = 0 then
        raise exception 'A tied moneyline must be void';
      end if;
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

-- The performance view and summary function gain the market CLV aggregates.
-- Adding columns before roi changes the view's shape, so both are recreated.
drop function cfb.recommendation_summary(integer, text, timestamptz);
drop view cfb.recommendation_performance;
create view cfb.recommendation_performance
with (security_invoker = true) as
with segments as (
  select r.*, s.kind, s.label
  from cfb.recommendations r
  cross join lateral (
    select * from (values
      ('overall', 'All picks'), ('market', r.market),
      ('week', r.season::text || ' Week ' || r.week::text),
      ('policy', r.policy_version),
      ('edge', case
         when r.edge_points is null then case
           when r.probability_edge < 0.075 then '4.5-7.5 pp'
           when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end
         when r.edge_points < 2 then 'under 2 pts (floor)'
         when r.edge_points < 4 then '2-4 pts'
         when r.edge_points < 7 then '4-7 pts' else '7+ pts' end)
    ) v(kind,label)
    union all
    select 'qualification', case when r.reason = 'volume_floor' then 'exploratory'
      when r.reason = 'qualifying_edge' then 'edge_qualified' else 'legacy' end
    where r.status = 'recommended'
    union all
    select 'side', r.market || ':' || case
      when r.market = 'totals' then r.side
      when r.market = 'spreads' then case when r.point < 0 then 'favorite'
        when r.point > 0 then 'underdog' else 'pickem' end
      else case when r.price < -100 then 'favorite' when r.price > 100 then 'underdog' else 'even' end
    end
    where r.status = 'recommended'
  ) s(kind,label)
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
    max(graded_at) as last_graded_at,
    avg(clv_points) as average_clv_points,
    count(clv_points) as clv_sample,
    (count(*) filter (where clv_points > 0))::float8 / nullif(count(clv_points), 0) as clv_positive_share,
    avg(clv_market_points) as average_clv_market_points,
    count(clv_market_points) as clv_market_points_sample,
    (count(*) filter (where clv_market_points > 0))::float8
      / nullif(count(clv_market_points), 0) as clv_market_points_positive_share,
    avg(clv_market_probability) as average_clv_market_probability,
    count(clv_market_probability) as clv_market_probability_sample,
    (count(*) filter (where clv_market_probability > 0))::float8
      / nullif(count(clv_market_probability), 0) as clv_market_probability_positive_share
  from segments group by season, kind, label
)
select *, profit_units / nullif(staked_units, 0) as roi,
  wins::float8 / nullif(wins + losses, 0) as win_rate,
  wins + losses + pushes < 30 as thin_sample
from aggregates;

create function cfb.recommendation_summary(
  p_season integer default null,
  p_market text default 'all',
  p_from timestamptz default null
) returns setof cfb.recommendation_performance
language sql stable security invoker set search_path = pg_catalog, cfb as $$
  with filtered as (
    select * from cfb.recommendations
    where (p_season is null or season = p_season)
      and (p_market = 'all' or market = p_market)
      and (p_from is null or decision_at >= p_from)
  ), segments as (
    select r.*, s.kind, s.label
    from filtered r
    cross join lateral (
      select * from (values
        ('overall', 'All picks'), ('market', r.market),
        ('week', r.season::text || ' Week ' || r.week::text),
        ('policy', r.policy_version),
        ('edge', case
           when r.edge_points is null then case
             when r.probability_edge < 0.075 then '4.5-7.5 pp'
             when r.probability_edge < 0.10 then '7.5-10 pp' else '10+ pp' end
           when r.edge_points < 2 then 'under 2 pts (floor)'
           when r.edge_points < 4 then '2-4 pts'
           when r.edge_points < 7 then '4-7 pts' else '7+ pts' end)
      ) v(kind,label)
      union all
      select 'qualification', case when r.reason = 'volume_floor' then 'exploratory'
        when r.reason = 'qualifying_edge' then 'edge_qualified' else 'legacy' end
      where r.status = 'recommended'
      union all
      select 'side', r.market || ':' || case
        when r.market = 'totals' then r.side
        when r.market = 'spreads' then case when r.point < 0 then 'favorite'
          when r.point > 0 then 'underdog' else 'pickem' end
        else case when r.price < -100 then 'favorite' when r.price > 100 then 'underdog' else 'even' end
      end
      where r.status = 'recommended'
    ) s(kind,label)
  ), aggregates as (
    select p_season as season, kind as segment_kind, label as segment,
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
      max(graded_at) as last_graded_at,
      avg(clv_points) as average_clv_points,
      count(clv_points) as clv_sample,
      (count(*) filter (where clv_points > 0))::float8 / nullif(count(clv_points), 0) as clv_positive_share,
      avg(clv_market_points) as average_clv_market_points,
      count(clv_market_points) as clv_market_points_sample,
      (count(*) filter (where clv_market_points > 0))::float8
        / nullif(count(clv_market_points), 0) as clv_market_points_positive_share,
      avg(clv_market_probability) as average_clv_market_probability,
      count(clv_market_probability) as clv_market_probability_sample,
      (count(*) filter (where clv_market_probability > 0))::float8
        / nullif(count(clv_market_probability), 0) as clv_market_probability_positive_share
    from segments group by kind, label
  )
  select *, profit_units / nullif(staked_units, 0) as roi,
    wins::float8 / nullif(wins + losses, 0) as win_rate,
    wins + losses + pushes < 30 as thin_sample
  from aggregates;
$$;

grant select on cfb.recommendation_performance to anon, authenticated, service_role;
revoke all on function cfb.recommendation_summary(integer,text,timestamptz) from public;
grant execute on function cfb.recommendation_summary(integer,text,timestamptz)
  to anon, authenticated, service_role;
notify pgrst, 'reload schema';
commit;
