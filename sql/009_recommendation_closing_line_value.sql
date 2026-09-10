begin;

-- Closing line value. Every settled pick records the CFBD median closing
-- number for its side, so the ledger shows whether the market moved toward
-- the pick after the decision. clv_points is positive when a spread or total
-- pick beat the close; a moneyline records only its closing price because its
-- value lives in the price. Picks settled before this migration receive their
-- closing fields once; no other change to a settled pick is admitted.
alter table cfb.recommendations add column closing_point double precision;
alter table cfb.recommendations add column closing_price double precision;
alter table cfb.recommendations add column closing_source text;
alter table cfb.recommendations add column clv_points double precision;
alter table cfb.recommendations add constraint recommendation_closing_line_value
check ((closing_source is null and closing_point is null
        and closing_price is null and clv_points is null) or
  (closing_source is not null and status = 'recommended'
   and outcome in ('win', 'loss', 'push') and
   ((market = 'h2h' and closing_price is not null
     and closing_point is null and clv_points is null) or
    (market = 'spreads' and closing_point is not null and closing_price is null
     and abs(clv_points - (point - closing_point)) < 1e-9) or
    (market = 'totals' and closing_point is not null and closing_price is null
     and abs(clv_points - case side when 'over' then closing_point - point
                                    else point - closing_point end) < 1e-9))) is true);

create or replace function cfb.protect_recommendation() returns trigger
language plpgsql set search_path = pg_catalog, cfb as $$
declare
  balance float8;
  expected_outcome text;
  expected_profit float8;
  settlement_fields text[] := array['outcome','home_points','away_points','profit_units',
    'graded_at','settlement_reason','closing_point','closing_price','closing_source','clv_points'];
  closing_fields text[] := array['closing_point','closing_price','closing_source','clv_points'];
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
      -- A settled pick may receive its closing line exactly once.
      if old.closing_source is not null or new.closing_source is null
         or (to_jsonb(new) - closing_fields) is distinct from (to_jsonb(old) - closing_fields) then
        raise exception 'Settled recommendations are immutable';
      end if;
      return new;
    end if;
    if old.status = 'recommended' or old.start_date <= clock_timestamp() then
      if (to_jsonb(new) - settlement_fields) is distinct from (to_jsonb(old) - settlement_fields) then
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
        raise exception 'A policy withdrawal must settle a pending pick before kickoff';
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

-- The performance view and summary function gain closing line value
-- aggregates over settled spread and total picks: mean points beaten, the
-- share of picks that beat the close, and the sample behind both.
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
    (count(*) filter (where clv_points > 0))::float8 / nullif(count(clv_points), 0) as clv_positive_share
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
      (count(*) filter (where clv_points > 0))::float8 / nullif(count(clv_points), 0) as clv_positive_share
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
