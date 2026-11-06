begin;
-- DraftKings lines at three checkpoints per game, for closing-line value and
-- line-timing tests: opened (DraftKings' opening line, read from ESPN on
-- Monday), decided (the line when the Monday forecast and picks publish) and
-- closed (the captured closing line). Each checkpoint is a smallint array in
-- this order: home spread in half-points, spread home price, spread away
-- price, total in half-points, over price, under price, home moneyline, away
-- moneyline. About 0.2 MB a season. Written once per checkpoint by the
-- backend; research data, so no public read and no site_revalidate trigger.
create table if not exists cfb.line_timing (
  game_id bigint primary key,
  season smallint not null,
  opened smallint[],
  decided smallint[],
  closed smallint[],
  decided_at timestamptz,
  closed_at timestamptz,
  constraint line_timing_shape check (
    (opened is null or array_length(opened, 1) = 8)
    and (decided is null or array_length(decided, 1) = 8)
    and (closed is null or array_length(closed, 1) = 8)
    and (decided is null) = (decided_at is null)
    and (closed is null) = (closed_at is null))
);
alter table cfb.line_timing enable row level security;
revoke all on cfb.line_timing from anon, authenticated;
grant select, insert, update on cfb.line_timing to service_role;

-- Closing-line value of every recommended pick against DraftKings, from the
-- line it was taken at to the close, and how far the market itself moved from
-- the Monday decision to the close. Positive favours the pick. Spreads and
-- totals in points; moneylines in implied probability (vig included).
create or replace view cfb.pick_line_value with (security_invoker = true) as
with picks as (
  select r.game_id, r.season, r.week, r.market, r.side, r.point, r.price,
         r.policy_version, r.reason, r.outcome, t.decided, t.closed
  from cfb.recommendations r
  join cfb.line_timing t using (game_id)
  where r.status = 'recommended'
)
select game_id, season, week, market, side, point, price, policy_version, reason, outcome,
  case
    when market = 'spreads' and closed[1] is not null then
      point - (case when side = 'home' then 1 else -1 end) * closed[1] / 2.0
    when market = 'totals' and closed[4] is not null then
      (case when side = 'over' then closed[4] / 2.0 - point else point - closed[4] / 2.0 end)
  end as clv_points,
  case
    when market = 'spreads' and closed[1] is not null and decided[1] is not null then
      (case when side = 'home' then 1 else -1 end) * (decided[1] - closed[1]) / 2.0
    when market = 'totals' and closed[4] is not null and decided[4] is not null then
      (case when side = 'over' then 1 else -1 end) * (closed[4] - decided[4]) / 2.0
  end as market_move_points,
  case when market = 'h2h' then
    (select case when c < 0 then -c::float8 / (-c + 100) else 100.0 / (c + 100) end
       from (select case when side = 'home' then closed[7] else closed[8] end as c) x
      where c is not null)
    - case when price < 0 then -price / (-price + 100) else 100.0 / (price + 100) end
  end as clv_probability
from picks;
revoke all on cfb.pick_line_value from anon, authenticated;
grant select on cfb.pick_line_value to service_role;
commit;
