begin;
-- Keep historical decisions unchanged; new decisions carry replayable pricing evidence.
alter table cfb.recommendations add column if not exists pricing_context jsonb
  check (pricing_context is null or jsonb_typeof(pricing_context) = 'object');
alter table cfb.recommendations drop constraint recommendation_eligibility_v8;
alter table cfb.recommendations add constraint recommendation_eligibility_v11
check ((status = 'no_play' and stake_units = 0) or
    (status = 'recommended' and stake_units = 1 and
     ((home_missing_input_count = 0 and away_missing_input_count = 0) or
      (policy_version in ('cfb-picks-v2', 'cfb-picks-v3', 'cfb-picks-v4', 'cfb-picks-v5', 'cfb-picks-v6', 'cfb-picks-v7', 'cfb-picks-v8', 'cfb-picks-v9', 'cfb-picks-v10', 'cfb-picks-v11')
       and home_missing_input_count between 0 and 1
       and away_missing_input_count between 0 and 1)) and
     match_score >= 0.95 and match_score <= 1 and
     selection is not null and side is not null and
     ((market in ('spreads','h2h') and side in ('home','away')) or
      (market = 'totals' and side in ('over','under'))) and
     selection = case side when 'home' then home_team when 'away' then away_team
                           when 'over' then 'Over' else 'Under' end and
     ((market = 'h2h' and point is null and push_probability = 0) or
      (market <> 'h2h' and point is not null and abs(point) < 'Infinity'::float8
       and point * 2 = round(point * 2))) and
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
     probability_edge > -1 and probability_edge < 1 and
     ((policy_version in ('cfb-picks-v2', 'cfb-picks-v3', 'cfb-picks-v4')
       and probability_edge >= 0.045) or
      (policy_version = 'cfb-picks-v5'
       and edge_points >= 2 and edge_points < 'Infinity'::float8) or
      (policy_version in ('cfb-picks-v6', 'cfb-picks-v7', 'cfb-picks-v8')
       and edge_points > 0 and edge_points < 'Infinity'::float8) or
      (policy_version in ('cfb-picks-v9', 'cfb-picks-v10', 'cfb-picks-v11') and pricing_context is not null
       and pricing_context->>'purpose' = 'research'
       and pricing_context->>'validation_status' = 'unvalidated'
       and edge_points > 0 and edge_points < 'Infinity'::float8
       and (market <> 'h2h' or
         (edge_points >= 2 and reason = 'qualifying_edge'
          and pricing_context->>'probability_source' = 'paired_moneyline_margin_blend_v1'
          and (pricing_context->>'model_weight')::float8 = 0.2
          and pricing_context->>'blend_space' = 'margin'
          and abs((pricing_context->>'pricing_home_margin')::float8
            - (0.8 * (pricing_context->>'market_equivalent_home_margin')::float8
               + 0.2 * (pricing_context->>'model_home_margin')::float8)) < 1e-12
          and jsonb_typeof(pricing_context->'reference_pairs') = 'array'
          and jsonb_array_length(pricing_context->'reference_pairs') > 0
          and abs(win_probability - case when side = 'home'
            then (pricing_context->>'home_probability')::float8
            else 1 - (pricing_context->>'home_probability')::float8 end) < 1e-12)))) and
     (policy_version not in ('cfb-picks-v10', 'cfb-picks-v11') or (
       ((reason = 'qualifying_edge' and edge_points >= 2
         and pricing_context->>'recommendation_tier' = 'edge_qualified') or
        (market <> 'h2h' and reason = 'volume_floor' and edge_points < 2
         and pricing_context->>'recommendation_tier' = 'exploratory'))
       and (market <> 'spreads' or (
         pricing_context->>'probability_source' = 'market_history_margin_blend_keys_v1'
         and pricing_context->>'calibration_version' = 'spread_keys_020_v1'
         and pricing_context->>'blend_space' = 'margin'
         and (pricing_context->>'model_weight')::float8 = 0.2
         and margin_sd = 15.35 and degrees_of_freedom = 500
         and abs(model_home_margin - (pricing_context->>'pricing_home_margin')::float8) < 1e-12
         and abs(model_home_margin - (
           0.8 * (pricing_context->>'market_home_margin')::float8
           + 0.2 * (pricing_context->>'model_side_home_margin')::float8)) < 1e-12
         and (pricing_context->>'reference_as_of')::timestamptz = forecast_as_of
       ))
       and (market <> 'totals' or total_sd = 15.93)
     )) and
     (policy_version <> 'cfb-picks-v11' or (
       pricing_context->>'floor_policy' = 'shortlist_market_soft_cap_v1'
       and pricing_context->'floor_backfill' = 'false'::jsonb
       and (pricing_context->>'floor_shortfall')::integer >= 0
       and (pricing_context->>'floor_market_cap')::integer =
         ((pricing_context->>'floor_shortfall')::integer + 1) / 2
       and (reason <> 'volume_floor' or (
         pricing_context->'floor_shortlisted' = 'true'::jsonb
         and pricing_context->'floor_capped' = 'false'::jsonb))
       and jsonb_typeof(pricing_context->'research_flags') = 'array'
       and (market <> 'spreads' or (
         pricing_context->'spread_edge_breakdown'->>'perspective' = 'selected_side'
         and abs(edge_points - (
           (pricing_context->'spread_edge_breakdown'->>'model_contribution_points')::float8
           + (pricing_context->'spread_edge_breakdown'->>'quote_difference_points')::float8
           + (pricing_context->'spread_edge_breakdown'->>'price_distribution_adjustment_points')::float8
         )) < 1e-12
         and abs((pricing_context->'spread_edge_breakdown'->>'model_contribution_points')::float8
           - 0.2 * (pricing_context->'spread_edge_breakdown'->>'model_disagreement_points')::float8) < 1e-12
       ))
     )) and
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true);

create or replace view cfb.recommendation_performance
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
    (count(*) filter (where clv_points > 0))::float8 / nullif(count(clv_points), 0) as clv_positive_share
  from segments group by season, kind, label
)
select *, profit_units / nullif(staked_units, 0) as roi,
  wins::float8 / nullif(wins + losses, 0) as win_rate,
  wins + losses + pushes < 30 as thin_sample
from aggregates;

create or replace function cfb.recommendation_summary(
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
