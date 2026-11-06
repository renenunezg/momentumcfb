begin;
-- cfb-picks-v14 adds the rivalry road-favourite rule: in a listed rivalry where the
-- home team is the underdog on its own field, spreads are priced 1.37 points toward
-- the road favourite, which qualifies at any positive edge (reason
-- rivalry_road_favorite, tier rivalry_rule). Earlier decisions keep their rules.
alter table cfb.recommendations drop constraint recommendation_eligibility_v13;
alter table cfb.recommendations add constraint recommendation_eligibility_v14
check ((status = 'no_play' and stake_units = 0) or
    (status = 'recommended' and stake_units = 1 and
     ((home_missing_input_count = 0 and away_missing_input_count = 0) or
      (policy_version in ('cfb-picks-v2', 'cfb-picks-v3', 'cfb-picks-v4', 'cfb-picks-v5', 'cfb-picks-v6', 'cfb-picks-v7', 'cfb-picks-v8', 'cfb-picks-v9', 'cfb-picks-v10', 'cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14')
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
      (policy_version in ('cfb-picks-v9', 'cfb-picks-v10', 'cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14') and pricing_context is not null
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
     (policy_version not in ('cfb-picks-v10', 'cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14') or (
       ((reason = 'qualifying_edge' and edge_points >= 2
         and pricing_context->>'recommendation_tier' = 'edge_qualified') or
        (market <> 'h2h' and reason = 'volume_floor' and edge_points < 2
         and pricing_context->>'recommendation_tier' = 'exploratory') or
        (policy_version = 'cfb-picks-v14' and market = 'spreads' and side = 'away'
         and reason = 'rivalry_road_favorite' and edge_points > 0
         and pricing_context->>'recommendation_tier' = 'rivalry_rule'
         and (pricing_context->>'rivalry_shift_points')::float8 = 1.37))
       and (market <> 'spreads' or (
         pricing_context->>'probability_source' = case
           when policy_version in ('cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14')
           then 'market_history_margin_blend_push_keys_v2'
           else 'market_history_margin_blend_keys_v1' end
         and pricing_context->>'calibration_version' = case
           when policy_version in ('cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14')
           then 'spread_keys_cover_v2' else 'spread_keys_020_v1' end
         and pricing_context->>'blend_space' = 'margin'
         and (pricing_context->>'model_weight')::float8 = (case when policy_version in ('cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14') then 0.0075 else 0.2 end)
         and margin_sd = 15.35 and degrees_of_freedom = 500
         and abs(model_home_margin - (pricing_context->>'pricing_home_margin')::float8) < 1e-12
         and abs(model_home_margin - (
           (pricing_context->>'market_home_margin')::float8 + (case when policy_version in ('cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14') then 0.0075 else 0.2 end)
           * ((pricing_context->>'model_side_home_margin')::float8
              - (pricing_context->>'market_home_margin')::float8))
           + coalesce((pricing_context->>'rivalry_shift_points')::float8, 0)) < 1e-12
         and (pricing_context->>'reference_as_of')::timestamptz = forecast_as_of
       ))
       and (market <> 'totals' or total_sd = 15.93)
       and (market <> 'totals' or policy_version not in ('cfb-picks-v13', 'cfb-picks-v14') or (
         pricing_context->>'probability_source' = 'market_total_blend_v1'
         and (pricing_context->>'model_weight')::float8 = 0.209
         and (pricing_context->>'market_total')::float8 = market_total
         and abs(model_total - (pricing_context->>'pricing_total')::float8) < 1e-9
         and abs(model_total - (market_total + 0.209
           * ((pricing_context->>'model_side_total')::float8 - market_total))) < 1e-9
       ))
     )) and
     (policy_version not in ('cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14') or (
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
           + coalesce((pricing_context->'spread_edge_breakdown'->>'rivalry_adjustment_points')::float8, 0)
           + (pricing_context->'spread_edge_breakdown'->>'quote_difference_points')::float8
           + (pricing_context->'spread_edge_breakdown'->>'price_distribution_adjustment_points')::float8
         )) < 1e-12
         and abs((pricing_context->'spread_edge_breakdown'->>'model_contribution_points')::float8
           - (case when policy_version in ('cfb-picks-v12', 'cfb-picks-v13', 'cfb-picks-v14') then 0.0075 else 0.2 end)
             * (pricing_context->'spread_edge_breakdown'->>'model_disagreement_points')::float8) < 1e-12
       ))
     )) and
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true);
-- Rivalry picks get their own qualification segment instead of 'legacy'.
create or replace view cfb.recommendation_performance as
 WITH segments AS (
         SELECT r.game_id,
            r.market,
            r.season,
            r.week,
            r.start_date,
            r.home_team,
            r.away_team,
            r.model_version,
            r.forecast_as_of,
            r.home_missing_input_count,
            r.away_missing_input_count,
            r.policy_version,
            r.decision_at,
            r.published_at,
            r.status,
            r.reason,
            r.selection,
            r.side,
            r.point,
            r.price,
            r.provider,
            r.provider_key,
            r.market_fetched_at,
            r.odds_api_event_id,
            r.provider_start_date,
            r.provider_last_update,
            r.match_score,
            r.win_probability,
            r.push_probability,
            r.probability_edge,
            r.expected_value_per_unit,
            r.stake_units,
            r.model_home_margin,
            r.model_total,
            r.margin_sd,
            r.total_sd,
            r.degrees_of_freedom,
            r.outcome,
            r.home_points,
            r.away_points,
            r.profit_units,
            r.graded_at,
            r.market_total,
            r.edge_points,
            r.settlement_reason,
            r.closing_point,
            r.closing_price,
            r.closing_source,
            r.clv_points,
            r.decision_forecast,
            r.pricing_context,
            r.decision_market_point,
            r.decision_market_probability,
            r.closing_probability,
            r.clv_market_points,
            r.clv_market_probability,
            s.kind,
            s.label
           FROM cfb.recommendations r
             CROSS JOIN LATERAL ( SELECT v.kind,
                    v.label
                   FROM ( VALUES ('overall'::text,'All picks'::text), ('market'::text,r.market), ('week'::text,(r.season::text || ' Week '::text) || r.week::text), ('policy'::text,r.policy_version), ('edge'::text,
                                CASE
                                    WHEN r.edge_points IS NULL THEN
                                    CASE
WHEN r.probability_edge < 0.075::double precision THEN '4.5-7.5 pp'::text
WHEN r.probability_edge < 0.10::double precision THEN '7.5-10 pp'::text
ELSE '10+ pp'::text
                                    END
                                    WHEN r.edge_points < 2::double precision THEN 'under 2 pts (floor)'::text
                                    WHEN r.edge_points < 4::double precision THEN '2-4 pts'::text
                                    WHEN r.edge_points < 7::double precision THEN '4-7 pts'::text
                                    ELSE '7+ pts'::text
                                END)) v(kind, label)
                UNION ALL
                 SELECT 'qualification'::text,
                        CASE
                            WHEN r.reason = 'volume_floor'::text THEN 'exploratory'::text
                            WHEN r.reason = 'qualifying_edge'::text THEN 'edge_qualified'::text
                            WHEN r.reason = 'rivalry_road_favorite'::text THEN 'rivalry_rule'::text
                            ELSE 'legacy'::text
                        END AS "case"
                  WHERE r.status = 'recommended'::text
                UNION ALL
                 SELECT 'side'::text,
                    (r.market || ':'::text) ||
                        CASE
                            WHEN r.market = 'totals'::text THEN r.side
                            WHEN r.market = 'spreads'::text THEN
                            CASE
                                WHEN r.point < 0::double precision THEN 'favorite'::text
                                WHEN r.point > 0::double precision THEN 'underdog'::text
                                ELSE 'pickem'::text
                            END
                            ELSE
                            CASE
                                WHEN r.price < '-100'::integer::double precision THEN 'favorite'::text
                                WHEN r.price > 100::double precision THEN 'underdog'::text
                                ELSE 'even'::text
                            END
                        END
                  WHERE r.status = 'recommended'::text) s(kind, label)
        ), aggregates AS (
         SELECT segments.season,
            segments.kind AS segment_kind,
            segments.label AS segment,
            count(*) FILTER (WHERE segments.status = 'recommended'::text) AS picks,
            count(*) FILTER (WHERE segments.status = 'no_play'::text) AS no_plays,
            count(*) FILTER (WHERE segments.status = 'recommended'::text AND segments.outcome = 'pending'::text) AS pending,
            count(*) FILTER (WHERE segments.outcome = 'win'::text) AS wins,
            count(*) FILTER (WHERE segments.outcome = 'loss'::text) AS losses,
            count(*) FILTER (WHERE segments.outcome = 'push'::text) AS pushes,
            count(*) FILTER (WHERE segments.status = 'recommended'::text AND segments.outcome = 'void'::text) AS voids,
            COALESCE(sum(segments.stake_units) FILTER (WHERE segments.outcome = ANY (ARRAY['win'::text, 'loss'::text, 'push'::text])), 0::double precision) AS staked_units,
            COALESCE(sum(segments.profit_units) FILTER (WHERE segments.outcome = ANY (ARRAY['win'::text, 'loss'::text, 'push'::text])), 0::double precision) AS profit_units,
            avg(segments.expected_value_per_unit) FILTER (WHERE segments.status = 'recommended'::text) AS average_ev,
            min(segments.decision_at) AS first_decision_at,
            max(segments.decision_at) AS last_decision_at,
            max(segments.graded_at) AS last_graded_at,
            avg(segments.clv_points) AS average_clv_points,
            count(segments.clv_points) AS clv_sample,
            count(*) FILTER (WHERE segments.clv_points > 0::double precision)::double precision / NULLIF(count(segments.clv_points), 0)::double precision AS clv_positive_share,
            avg(segments.clv_market_points) AS average_clv_market_points,
            count(segments.clv_market_points) AS clv_market_points_sample,
            count(*) FILTER (WHERE segments.clv_market_points > 0::double precision)::double precision / NULLIF(count(segments.clv_market_points), 0)::double precision AS clv_market_points_positive_share,
            avg(segments.clv_market_probability) AS average_clv_market_probability,
            count(segments.clv_market_probability) AS clv_market_probability_sample,
            count(*) FILTER (WHERE segments.clv_market_probability > 0::double precision)::double precision / NULLIF(count(segments.clv_market_probability), 0)::double precision AS clv_market_probability_positive_share
           FROM segments
          GROUP BY segments.season, segments.kind, segments.label
        )
 SELECT season,
    segment_kind,
    segment,
    picks,
    no_plays,
    pending,
    wins,
    losses,
    pushes,
    voids,
    staked_units,
    profit_units,
    average_ev,
    first_decision_at,
    last_decision_at,
    last_graded_at,
    average_clv_points,
    clv_sample,
    clv_positive_share,
    average_clv_market_points,
    clv_market_points_sample,
    clv_market_points_positive_share,
    average_clv_market_probability,
    clv_market_probability_sample,
    clv_market_probability_positive_share,
    profit_units / NULLIF(staked_units, 0::double precision) AS roi,
    wins::double precision / NULLIF(wins + losses, 0)::double precision AS win_rate,
    (wins + losses + pushes) < 30 AS thin_sample
   FROM aggregates;
commit;
