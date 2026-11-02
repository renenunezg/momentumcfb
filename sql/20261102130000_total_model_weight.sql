begin;
-- cfb-picks-v13 prices totals at the development-fitted 0.209 share of the
-- model side's lean from the decision-time median; earlier decisions keep their rules.
alter table cfb.recommendations drop constraint recommendation_eligibility_v12;
alter table cfb.recommendations add constraint recommendation_eligibility_v13
check ((status = 'no_play' and stake_units = 0) or
    (status = 'recommended' and stake_units = 1 and
     ((home_missing_input_count = 0 and away_missing_input_count = 0) or
      (policy_version in ('cfb-picks-v2', 'cfb-picks-v3', 'cfb-picks-v4', 'cfb-picks-v5', 'cfb-picks-v6', 'cfb-picks-v7', 'cfb-picks-v8', 'cfb-picks-v9', 'cfb-picks-v10', 'cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13')
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
      (policy_version in ('cfb-picks-v9', 'cfb-picks-v10', 'cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13') and pricing_context is not null
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
     (policy_version not in ('cfb-picks-v10', 'cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13') or (
       ((reason = 'qualifying_edge' and edge_points >= 2
         and pricing_context->>'recommendation_tier' = 'edge_qualified') or
        (market <> 'h2h' and reason = 'volume_floor' and edge_points < 2
         and pricing_context->>'recommendation_tier' = 'exploratory'))
       and (market <> 'spreads' or (
         pricing_context->>'probability_source' = case
           when policy_version in ('cfb-picks-v12', 'cfb-picks-v13')
           then 'market_history_margin_blend_push_keys_v2'
           else 'market_history_margin_blend_keys_v1' end
         and pricing_context->>'calibration_version' = case
           when policy_version in ('cfb-picks-v12', 'cfb-picks-v13')
           then 'spread_keys_cover_v2' else 'spread_keys_020_v1' end
         and pricing_context->>'blend_space' = 'margin'
         and (pricing_context->>'model_weight')::float8 = (case when policy_version in ('cfb-picks-v12', 'cfb-picks-v13') then 0.0075 else 0.2 end)
         and margin_sd = 15.35 and degrees_of_freedom = 500
         and abs(model_home_margin - (pricing_context->>'pricing_home_margin')::float8) < 1e-12
         and abs(model_home_margin - (
           (pricing_context->>'market_home_margin')::float8 + (case when policy_version in ('cfb-picks-v12', 'cfb-picks-v13') then 0.0075 else 0.2 end)
           * ((pricing_context->>'model_side_home_margin')::float8
              - (pricing_context->>'market_home_margin')::float8))) < 1e-12
         and (pricing_context->>'reference_as_of')::timestamptz = forecast_as_of
       ))
       and (market <> 'totals' or total_sd = 15.93)
       and (market <> 'totals' or policy_version <> 'cfb-picks-v13' or (
         pricing_context->>'probability_source' = 'market_total_blend_v1'
         and (pricing_context->>'model_weight')::float8 = 0.209
         and (pricing_context->>'market_total')::float8 = market_total
         and abs(model_total - (pricing_context->>'pricing_total')::float8) < 1e-9
         and abs(model_total - (market_total + 0.209
           * ((pricing_context->>'model_side_total')::float8 - market_total))) < 1e-9
       ))
     )) and
     (policy_version not in ('cfb-picks-v11', 'cfb-picks-v12', 'cfb-picks-v13') or (
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
           - (case when policy_version in ('cfb-picks-v12', 'cfb-picks-v13') then 0.0075 else 0.2 end)
             * (pricing_context->'spread_edge_breakdown'->>'model_disagreement_points')::float8) < 1e-12
       ))
     )) and
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true);
commit;
