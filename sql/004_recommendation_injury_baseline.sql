begin;

-- Every forecast count includes unavailable injury data. Policy v2 preserves
-- that flag while rejecting any additional missing input. All price, timing,
-- probability and settlement protections remain in force.
alter table cfb.recommendations drop constraint if exists recommendations_check2;
alter table cfb.recommendations drop constraint if exists recommendation_eligibility_v2;
alter table cfb.recommendations add constraint recommendation_eligibility_v2
check ((status = 'no_play' and stake_units = 0) or
    (status = 'recommended' and stake_units = 1 and
     ((home_missing_input_count = 0 and away_missing_input_count = 0) or
      (policy_version = 'cfb-picks-v2' and home_missing_input_count between 0 and 1
       and away_missing_input_count between 0 and 1)) and
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
     expected_value_per_unit > 0 and expected_value_per_unit < 'Infinity'::float8) is true);

commit;
