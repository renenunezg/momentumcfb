--
-- PostgreSQL database dump
--

\restrict f35x3bPYoLzsexc67pmKoLwPHd6ObOahxELwrSHxAtEm58vUMqM6ANPQLbR38sk

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.9 (Postgres.app)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: cfb; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA cfb;


--
-- Name: protect_recommendation(); Type: FUNCTION; Schema: cfb; Owner: -
--

CREATE FUNCTION cfb.protect_recommendation() RETURNS trigger
    LANGUAGE plpgsql
    SET search_path TO 'pg_catalog', 'cfb'
    AS $$
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


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: recommendations; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.recommendations (
    game_id bigint NOT NULL,
    market text NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    start_date timestamp with time zone NOT NULL,
    home_team text NOT NULL,
    away_team text NOT NULL,
    model_version text NOT NULL,
    forecast_as_of timestamp with time zone NOT NULL,
    home_missing_input_count integer,
    away_missing_input_count integer,
    policy_version text NOT NULL,
    decision_at timestamp with time zone NOT NULL,
    published_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    status text NOT NULL,
    reason text NOT NULL,
    selection text,
    side text,
    point double precision,
    price double precision,
    provider text,
    provider_key text,
    market_fetched_at timestamp with time zone,
    odds_api_event_id text,
    provider_start_date timestamp with time zone,
    provider_last_update timestamp with time zone,
    match_score double precision,
    win_probability double precision,
    push_probability double precision,
    probability_edge double precision,
    expected_value_per_unit double precision,
    stake_units double precision NOT NULL,
    model_home_margin double precision NOT NULL,
    model_total double precision NOT NULL,
    margin_sd double precision NOT NULL,
    total_sd double precision NOT NULL,
    degrees_of_freedom double precision,
    outcome text DEFAULT 'pending'::text NOT NULL,
    home_points integer,
    away_points integer,
    profit_units double precision,
    graded_at timestamp with time zone,
    market_total double precision,
    edge_points double precision,
    settlement_reason text,
    closing_point double precision,
    closing_price double precision,
    closing_source text,
    clv_points double precision,
    CONSTRAINT recommendation_closing_line_value CHECK ((((closing_source IS NULL) AND (closing_point IS NULL) AND (closing_price IS NULL) AND (clv_points IS NULL)) OR (((closing_source IS NOT NULL) AND (status = 'recommended'::text) AND (outcome = ANY (ARRAY['win'::text, 'loss'::text, 'push'::text])) AND (((market = 'h2h'::text) AND (closing_price IS NOT NULL) AND (closing_point IS NULL) AND (clv_points IS NULL)) OR ((market = 'spreads'::text) AND (closing_point IS NOT NULL) AND (closing_price IS NULL) AND (abs((clv_points - (point - closing_point))) < (0.000000001)::double precision)) OR ((market = 'totals'::text) AND (closing_point IS NOT NULL) AND (closing_price IS NULL) AND (abs((clv_points -
CASE side
    WHEN 'over'::text THEN (closing_point - point)
    ELSE (point - closing_point)
END)) < (0.000000001)::double precision)))) IS TRUE))),
    CONSTRAINT recommendation_eligibility_v7 CHECK ((((status = 'no_play'::text) AND (stake_units = (0)::double precision)) OR (((status = 'recommended'::text) AND (stake_units = (1)::double precision) AND (((home_missing_input_count = 0) AND (away_missing_input_count = 0)) OR ((policy_version = ANY (ARRAY['cfb-picks-v2'::text, 'cfb-picks-v3'::text, 'cfb-picks-v4'::text, 'cfb-picks-v5'::text, 'cfb-picks-v6'::text, 'cfb-picks-v7'::text])) AND ((home_missing_input_count >= 0) AND (home_missing_input_count <= 1)) AND ((away_missing_input_count >= 0) AND (away_missing_input_count <= 1)))) AND (match_score >= (0.95)::double precision) AND (match_score <= (1)::double precision) AND (selection IS NOT NULL) AND (side IS NOT NULL) AND (((market = ANY (ARRAY['spreads'::text, 'h2h'::text])) AND (side = ANY (ARRAY['home'::text, 'away'::text]))) OR ((market = 'totals'::text) AND (side = ANY (ARRAY['over'::text, 'under'::text])))) AND (selection =
CASE side
    WHEN 'home'::text THEN home_team
    WHEN 'away'::text THEN away_team
    WHEN 'over'::text THEN 'Over'::text
    ELSE 'Under'::text
END) AND (((market = 'h2h'::text) AND (point IS NULL) AND (push_probability = (0)::double precision)) OR ((market <> 'h2h'::text) AND (point IS NOT NULL) AND (abs(point) < 'Infinity'::double precision) AND ((point * (2)::double precision) = round((point * (2)::double precision))))) AND (abs(price) >= (100)::double precision) AND (abs(price) < 'Infinity'::double precision) AND (provider IS NOT NULL) AND (provider_key IS NOT NULL) AND (odds_api_event_id IS NOT NULL) AND (provider_start_date = start_date) AND (provider_last_update <= market_fetched_at) AND (market_fetched_at <= decision_at) AND ((published_at - provider_last_update) <= '01:00:00'::interval) AND ((published_at - forecast_as_of) <= '7 days'::interval) AND (abs(model_home_margin) < 'Infinity'::double precision) AND (model_total >= (0)::double precision) AND (model_total < 'Infinity'::double precision) AND (margin_sd > (0)::double precision) AND (margin_sd < 'Infinity'::double precision) AND (total_sd > (0)::double precision) AND (total_sd < 'Infinity'::double precision) AND ((degrees_of_freedom IS NULL) OR (degrees_of_freedom > (2)::double precision)) AND (win_probability > (0)::double precision) AND (win_probability < (1)::double precision) AND (push_probability >= (0)::double precision) AND ((win_probability + push_probability) <= (1)::double precision) AND (probability_edge > ('-1'::integer)::double precision) AND (probability_edge < (1)::double precision) AND (((policy_version = ANY (ARRAY['cfb-picks-v2'::text, 'cfb-picks-v3'::text, 'cfb-picks-v4'::text])) AND (probability_edge >= (0.045)::double precision)) OR ((policy_version = 'cfb-picks-v5'::text) AND (edge_points >= (2)::double precision) AND (edge_points < 'Infinity'::double precision)) OR ((policy_version = ANY (ARRAY['cfb-picks-v6'::text, 'cfb-picks-v7'::text])) AND (edge_points > (0)::double precision) AND (edge_points < 'Infinity'::double precision))) AND (expected_value_per_unit > (0)::double precision) AND (expected_value_per_unit < 'Infinity'::double precision)) IS TRUE))),
    CONSTRAINT recommendation_market_total_finite CHECK (((market_total IS NULL) OR ((market_total >= (0)::double precision) AND (market_total < 'Infinity'::double precision)))),
    CONSTRAINT recommendations_check CHECK (((forecast_as_of <= decision_at) AND (decision_at <= published_at))),
    CONSTRAINT recommendations_check1 CHECK ((published_at < start_date)),
    CONSTRAINT recommendations_check3 CHECK ((((outcome = 'pending'::text) AND (graded_at IS NULL) AND (profit_units IS NULL)) OR ((outcome <> 'pending'::text) AND (graded_at IS NOT NULL) AND (profit_units IS NOT NULL)))),
    CONSTRAINT recommendations_market_check CHECK ((market = ANY (ARRAY['h2h'::text, 'spreads'::text, 'totals'::text]))),
    CONSTRAINT recommendations_outcome_check CHECK ((outcome = ANY (ARRAY['pending'::text, 'win'::text, 'loss'::text, 'push'::text, 'void'::text, 'no_play'::text]))),
    CONSTRAINT recommendations_side_check CHECK ((side = ANY (ARRAY['home'::text, 'away'::text, 'over'::text, 'under'::text]))),
    CONSTRAINT recommendations_status_check CHECK ((status = ANY (ARRAY['recommended'::text, 'no_play'::text])))
);


--
-- Name: recommendation_performance; Type: VIEW; Schema: cfb; Owner: -
--

CREATE VIEW cfb.recommendation_performance WITH (security_invoker='true') AS
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
            s.kind,
            s.label
           FROM (cfb.recommendations r
             CROSS JOIN LATERAL ( SELECT v.kind,
                    v.label
                   FROM ( VALUES ('overall'::text,'All picks'::text), ('market'::text,r.market), ('week'::text,(((r.season)::text || ' Week '::text) || (r.week)::text)), ('policy'::text,r.policy_version), ('edge'::text,
                                CASE
                                    WHEN (r.edge_points IS NULL) THEN
                                    CASE
WHEN (r.probability_edge < (0.075)::double precision) THEN '4.5-7.5 pp'::text
WHEN (r.probability_edge < (0.10)::double precision) THEN '7.5-10 pp'::text
ELSE '10+ pp'::text
                                    END
                                    WHEN (r.edge_points < (2)::double precision) THEN 'under 2 pts (floor)'::text
                                    WHEN (r.edge_points < (4)::double precision) THEN '2-4 pts'::text
                                    WHEN (r.edge_points < (7)::double precision) THEN '4-7 pts'::text
                                    ELSE '7+ pts'::text
                                END)) v(kind, label)
                UNION ALL
                 SELECT 'side'::text,
                    ((r.market || ':'::text) ||
                        CASE
                            WHEN (r.market = 'totals'::text) THEN r.side
                            WHEN (r.market = 'spreads'::text) THEN
                            CASE
                                WHEN (r.point < (0)::double precision) THEN 'favorite'::text
                                WHEN (r.point > (0)::double precision) THEN 'underdog'::text
                                ELSE 'pickem'::text
                            END
                            ELSE
                            CASE
                                WHEN (r.price < ('-100'::integer)::double precision) THEN 'favorite'::text
                                WHEN (r.price > (100)::double precision) THEN 'underdog'::text
                                ELSE 'even'::text
                            END
                        END)
                  WHERE (r.status = 'recommended'::text)) s(kind, label))
        ), aggregates AS (
         SELECT segments.season,
            segments.kind AS segment_kind,
            segments.label AS segment,
            count(*) FILTER (WHERE (segments.status = 'recommended'::text)) AS picks,
            count(*) FILTER (WHERE (segments.status = 'no_play'::text)) AS no_plays,
            count(*) FILTER (WHERE ((segments.status = 'recommended'::text) AND (segments.outcome = 'pending'::text))) AS pending,
            count(*) FILTER (WHERE (segments.outcome = 'win'::text)) AS wins,
            count(*) FILTER (WHERE (segments.outcome = 'loss'::text)) AS losses,
            count(*) FILTER (WHERE (segments.outcome = 'push'::text)) AS pushes,
            count(*) FILTER (WHERE ((segments.status = 'recommended'::text) AND (segments.outcome = 'void'::text))) AS voids,
            COALESCE(sum(segments.stake_units) FILTER (WHERE (segments.outcome = ANY (ARRAY['win'::text, 'loss'::text, 'push'::text]))), (0)::double precision) AS staked_units,
            COALESCE(sum(segments.profit_units) FILTER (WHERE (segments.outcome = ANY (ARRAY['win'::text, 'loss'::text, 'push'::text]))), (0)::double precision) AS profit_units,
            avg(segments.expected_value_per_unit) FILTER (WHERE (segments.status = 'recommended'::text)) AS average_ev,
            min(segments.decision_at) AS first_decision_at,
            max(segments.decision_at) AS last_decision_at,
            max(segments.graded_at) AS last_graded_at,
            avg(segments.clv_points) AS average_clv_points,
            count(segments.clv_points) AS clv_sample,
            ((count(*) FILTER (WHERE (segments.clv_points > (0)::double precision)))::double precision / (NULLIF(count(segments.clv_points), 0))::double precision) AS clv_positive_share
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
    (profit_units / NULLIF(staked_units, (0)::double precision)) AS roi,
    ((wins)::double precision / (NULLIF((wins + losses), 0))::double precision) AS win_rate,
    (((wins + losses) + pushes) < 30) AS thin_sample
   FROM aggregates;


--
-- Name: recommendation_summary(integer, text, timestamp with time zone); Type: FUNCTION; Schema: cfb; Owner: -
--

CREATE FUNCTION cfb.recommendation_summary(p_season integer DEFAULT NULL::integer, p_market text DEFAULT 'all'::text, p_from timestamp with time zone DEFAULT NULL::timestamp with time zone) RETURNS SETOF cfb.recommendation_performance
    LANGUAGE sql STABLE
    SET search_path TO 'pg_catalog', 'cfb'
    AS $$
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


--
-- Name: backtest_predictions; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.backtest_predictions (
    game_id bigint NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    week_index integer,
    season_type text,
    home_team text NOT NULL,
    away_team text NOT NULL,
    neutral_site boolean,
    home_points integer,
    away_points integer,
    margin double precision,
    closing_spread double precision,
    model_margin double precision,
    actual_margin double precision
);


--
-- Name: game_projections; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.game_projections (
    game_id bigint NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    as_of timestamp with time zone NOT NULL,
    model_version text NOT NULL,
    start_date timestamp with time zone,
    home_team_id bigint,
    home_team text NOT NULL,
    away_team_id bigint,
    away_team text NOT NULL,
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
    market_informed_home_spread double precision,
    home_qb_out boolean,
    away_qb_out boolean,
    qb_availability_points double precision,
    market_history_home_margin double precision,
    market_history_weight double precision,
    market_informed_total double precision,
    market_informed_home_points double precision,
    market_informed_away_points double precision,
    market_total_weight double precision
);


--
-- Name: graded_games; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.graded_games (
    game_id bigint NOT NULL,
    season integer NOT NULL,
    week integer NOT NULL,
    season_type text,
    forecast_week integer,
    start_date timestamp with time zone,
    neutral_site boolean,
    conference_game boolean,
    home_team_id bigint,
    home_team text,
    away_team_id bigint,
    away_team text,
    home_classification text,
    away_classification text,
    home_missing_input_count integer,
    away_missing_input_count integer,
    model_version text,
    forecast_as_of timestamp with time zone,
    pure_home_margin double precision,
    market_informed_home_margin double precision,
    market_weight double precision,
    forecast_market_home_spread double precision,
    model_total double precision,
    margin_sd double precision,
    total_sd double precision,
    distribution text,
    degrees_of_freedom double precision,
    home_win_probability double precision,
    probability_method text,
    closing_spread double precision,
    closing_total double precision,
    n_spread_offers integer,
    n_total_offers integer,
    closing_source text,
    home_points integer,
    away_points integer,
    actual_margin integer,
    actual_total integer,
    score_source text,
    source_ingested_at timestamp with time zone,
    graded_at timestamp with time zone,
    market_informed_total double precision
);


--
-- Name: heisman_board; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.heisman_board (
    season integer NOT NULL,
    week integer NOT NULL,
    as_of timestamp with time zone NOT NULL,
    model_version text NOT NULL,
    athlete_id text NOT NULL,
    athlete_name text NOT NULL,
    team text NOT NULL,
    "position" text,
    games double precision NOT NULL,
    predicted_share double precision NOT NULL,
    predicted_rank integer NOT NULL,
    value_rank integer,
    rank_gap integer,
    win_pct double precision,
    ap_rank double precision,
    pass_yards double precision,
    pass_touchdowns double precision,
    interceptions double precision,
    rush_yards double precision,
    rush_touchdowns double precision,
    receiving_yards double precision,
    receiving_touchdowns double precision,
    total_touchdowns double precision,
    tackles double precision,
    sacks double precision,
    defensive_interceptions double precision
);


--
-- Name: heisman_history; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.heisman_history (
    season integer NOT NULL,
    actual_winner text NOT NULL,
    actual_winner_team text NOT NULL,
    actual_share double precision NOT NULL,
    predicted_winner text NOT NULL,
    predicted_winner_team text NOT NULL,
    predicted_winner_share double precision NOT NULL,
    actual_winner_predicted_share double precision NOT NULL,
    actual_winner_predicted_rank integer,
    winner_hit boolean NOT NULL,
    top_three_hit boolean NOT NULL,
    winner_value_rank integer
);


--
-- Name: live_win_probability; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.live_win_probability (
    game_id bigint NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    payload jsonb NOT NULL,
    CONSTRAINT live_win_probability_check CHECK ((((payload ->> 'schema_version'::text) = '1'::text) AND (((payload ->> 'game_id'::text))::bigint = game_id))),
    CONSTRAINT live_win_probability_game_id_check CHECK ((game_id > 0))
);


--
-- Name: market_comparisons; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.market_comparisons (
    game_id bigint NOT NULL,
    start_date timestamp with time zone,
    home_team text,
    away_team text,
    model_home_spread double precision,
    model_total double precision,
    margin_sd double precision,
    total_sd double precision,
    model_as_of timestamp with time zone,
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
    best_offer_provider_last_update timestamp with time zone,
    best_offer_event_link text,
    best_offer_market_link text,
    best_offer_bet_link text,
    best_offer_edge_points double precision,
    best_offer_edge_standardized double precision,
    best_offer_model_cover_probability double precision,
    best_offer_model_fair_price double precision,
    best_offer_expected_value_per_unit double precision
);


--
-- Name: performance_metrics; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.performance_metrics (
    season integer NOT NULL,
    prediction_source text NOT NULL,
    segment_kind text NOT NULL,
    segment text NOT NULL,
    segment_order integer,
    games integer,
    thin_sample boolean,
    margin_mae double precision,
    margin_rmse double precision,
    margin_bias double precision,
    total_games integer,
    total_mae double precision,
    total_rmse double precision,
    total_bias double precision,
    coverage_50 double precision,
    coverage_80 double precision,
    coverage_90 double precision,
    games_with_market integer,
    market_mae double precision,
    model_minus_market_mae double precision,
    closer_than_market_share double precision,
    probability_games integer,
    brier_score double precision,
    log_loss double precision,
    computed_at timestamp with time zone
);


--
-- Name: player_model_meta; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.player_model_meta (
    season integer NOT NULL,
    as_of timestamp with time zone NOT NULL,
    value_model_version text NOT NULL,
    heisman_model_version text NOT NULL,
    credit_shares text NOT NULL,
    replacement_percentile double precision NOT NULL,
    fcs_opponent_weight double precision NOT NULL,
    reliability text NOT NULL,
    heisman_training_seasons text NOT NULL,
    heisman_winner_hit_rate double precision NOT NULL,
    heisman_top_three_rate double precision NOT NULL,
    heisman_coefficients text NOT NULL,
    opponent_effect_prior_games double precision DEFAULT 3 NOT NULL,
    prior_games double precision DEFAULT 4 NOT NULL,
    qualifying_games integer DEFAULT 6 NOT NULL,
    heisman_evaluation_kind text,
    heisman_evaluation_week integer,
    heisman_evaluation_seasons integer,
    heisman_winner_pool_coverage double precision,
    heisman_ballot_share_covered double precision
);


--
-- Name: player_values; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.player_values (
    season integer NOT NULL,
    week integer NOT NULL,
    as_of timestamp with time zone NOT NULL,
    model_version text NOT NULL,
    athlete_id text NOT NULL,
    athlete_name text NOT NULL,
    team text NOT NULL,
    team_id bigint,
    classification text,
    "position" text,
    position_group text NOT NULL,
    games integer NOT NULL,
    plays double precision NOT NULL,
    raw_epa double precision NOT NULL,
    adjusted_epa double precision NOT NULL,
    adjusted_rate double precision NOT NULL,
    value_above_replacement double precision NOT NULL,
    wpa double precision NOT NULL,
    fcs_play_share double precision NOT NULL,
    overall_rank integer NOT NULL,
    position_rank integer NOT NULL,
    adjusted_per_game double precision DEFAULT 0 NOT NULL,
    shrunk_per_game double precision DEFAULT 0 NOT NULL,
    replacement_per_game double precision DEFAULT 0 NOT NULL
);


--
-- Name: qb_availability; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.qb_availability (
    season integer NOT NULL,
    week integer NOT NULL,
    team text NOT NULL,
    player text NOT NULL,
    status text NOT NULL,
    source text NOT NULL,
    source_url text,
    reported_at timestamp with time zone NOT NULL,
    notes text,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT qb_availability_status_check CHECK ((status = ANY (ARRAY['out'::text, 'doubtful'::text, 'questionable'::text, 'probable'::text])))
);


--
-- Name: serving_anchors; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.serving_anchors (
    season integer NOT NULL,
    anchor_week integer NOT NULL,
    game_id bigint NOT NULL,
    model_week integer NOT NULL,
    home_margin double precision NOT NULL,
    margin_sd double precision NOT NULL,
    closing_spread double precision,
    n_spread_offers integer,
    margin_sd_method text,
    market_anchor_source text,
    closing_snapshot_id text,
    closing_fetched_at timestamp with time zone,
    latest_provider_update timestamp with time zone,
    published_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: TABLE serving_anchors; Type: COMMENT; Schema: cfb; Owner: -
--

COMMENT ON TABLE cfb.serving_anchors IS 'Serving anchor artifacts keyed by artifact week: anchor_week 0 is the frozen market closing-spread capture, anchor_week >= 1 is a projection artifact for that week. model_week is the contract column served to the in-game model.';


--
-- Name: team_ratings; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.team_ratings (
    season integer NOT NULL,
    week integer NOT NULL,
    as_of timestamp with time zone NOT NULL,
    model_version text NOT NULL,
    team_id bigint NOT NULL,
    team text NOT NULL,
    conference text,
    classification text,
    offense_points double precision,
    defense_points double precision,
    power_rating double precision NOT NULL,
    scoring_environment double precision,
    expected_possessions double precision,
    power_rating_sd double precision,
    missing_input_count integer,
    market_rating double precision,
    market_rating_sd double precision,
    market_rating_games integer,
    forecast_alignment_points double precision
);


--
-- Name: COLUMN team_ratings.market_rating; Type: COMMENT; Schema: cfb; Owner: -
--

COMMENT ON COLUMN cfb.team_ratings.market_rating IS 'Neutral-field points above average FBS, fitted to prior-week closing spreads with a carried market prior. Same forecast snapshot as the model rating.';


--
-- Name: COLUMN team_ratings.market_rating_sd; Type: COMMENT; Schema: cfb; Owner: -
--

COMMENT ON COLUMN cfb.team_ratings.market_rating_sd IS 'Conditional rating posterior SD, not game-margin or future-line prediction SD.';


--
-- Name: COLUMN team_ratings.market_rating_games; Type: COMMENT; Schema: cfb; Owner: -
--

COMMENT ON COLUMN cfb.team_ratings.market_rating_games IS 'Current-season earlier games with a closing spread used for this team. Zero means prior-only strength.';


--
-- Name: team_unit_ratings; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.team_unit_ratings (
    season integer NOT NULL,
    week integer NOT NULL,
    as_of timestamp with time zone NOT NULL,
    model_version text NOT NULL,
    source_season integer,
    team_id bigint NOT NULL,
    team text NOT NULL,
    classification text,
    unit_history_missing boolean DEFAULT false NOT NULL,
    rush_offense double precision,
    pass_offense double precision,
    rush_defense double precision,
    pass_defense double precision,
    pass_block double precision,
    run_block double precision
);


--
-- Name: teams; Type: TABLE; Schema: cfb; Owner: -
--

CREATE TABLE cfb.teams (
    team_id integer NOT NULL,
    team text NOT NULL,
    color text,
    alternate_color text,
    logo_light text,
    logo_dark text
);


--
-- Name: backtest_predictions backtest_predictions_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.backtest_predictions
    ADD CONSTRAINT backtest_predictions_pkey PRIMARY KEY (game_id);


--
-- Name: game_projections game_projections_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.game_projections
    ADD CONSTRAINT game_projections_pkey PRIMARY KEY (game_id);


--
-- Name: graded_games graded_games_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.graded_games
    ADD CONSTRAINT graded_games_pkey PRIMARY KEY (game_id);


--
-- Name: heisman_board heisman_board_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.heisman_board
    ADD CONSTRAINT heisman_board_pkey PRIMARY KEY (season, week, athlete_id);


--
-- Name: heisman_history heisman_history_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.heisman_history
    ADD CONSTRAINT heisman_history_pkey PRIMARY KEY (season);


--
-- Name: live_win_probability live_win_probability_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.live_win_probability
    ADD CONSTRAINT live_win_probability_pkey PRIMARY KEY (game_id);


--
-- Name: market_comparisons market_comparisons_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.market_comparisons
    ADD CONSTRAINT market_comparisons_pkey PRIMARY KEY (game_id);


--
-- Name: performance_metrics performance_metrics_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.performance_metrics
    ADD CONSTRAINT performance_metrics_pkey PRIMARY KEY (season, prediction_source, segment_kind, segment);


--
-- Name: player_model_meta player_model_meta_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.player_model_meta
    ADD CONSTRAINT player_model_meta_pkey PRIMARY KEY (season);


--
-- Name: player_values player_values_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.player_values
    ADD CONSTRAINT player_values_pkey PRIMARY KEY (season, week, athlete_id);


--
-- Name: qb_availability qb_availability_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.qb_availability
    ADD CONSTRAINT qb_availability_pkey PRIMARY KEY (season, week, team);


--
-- Name: recommendations recommendations_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.recommendations
    ADD CONSTRAINT recommendations_pkey PRIMARY KEY (game_id, market);


--
-- Name: serving_anchors serving_anchors_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.serving_anchors
    ADD CONSTRAINT serving_anchors_pkey PRIMARY KEY (season, anchor_week, game_id);


--
-- Name: team_ratings team_ratings_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.team_ratings
    ADD CONSTRAINT team_ratings_pkey PRIMARY KEY (season, week, team_id);


--
-- Name: team_unit_ratings team_unit_ratings_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.team_unit_ratings
    ADD CONSTRAINT team_unit_ratings_pkey PRIMARY KEY (season, week, team_id);


--
-- Name: teams teams_pkey; Type: CONSTRAINT; Schema: cfb; Owner: -
--

ALTER TABLE ONLY cfb.teams
    ADD CONSTRAINT teams_pkey PRIMARY KEY (team_id);


--
-- Name: backtest_predictions_season_week; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX backtest_predictions_season_week ON cfb.backtest_predictions USING btree (season, week);


--
-- Name: game_projections_season_week; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX game_projections_season_week ON cfb.game_projections USING btree (season, week);


--
-- Name: graded_games_season_week_idx; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX graded_games_season_week_idx ON cfb.graded_games USING btree (season, week);


--
-- Name: player_values_season_week_rank_idx; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX player_values_season_week_rank_idx ON cfb.player_values USING btree (season, week, overall_rank);


--
-- Name: recommendations_decision_history; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX recommendations_decision_history ON cfb.recommendations USING btree (decision_at DESC, game_id, market);


--
-- Name: recommendations_season_decision_history; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX recommendations_season_decision_history ON cfb.recommendations USING btree (season, decision_at DESC, game_id, market);


--
-- Name: recommendations_season_start; Type: INDEX; Schema: cfb; Owner: -
--

CREATE INDEX recommendations_season_start ON cfb.recommendations USING btree (season, start_date DESC, game_id, market);


--
-- Name: recommendations protect_recommendation; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER protect_recommendation BEFORE INSERT OR DELETE OR UPDATE ON cfb.recommendations FOR EACH ROW EXECUTE FUNCTION cfb.protect_recommendation();


--
-- Name: backtest_predictions site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.backtest_predictions FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: game_projections site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.game_projections FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: graded_games site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.graded_games FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: heisman_board site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.heisman_board FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: heisman_history site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.heisman_history FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: market_comparisons site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.market_comparisons FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: performance_metrics site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.performance_metrics FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: player_model_meta site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.player_model_meta FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: player_values site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.player_values FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: recommendations site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.recommendations FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: team_ratings site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.team_ratings FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: team_unit_ratings site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.team_unit_ratings FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: teams site_revalidate; Type: TRIGGER; Schema: cfb; Owner: -
--

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.teams FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();


--
-- Name: backtest_predictions; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.backtest_predictions ENABLE ROW LEVEL SECURITY;

--
-- Name: game_projections; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.game_projections ENABLE ROW LEVEL SECURITY;

--
-- Name: graded_games; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.graded_games ENABLE ROW LEVEL SECURITY;

--
-- Name: heisman_board; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.heisman_board ENABLE ROW LEVEL SECURITY;

--
-- Name: heisman_history; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.heisman_history ENABLE ROW LEVEL SECURITY;

--
-- Name: live_win_probability; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.live_win_probability ENABLE ROW LEVEL SECURITY;

--
-- Name: market_comparisons; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.market_comparisons ENABLE ROW LEVEL SECURITY;

--
-- Name: performance_metrics; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.performance_metrics ENABLE ROW LEVEL SECURITY;

--
-- Name: player_model_meta; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.player_model_meta ENABLE ROW LEVEL SECURITY;

--
-- Name: player_values; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.player_values ENABLE ROW LEVEL SECURITY;

--
-- Name: team_unit_ratings public read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY "public read" ON cfb.team_unit_ratings FOR SELECT TO authenticated, anon USING (true);


--
-- Name: backtest_predictions public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.backtest_predictions FOR SELECT TO authenticated, anon USING (true);


--
-- Name: game_projections public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.game_projections FOR SELECT TO authenticated, anon USING (true);


--
-- Name: graded_games public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.graded_games FOR SELECT TO authenticated, anon USING (true);


--
-- Name: heisman_board public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.heisman_board FOR SELECT TO authenticated, anon USING (true);


--
-- Name: heisman_history public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.heisman_history FOR SELECT TO authenticated, anon USING (true);


--
-- Name: live_win_probability public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.live_win_probability FOR SELECT TO authenticated, anon USING (true);


--
-- Name: market_comparisons public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.market_comparisons FOR SELECT TO authenticated, anon USING (true);


--
-- Name: performance_metrics public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.performance_metrics FOR SELECT TO authenticated, anon USING (true);


--
-- Name: player_model_meta public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.player_model_meta FOR SELECT TO authenticated, anon USING (true);


--
-- Name: player_values public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.player_values FOR SELECT TO authenticated, anon USING (true);


--
-- Name: qb_availability public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.qb_availability FOR SELECT TO authenticated, anon USING (true);


--
-- Name: recommendations public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.recommendations FOR SELECT TO authenticated, anon USING (true);


--
-- Name: serving_anchors public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.serving_anchors FOR SELECT TO authenticated, anon USING (true);


--
-- Name: team_ratings public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.team_ratings FOR SELECT TO authenticated, anon USING (true);


--
-- Name: teams public_read; Type: POLICY; Schema: cfb; Owner: -
--

CREATE POLICY public_read ON cfb.teams FOR SELECT TO authenticated, anon USING (true);


--
-- Name: qb_availability; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.qb_availability ENABLE ROW LEVEL SECURITY;

--
-- Name: recommendations; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.recommendations ENABLE ROW LEVEL SECURITY;

--
-- Name: serving_anchors; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.serving_anchors ENABLE ROW LEVEL SECURITY;

--
-- Name: team_ratings; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.team_ratings ENABLE ROW LEVEL SECURITY;

--
-- Name: team_unit_ratings; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.team_unit_ratings ENABLE ROW LEVEL SECURITY;

--
-- Name: teams; Type: ROW SECURITY; Schema: cfb; Owner: -
--

ALTER TABLE cfb.teams ENABLE ROW LEVEL SECURITY;

--
-- Name: SCHEMA cfb; Type: ACL; Schema: -; Owner: -
--

GRANT USAGE ON SCHEMA cfb TO anon;
GRANT USAGE ON SCHEMA cfb TO authenticated;
GRANT USAGE ON SCHEMA cfb TO service_role;


--
-- Name: TABLE recommendations; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.recommendations TO anon;
GRANT SELECT ON TABLE cfb.recommendations TO authenticated;
GRANT SELECT,INSERT,UPDATE ON TABLE cfb.recommendations TO service_role;


--
-- Name: TABLE recommendation_performance; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.recommendation_performance TO anon;
GRANT SELECT ON TABLE cfb.recommendation_performance TO authenticated;
GRANT ALL ON TABLE cfb.recommendation_performance TO service_role;


--
-- Name: FUNCTION recommendation_summary(p_season integer, p_market text, p_from timestamp with time zone); Type: ACL; Schema: cfb; Owner: -
--

REVOKE ALL ON FUNCTION cfb.recommendation_summary(p_season integer, p_market text, p_from timestamp with time zone) FROM PUBLIC;
GRANT ALL ON FUNCTION cfb.recommendation_summary(p_season integer, p_market text, p_from timestamp with time zone) TO anon;
GRANT ALL ON FUNCTION cfb.recommendation_summary(p_season integer, p_market text, p_from timestamp with time zone) TO authenticated;
GRANT ALL ON FUNCTION cfb.recommendation_summary(p_season integer, p_market text, p_from timestamp with time zone) TO service_role;


--
-- Name: TABLE backtest_predictions; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.backtest_predictions TO anon;
GRANT SELECT ON TABLE cfb.backtest_predictions TO authenticated;
GRANT ALL ON TABLE cfb.backtest_predictions TO service_role;


--
-- Name: TABLE game_projections; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.game_projections TO anon;
GRANT SELECT ON TABLE cfb.game_projections TO authenticated;
GRANT ALL ON TABLE cfb.game_projections TO service_role;


--
-- Name: TABLE graded_games; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.graded_games TO anon;
GRANT SELECT ON TABLE cfb.graded_games TO authenticated;
GRANT ALL ON TABLE cfb.graded_games TO service_role;


--
-- Name: TABLE heisman_board; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.heisman_board TO anon;
GRANT SELECT ON TABLE cfb.heisman_board TO authenticated;
GRANT ALL ON TABLE cfb.heisman_board TO service_role;


--
-- Name: TABLE heisman_history; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.heisman_history TO anon;
GRANT SELECT ON TABLE cfb.heisman_history TO authenticated;
GRANT ALL ON TABLE cfb.heisman_history TO service_role;


--
-- Name: TABLE live_win_probability; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.live_win_probability TO anon;
GRANT SELECT ON TABLE cfb.live_win_probability TO authenticated;
GRANT ALL ON TABLE cfb.live_win_probability TO service_role;


--
-- Name: TABLE market_comparisons; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.market_comparisons TO anon;
GRANT SELECT ON TABLE cfb.market_comparisons TO authenticated;
GRANT ALL ON TABLE cfb.market_comparisons TO service_role;


--
-- Name: TABLE performance_metrics; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.performance_metrics TO anon;
GRANT SELECT ON TABLE cfb.performance_metrics TO authenticated;
GRANT ALL ON TABLE cfb.performance_metrics TO service_role;


--
-- Name: TABLE player_model_meta; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.player_model_meta TO anon;
GRANT SELECT ON TABLE cfb.player_model_meta TO authenticated;
GRANT ALL ON TABLE cfb.player_model_meta TO service_role;


--
-- Name: TABLE player_values; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.player_values TO anon;
GRANT SELECT ON TABLE cfb.player_values TO authenticated;
GRANT ALL ON TABLE cfb.player_values TO service_role;


--
-- Name: TABLE qb_availability; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.qb_availability TO anon;
GRANT SELECT ON TABLE cfb.qb_availability TO authenticated;
GRANT ALL ON TABLE cfb.qb_availability TO service_role;


--
-- Name: TABLE serving_anchors; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.serving_anchors TO anon;
GRANT SELECT ON TABLE cfb.serving_anchors TO authenticated;
GRANT ALL ON TABLE cfb.serving_anchors TO service_role;


--
-- Name: TABLE team_ratings; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.team_ratings TO anon;
GRANT SELECT ON TABLE cfb.team_ratings TO authenticated;
GRANT ALL ON TABLE cfb.team_ratings TO service_role;


--
-- Name: TABLE team_unit_ratings; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.team_unit_ratings TO anon;
GRANT SELECT ON TABLE cfb.team_unit_ratings TO authenticated;
GRANT ALL ON TABLE cfb.team_unit_ratings TO service_role;


--
-- Name: TABLE teams; Type: ACL; Schema: cfb; Owner: -
--

GRANT SELECT ON TABLE cfb.teams TO anon;
GRANT SELECT ON TABLE cfb.teams TO authenticated;
GRANT ALL ON TABLE cfb.teams TO service_role;


--
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: cfb; Owner: -
--

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA cfb GRANT SELECT ON TABLES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA cfb GRANT SELECT ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA cfb GRANT ALL ON TABLES TO service_role;


--
-- PostgreSQL database dump complete
--

\unrestrict f35x3bPYoLzsexc67pmKoLwPHd6ObOahxELwrSHxAtEm58vUMqM6ANPQLbR38sk

