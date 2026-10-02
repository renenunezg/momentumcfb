begin;

-- The site publishes the market-informed forecast. Its line was already
-- stored; the total and the two scores that agree with that line are stored
-- beside the pure model's so a published score never contradicts the
-- published line. Rows published before this migration keep NULL.
alter table cfb.game_projections
  add column if not exists market_informed_total double precision,
  add column if not exists market_informed_home_points double precision,
  add column if not exists market_informed_away_points double precision;

commit;
