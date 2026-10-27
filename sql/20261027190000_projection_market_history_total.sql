begin;

-- The market-informed total mixes a rating fitted to earlier games' closing
-- totals into its model side, the way market_history_home_margin does for the
-- margin. Storing that total and its weight keeps the published blend
-- reproducible and lets a late quarterback report scale by the share of the
-- total that is still the model's.
alter table cfb.game_projections
  add column if not exists market_history_total double precision,
  add column if not exists market_history_total_weight double precision;

commit;
