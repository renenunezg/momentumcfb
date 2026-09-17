begin;

-- The market-informed margin mixes a rating fitted to earlier games' closing
-- lines into its model side. Storing that margin and its weight keeps the
-- published blend reproducible and lets a late pick refresh scale a
-- quarterback report by the share of the margin that is still the model's.
alter table cfb.game_projections
  add column if not exists market_history_home_margin double precision,
  add column if not exists market_history_weight double precision;

commit;
