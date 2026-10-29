begin;
-- 95 percent paired bootstrap interval for model_minus_market_mae: games are
-- resampled with replacement (4,000 draws, fixed seed) and the interval is the
-- 2.5th and 97.5th percentile of the mean per-game difference in absolute
-- margin error, on the games in the row that have a closing spread. Written by
-- backend/grading.py; null on closing_market rows, which have no gap.
alter table cfb.performance_metrics
    add column if not exists model_minus_market_mae_lo_95 double precision,
    add column if not exists model_minus_market_mae_hi_95 double precision;
commit;
