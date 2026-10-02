begin;

-- The published forecast is the market-informed one, so its total is graded
-- beside its margin. Games graded before the blended total was published
-- keep NULL and are left out of the blended total's error.
alter table cfb.graded_games
  add column if not exists market_informed_total double precision;

commit;
