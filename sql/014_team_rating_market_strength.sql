begin;

alter table cfb.team_ratings
  add column if not exists market_rating double precision,
  add column if not exists market_rating_sd double precision,
  add column if not exists market_rating_games integer;

comment on column cfb.team_ratings.market_rating is
  'Neutral-field points above average FBS, fitted to prior-week closing spreads with a carried market prior. Same forecast snapshot as the model rating.';
comment on column cfb.team_ratings.market_rating_sd is
  'Conditional rating posterior SD, not game-margin or future-line prediction SD.';
comment on column cfb.team_ratings.market_rating_games is
  'Current-season earlier games with a closing spread used for this team. Zero means prior-only strength.';

commit;
