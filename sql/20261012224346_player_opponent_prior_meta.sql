begin;
-- Player value v4: the opponent effect is the ridge fit around a regressed
-- previous-season anchor; the second count-based blend and its parameter go.
alter table cfb.player_model_meta drop column if exists opponent_effect_prior_games;
alter table cfb.player_model_meta add column if not exists opponent_prior_regression double precision;
alter table cfb.player_model_meta add column if not exists opponent_prior_share double precision;
commit;
