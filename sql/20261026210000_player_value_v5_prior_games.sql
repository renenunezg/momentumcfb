begin;
-- Player value v5: the shrinkage prior is per position group, so the single
-- prior_games figure no longer describes the model and is left null.
alter table cfb.player_model_meta alter column prior_games drop not null;
alter table cfb.player_model_meta alter column prior_games drop default;
alter table cfb.player_model_meta add column if not exists prior_games_by_group text;
commit;
