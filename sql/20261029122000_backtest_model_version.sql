begin;
-- The backtest table names the model that produced each row and the game's
-- classification. `python -m backend publish-backtest` replaces the table with
-- the served joint scoring model's walk-forward (FBS vs FBS). Rows already in
-- the table come from the June 2026 state-space EPA model, whose code was
-- removed on 2026-08-01; they are tagged so the site can tell them apart until
-- that publish replaces them. In that table an FCS opponent is named 'FCS'.
alter table cfb.backtest_predictions
    add column if not exists model_version text,
    add column if not exists classification text;
update cfb.backtest_predictions
   set model_version = 'state_space_epa_2026_06',
       classification = case when home_team = 'FCS' or away_team = 'FCS'
                             then 'FBS vs FCS' else 'FBS vs FBS' end
 where model_version is null;
alter table cfb.backtest_predictions
    alter column model_version set not null,
    alter column classification set not null,
    add constraint backtest_classification check (
        classification in ('FBS vs FBS', 'FBS vs FCS', 'FCS vs FCS'));
commit;
