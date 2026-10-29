begin;
-- Every version string the evaluated weights were served under, comma
-- separated. The 2026-10-17 weights were served as cfb_position_traits_v1
-- until the 2026-10-26 publish and re-stamped cfb_position_traits_v1.1 with
-- byte-identical weights, so position_traits rows stamped v1 are the weights
-- these v1.1 rows evaluate. Written by `python -m backend publish-grade-validation`.
alter table cfb.player_grade_validation
    add column if not exists served_model_versions text;
update cfb.player_grade_validation
   set served_model_versions = 'cfb_position_traits_v1, cfb_position_traits_v1.1'
 where model_sha = '3d85cb825df4d23113b51ef05470bf285ac7ea47e13f34d7d160a6591b01d6cd'
   and served_model_versions is null;
update cfb.player_grade_validation
   set served_model_versions = grade_model_version
 where served_model_versions is null;
alter table cfb.player_grade_validation
    alter column served_model_versions set not null,
    add constraint player_grade_validation_holdout_status check (holdout_status in (
        '2025 inspected during development', 'pre-registered, untouched'));
commit;
