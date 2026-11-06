begin;
-- cfb.recommendation_decisions copied recommendation_eligibility_v11 from
-- cfb.recommendations when it was created, and the v12, v13 and v14 migrations
-- replaced the check on cfb.recommendations only, so a recommended v12 or later
-- decision could not be stored. Copy the current definition so the two tables
-- enforce one rule.
alter table cfb.recommendation_decisions drop constraint recommendation_eligibility_v11;
do $$
begin
  execute 'alter table cfb.recommendation_decisions add constraint recommendation_eligibility_v14 '
    || (select pg_get_constraintdef(oid) from pg_constraint
        where conrelid = 'cfb.recommendations'::regclass
          and conname = 'recommendation_eligibility_v14');
end $$;
commit;
