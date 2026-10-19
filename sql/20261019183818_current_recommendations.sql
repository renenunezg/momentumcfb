begin;

-- The ledger retains original picks for settlement; this table serves the
-- latest pregame decision, including withdrawals caused by changed prices.
create table cfb.recommendation_decisions
  (like cfb.recommendations including defaults including constraints including indexes);
alter table cfb.recommendation_decisions enable row level security;
create policy current_decisions_read on cfb.recommendation_decisions
  for select to anon, authenticated using (true);
grant select on cfb.recommendation_decisions to anon, authenticated;
grant select, insert, update on cfb.recommendation_decisions to service_role;

create view cfb.current_recommendations with (security_invoker = true) as
  select * from cfb.recommendation_decisions where start_date > now()
  union all
  select * from cfb.recommendations where start_date <= now();
grant select on cfb.current_recommendations to anon, authenticated, service_role;

create trigger site_revalidate
  after insert or update on cfb.recommendation_decisions
  for each statement execute function public.site_revalidate();

notify pgrst, 'reload schema';
commit;
