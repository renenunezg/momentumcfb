-- One compact current snapshot per season; detailed simulation artifacts stay offline.
begin;
create table cfb.playoff_forecasts (
  season integer primary key check (season between 2026 and 2100),
  published_at timestamptz not null default now(),
  payload jsonb not null check (
    jsonb_typeof(payload) = 'object'
    and payload ?& array['schema_version','season','teams','bracket']
    and payload->>'schema_version' = '2'
    and (payload->>'season')::integer = season
    and jsonb_typeof(payload->'teams') = 'array'
    and jsonb_array_length(payload->'teams') between 12 and 200
    and jsonb_typeof(payload->'bracket') = 'array'
    and jsonb_array_length(payload->'bracket') = 12
  )
);
alter table cfb.playoff_forecasts enable row level security;
grant usage on schema cfb to anon, authenticated, service_role;
revoke all on cfb.playoff_forecasts from anon, authenticated;
grant select on cfb.playoff_forecasts to anon, authenticated;
grant all on cfb.playoff_forecasts to service_role;
create policy public_read on cfb.playoff_forecasts
  for select to anon, authenticated using (true);
create trigger site_revalidate
  after insert or update or delete or truncate on cfb.playoff_forecasts
  for each statement execute function public.site_revalidate();
commit;
