begin;

-- Explicit pregame quarterback availability. One row per season, week and
-- team names the starting quarterback's status with its source and the time
-- it was reported. Only rows reported before a forecast or decision cutoff
-- adjust that forecast (backend/model/availability.py), so frozen records
-- replay exactly. Rows are entered by hand from a cited source; nothing is
-- inferred from play-by-play text.
create table if not exists cfb.qb_availability (
  season integer not null,
  week integer not null,
  team text not null,
  player text not null,
  status text not null check (status in ('out', 'doubtful', 'questionable', 'probable')),
  source text not null,
  source_url text,
  reported_at timestamptz not null,
  notes text,
  updated_at timestamptz not null default clock_timestamp(),
  primary key (season, week, team)
);
alter table cfb.qb_availability enable row level security;
create policy public_read on cfb.qb_availability
  for select to anon, authenticated using (true);
grant select on cfb.qb_availability to anon, authenticated;
grant select, insert, update, delete on cfb.qb_availability to service_role;

-- Published projections record which side lost its starter and the net home
-- points applied, so the frozen forecast and its grading stay explainable.
alter table cfb.game_projections add column home_qb_out boolean;
alter table cfb.game_projections add column away_qb_out boolean;
alter table cfb.game_projections add column qb_availability_points double precision;

notify pgrst, 'reload schema';
commit;
