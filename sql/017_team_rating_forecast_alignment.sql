begin;

-- Published ratings are shifted so rating difference plus home field equals
-- the week's published market-informed line. The shift is stored so the
-- market-free fitted rating stays recoverable: fitted power rating is
-- power_rating minus this value, and offense and defense each carry half.
-- Rows published before this migration, and preseason rows, keep NULL.
alter table cfb.team_ratings
  add column if not exists forecast_alignment_points double precision;

commit;
