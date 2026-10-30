begin;

-- One captured closing line per game, the shared contract across the four
-- sport schemas. The kickoff capture writes each game once from the validated
-- final pregame snapshot: the provider median home spread and total, the
-- prices of one named bookmaker at those medians (price_book), the snapshot
-- fetch time (fetched_at) and that bookmaker's last update (observed_at).
-- home_spread is the home line, negative when home is favored. Prices are
-- American. Moneylines stay null unless the unmetered ESPN fallback carried
-- them; no Odds API market is bought for this table.
--
-- Grading and pick CLV read this table first and fall back to the CFBD
-- median, labelled closing_source = 'cfbd_lines_median'; the two populations
-- are never pooled. Raw snapshots stay out of the database.
create table if not exists cfb.closing_lines (
  game_id bigint primary key,
  season integer not null,
  start_date timestamptz not null,
  source text not null,
  observed_at timestamptz,
  fetched_at timestamptz not null,
  home_spread double precision,
  spread_home_price double precision,
  spread_away_price double precision,
  total double precision,
  over_price double precision,
  under_price double precision,
  home_moneyline double precision,
  away_moneyline double precision,
  n_spread_offers integer,
  n_total_offers integer,
  price_book text,
  constraint closing_line_source check (source <> ''),
  constraint closing_line_priced check (
    (home_spread is not null or total is not null)
    and (price_book is not null or (spread_home_price is null
      and spread_away_price is null and over_price is null and under_price is null)))
);

-- A closing line is captured before kickoff and written once, soon after it.
-- The capture publishes after its window closes, so the insert deadline is a
-- day after the start rather than minutes; the row's own fetched_at must
-- still precede the start it is the close for.
create or replace function cfb.protect_closing_line() returns trigger
language plpgsql set search_path = pg_catalog, cfb as $$
begin
  if TG_OP <> 'INSERT' then
    raise exception 'Closing lines are frozen records';
  end if;
  if new.fetched_at >= new.start_date then
    raise exception 'A closing line must be fetched before the game starts';
  end if;
  if now() > new.start_date + interval '24 hours' then
    raise exception 'A closing line must be recorded within a day of the start';
  end if;
  return new;
end
$$;
drop trigger if exists protect_closing_line on cfb.closing_lines;
create trigger protect_closing_line
    before insert or update or delete on cfb.closing_lines
    for each row execute function cfb.protect_closing_line();
drop trigger if exists protect_closing_line_truncate on cfb.closing_lines;
create trigger protect_closing_line_truncate
    before truncate on cfb.closing_lines
    for each statement execute function cfb.protect_closing_line();

alter table cfb.closing_lines enable row level security;
drop policy if exists public_read on cfb.closing_lines;
create policy public_read on cfb.closing_lines
    for select to anon, authenticated using (true);
grant select on cfb.closing_lines to anon, authenticated, service_role;
grant insert on cfb.closing_lines to service_role;
-- The site reads this table through its timed cache; no site_revalidate here.

-- The performance page shows which closing population each row was graded
-- against: a jsonb object of graded games per closing_source, for the games
-- in the row that have a closing spread.
alter table cfb.performance_metrics
    add column if not exists closing_source_games jsonb;

-- A spread or total pick may now record the captured closing price beside the
-- closing point; the CFBD fallback has no price and leaves it null.
alter table cfb.recommendations
    drop constraint if exists recommendation_closing_line_value;
alter table cfb.recommendations add constraint recommendation_closing_line_value
check ((closing_source is null and closing_point is null
        and closing_price is null and clv_points is null) or
  (closing_source is not null and status = 'recommended'
   and outcome in ('win', 'loss', 'push') and
   ((market = 'h2h' and closing_price is not null
     and closing_point is null and clv_points is null) or
    (market = 'spreads' and closing_point is not null
     and abs(clv_points - (point - closing_point)) < 1e-9) or
    (market = 'totals' and closing_point is not null
     and abs(clv_points - case side when 'over' then closing_point - point
                                    else point - closing_point end) < 1e-9))) is true);

notify pgrst, 'reload schema';
commit;
