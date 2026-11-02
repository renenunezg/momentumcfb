-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql.
-- Attached after the first season backfill, so the backfill never revalidated
-- the site. The weekly publish replaces one season's rows with one delete and
-- one insert, so the site's cache refreshes twice per publish.
DROP TRIGGER IF EXISTS site_revalidate ON cfb.player_season_stats;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.player_season_stats FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
