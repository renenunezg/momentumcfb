-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql.
-- The weekly publish replaces one season's rows in one statement, so the
-- site's hourly cache refreshes once per publish.
DROP TRIGGER IF EXISTS site_revalidate ON cfb.position_traits;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.position_traits FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
