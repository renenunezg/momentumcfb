-- Shared function is owned by momentumweb/sql/001_site_revalidate.sql.
-- Football live probability tables intentionally use timed polling, not callbacks.
DROP TRIGGER IF EXISTS site_revalidate ON cfb.recommendations;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.recommendations FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.team_ratings;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.team_ratings FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.game_projections;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.game_projections FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.market_comparisons;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.market_comparisons FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.graded_games;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.graded_games FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.backtest_predictions;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.backtest_predictions FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.player_values;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.player_values FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.player_model_meta;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.player_model_meta FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.heisman_board;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.heisman_board FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.heisman_history;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.heisman_history FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.team_unit_ratings;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.team_unit_ratings FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.teams;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.teams FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
DROP TRIGGER IF EXISTS site_revalidate ON cfb.performance_metrics;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON cfb.performance_metrics FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
