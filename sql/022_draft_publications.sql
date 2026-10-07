-- One atomic edition prevents mixing pick ownership, rankings, and roster dates.
CREATE TABLE cfb.draft_publications (
    draft_year integer PRIMARY KEY CHECK (draft_year BETWEEN 2027 AND 2100),
    payload jsonb NOT NULL CHECK (payload->>'schema_version' = '1'),
    updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE cfb.draft_publications ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON cfb.draft_publications TO anon, authenticated;
CREATE POLICY draft_publications_read ON cfb.draft_publications
    FOR SELECT TO anon, authenticated USING (true);
CREATE TRIGGER draft_publications_site_revalidate
    AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON cfb.draft_publications
    FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
