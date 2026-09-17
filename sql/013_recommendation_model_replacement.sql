begin;

-- Model withdrawals are gone. A published pick that a newer model version no
-- longer makes is replaced before kickoff by the fresh decision and simply
-- stops being a pick; line movement alone never replaces one. The rows that
-- migration 011 settled void as 'model_withdrawn' (all before kickoff) are
-- kept in a backup table and returned to open no-play decisions.
create table if not exists cfb.recommendations_backup_20260917 as
  select * from cfb.recommendations where settlement_reason = 'model_withdrawn';
alter table cfb.recommendations_backup_20260917 enable row level security;

alter table cfb.recommendations disable trigger user;
update cfb.recommendations
   set status = 'no_play', reason = 'below_edge_threshold', stake_units = 0,
       outcome = 'pending', profit_units = null, graded_at = null,
       settlement_reason = null
 where settlement_reason = 'model_withdrawn';
alter table cfb.recommendations enable trigger user;

create or replace function cfb.protect_recommendation() returns trigger
language plpgsql set search_path = pg_catalog, cfb as $$
declare
  balance float8;
  expected_outcome text;
  expected_profit float8;
  settlement_fields text[] := array['outcome','home_points','away_points','profit_units',
    'graded_at','settlement_reason','closing_point','closing_price','closing_source','clv_points'];
  closing_fields text[] := array['closing_point','closing_price','closing_source','clv_points'];
begin
  if TG_OP = 'DELETE' then
    raise exception 'Recommendation history cannot be deleted';
  end if;
  if TG_OP = 'INSERT' then
    -- Caller-supplied historical timestamps must never admit a past pick.
    new.published_at := clock_timestamp();
    if new.outcome <> 'pending' then
      raise exception 'New recommendations must be pending';
    end if;
    if new.closing_source is not null then
      raise exception 'Closing lines are recorded at settlement';
    end if;
  end if;
  if TG_OP = 'UPDATE' then
    if old.outcome <> 'pending' and new is distinct from old then
      -- A settled pick may receive its closing line exactly once.
      if old.closing_source is not null or new.closing_source is null
         or (to_jsonb(new) - closing_fields) is distinct from (to_jsonb(old) - closing_fields) then
        raise exception 'Settled recommendations are immutable';
      end if;
      return new;
    end if;
    if old.status = 'recommended' or old.start_date <= clock_timestamp() then
      if (to_jsonb(new) - settlement_fields) is distinct from (to_jsonb(old) - settlement_fields) then
        -- Before kickoff a newer model version may replace its own open pick.
        if not (old.status = 'recommended' and new.outcome = 'pending'
                and old.start_date > clock_timestamp()
                and new.model_version is distinct from old.model_version) is true then
          raise exception 'Published picks and started game decisions are frozen';
        end if;
        new.published_at := clock_timestamp();
      end if;
    end if;
    if old.status = 'no_play' and old.outcome = 'pending' and new.outcome = 'pending'
       and new is distinct from old then
      new.published_at := clock_timestamp();
    end if;
  end if;
  if new.outcome = 'pending' then
    if new.home_points is not null or new.away_points is not null then
      raise exception 'Pending recommendations cannot contain final scores';
    end if;
    if new.closing_source is not null then
      raise exception 'Closing lines are recorded at settlement';
    end if;
  else
    if not (new.graded_at >= new.published_at and new.graded_at <= clock_timestamp()) then
      raise exception 'Invalid settlement timestamp';
    end if;
    if new.outcome = 'void' then
      -- A policy withdrawal settles a published pick before kickoff. Every
      -- other void keeps the schedule-change and moneyline-tie semantics.
      if new.settlement_reason = 'policy_withdrawn'
         and not (TG_OP = 'UPDATE' and old.outcome = 'pending'
                  and new.status = 'recommended'
                  and new.graded_at < new.start_date) is true then
        raise exception 'A withdrawal must settle a pending pick before kickoff';
      end if;
      expected_profit := 0;
    elsif new.outcome = 'no_play' and new.status = 'no_play' then
      if new.graded_at < new.start_date then
        raise exception 'Cannot settle before kickoff';
      end if;
      expected_profit := 0;
    elsif new.status = 'recommended' and new.outcome in ('win','loss','push') then
      if not (new.graded_at >= new.start_date and new.home_points >= 0
              and new.away_points >= 0) is true then
        raise exception 'Settlement requires a started game and final scores';
      end if;
      balance := case when new.market = 'h2h' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end
        when new.market = 'spreads' then
        (new.home_points::float8 - new.away_points) * case new.side when 'home' then 1 else -1 end + new.point
        else (new.home_points::float8 + new.away_points - new.point) * case new.side when 'over' then 1 else -1 end end;
      if new.market = 'h2h' and balance = 0 then
        raise exception 'A tied moneyline must be void';
      end if;
      expected_outcome := case when balance > 0 then 'win' when balance < 0 then 'loss' else 'push' end;
      if new.outcome <> expected_outcome then
        raise exception 'Outcome disagrees with the recorded line and final score';
      end if;
      expected_profit := case new.outcome when 'win' then
        case when new.price > 0 then new.price / 100 else 100 / abs(new.price) end
        when 'loss' then -1 else 0 end;
    else
      raise exception 'Outcome is incompatible with the recorded decision';
    end if;
    if not (abs(new.profit_units - expected_profit) < 1e-10) is true then
      raise exception 'Profit disagrees with the recorded price and outcome';
    end if;
  end if;
  return new;
end
$$;

commit;
