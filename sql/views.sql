-- sql/views.sql
-- Core type-casts + convenience KPIs

DROP VIEW IF EXISTS v_interval_enriched;
CREATE VIEW v_interval_enriched AS
SELECT
  -- normalize to proper time objects for later queries
  datetime(interval_start)         AS interval_start,
  arrivals,
  avg_aht                          AS avg_aht_seconds,
  avg_acw                          AS avg_acw_seconds,
  (avg_aht + avg_acw)              AS aht_eff_seconds,
  fcr_pct,
  mix_json
FROM interval_load;

DROP VIEW IF EXISTS v_sla_policy_current;
CREATE VIEW v_sla_policy_current AS
SELECT
  kpi,
  target_minutes,
  coverage_hours,
  target_pct
FROM sla_policy
WHERE kpi = 'first_response';

-- Optional: quick join of staffing adherence per interval (headcount proxy)
DROP VIEW IF EXISTS v_staffing_per_interval;
CREATE VIEW v_staffing_per_interval AS
SELECT
  datetime(interval_start) AS interval_start,
  SUM(CASE WHEN adherent_minutes > 0 THEN 1 ELSE 0 END) AS agents_effective,
  AVG(adherent_minutes) AS avg_adherence_min
FROM agent_schedule
GROUP BY interval_start;

-- Convenience: combine interval KPIs + staffing for modeling
DROP VIEW IF EXISTS v_model_inputs;
CREATE VIEW v_model_inputs AS
SELECT
  e.interval_start,
  e.arrivals,
  e.avg_aht_seconds,
  e.avg_acw_seconds,
  e.aht_eff_seconds,
  e.fcr_pct,
  s.agents_effective
FROM v_interval_enriched e
LEFT JOIN v_staffing_per_interval s
  ON s.interval_start = e.interval_start;