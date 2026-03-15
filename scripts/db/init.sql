-- NeuroFusion-AD PostgreSQL initialization script
-- Creates audit log table for prediction tracking
--
-- IEC 62304 traceability: SRS-001 § 7.3, SAD-001 § 5.3
-- Audit requirements: all predictions logged for post-market surveillance

-- Ensure idempotent re-runs
CREATE TABLE IF NOT EXISTS prediction_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID        NOT NULL UNIQUE,
    patient_id_hash VARCHAR(64) NOT NULL,          -- SHA-256 hex of patient identifier
    source_system   VARCHAR(256),                  -- requesting EHR system
    probability     DOUBLE PRECISION,              -- amyloid positivity probability 0–1
    latency_ms      DOUBLE PRECISION,              -- end-to-end request latency
    model_version   VARCHAR(64) NOT NULL DEFAULT 'phase2b-v1.0',
    error           TEXT,                          -- null on success
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for patient-level audit queries
CREATE INDEX IF NOT EXISTS idx_audit_patient
    ON prediction_audit_log (patient_id_hash);

-- Index for time-range queries (post-market surveillance)
CREATE INDEX IF NOT EXISTS idx_audit_created_at
    ON prediction_audit_log (created_at DESC);

-- Index for error analysis
CREATE INDEX IF NOT EXISTS idx_audit_error
    ON prediction_audit_log (error)
    WHERE error IS NOT NULL;

-- Model performance tracking (populated by evaluate.py)
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    version_id      VARCHAR(64) NOT NULL UNIQUE,   -- e.g. phase2b-v1.0
    adni_auc        DOUBLE PRECISION,
    bh_auc          DOUBLE PRECISION,
    ece_after       DOUBLE PRECISION,
    temperature     DOUBLE PRECISION,
    embed_dim       INT,
    num_heads       INT,
    dropout         DOUBLE PRECISION,
    training_date   DATE,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed the Phase 2B model version
INSERT INTO model_versions (
    version_id, adni_auc, bh_auc, ece_after, temperature,
    embed_dim, num_heads, dropout, training_date, notes
) VALUES (
    'phase2b-v1.0', 0.8897, 0.9071, 0.0831, 0.756,
    256, 4, 0.4, '2026-03-10',
    'Phase 2B remediated model. ABETA42_CSF removed (leakage fix). fluid_input_dim=2.'
) ON CONFLICT (version_id) DO NOTHING;

-- Grant minimal privileges (app role)
-- Run as superuser; app connects as neurofusion user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'neurofusion') THEN
        CREATE ROLE neurofusion WITH LOGIN PASSWORD 'neurofusion';
    END IF;
END
$$;

GRANT SELECT, INSERT ON prediction_audit_log TO neurofusion;
GRANT SELECT ON model_versions TO neurofusion;
GRANT USAGE, SELECT ON SEQUENCE prediction_audit_log_id_seq TO neurofusion;
