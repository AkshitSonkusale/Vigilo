-- Vigilo database schema
-- Run this once against your Supabase PostgreSQL instance
-- to create all required tables.

-- Users table (for future auth, not used in MVP)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255),
  email VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Raw campaign data — one row per campaign per upload session
CREATE TABLE IF NOT EXISTS campaigns (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL,           -- groups campaigns from same upload
  campaign_name VARCHAR(255) NOT NULL,
  impressions INTEGER DEFAULT 0,
  clicks INTEGER DEFAULT 0,
  cost DECIMAL(12,2) DEFAULT 0,
  conversions INTEGER DEFAULT 0,
  ctr DECIMAL(8,4) DEFAULT 0,
  cpc DECIMAL(8,2) DEFAULT 0,
  conversion_rate DECIMAL(8,4) DEFAULT 0,
  roas DECIMAL(10,4) DEFAULT 0,
  uploaded_at TIMESTAMP DEFAULT NOW()
);

-- ML results — written back after pipeline completes
CREATE TABLE IF NOT EXISTS campaign_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
  cluster_label VARCHAR(100),
  is_anomaly BOOLEAN DEFAULT FALSE,
  is_standout BOOLEAN DEFAULT FALSE,
  anomaly_score DECIMAL(8,4),
  health_score INTEGER,
  health_category VARCHAR(50),
  severity VARCHAR(20),
  processed_at TIMESTAMP DEFAULT NOW()
);

-- Recommendations — Claude API output
CREATE TABLE IF NOT EXISTS recommendations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
  recommendation_text TEXT,
  recommendation_source VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_campaigns_session ON campaigns(session_id);
CREATE INDEX IF NOT EXISTS idx_results_campaign ON campaign_results(campaign_id);
CREATE INDEX IF NOT EXISTS idx_recs_campaign ON recommendations(campaign_id);
