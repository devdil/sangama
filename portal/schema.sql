CREATE TABLE IF NOT EXISTS invitations (
    id BIGSERIAL PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '24 hours',
    used_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS registrations (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 80),
    peer_id TEXT NOT NULL UNIQUE CHECK (length(peer_id) BETWEEN 32 AND 128),
    platform TEXT NOT NULL CHECK (platform IN ('macOS','Linux','Windows','Other')),
    memory_gib INTEGER NOT NULL CHECK (memory_gib BETWEEN 1 AND 4096),
    invitation_id BIGINT NOT NULL UNIQUE REFERENCES invitations(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS network_invitations (
    token_hash TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('worker','client','relay')),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '24 hours',
    nonce TEXT,
    nonce_expires TIMESTAMPTZ,
    used_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS network_members (
    peer_id TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('worker','client','relay')),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '24 hours',
    revoked BOOLEAN NOT NULL DEFAULT false,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS accounts (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE CHECK (username ~ '^[a-z0-9_]{3,32}$'),
    password_hash TEXT NOT NULL,
    invitation_id BIGINT NOT NULL UNIQUE REFERENCES invitations(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS account_sessions (
    token_hash TEXT PRIMARY KEY,
    account_id BIGINT NOT NULL UNIQUE REFERENCES accounts(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '8 hours'
);
CREATE INDEX IF NOT EXISTS account_sessions_expiry ON account_sessions(expires_at);

-- Contribution credits. Each side of a session signs a receipt; only what both agree on counts.
CREATE TABLE IF NOT EXISTS credit_receipts (
    session TEXT NOT NULL,
    signer TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('work','usage')),
    consumer TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    layers INTEGER NOT NULL CHECK (layers > 0),
    tokens BIGINT NOT NULL CHECK (tokens > 0),
    stages JSONB NOT NULL,
    signed JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session, signer)
);
CREATE INDEX IF NOT EXISTS credit_receipts_signer ON credit_receipts(signer, received_at);
-- A person's worker and client peers share one balance once linked to their account.
CREATE TABLE IF NOT EXISTS credit_links (
    peer_id TEXT PRIMARY KEY,
    account_id BIGINT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE
);
-- A worker's claim counts when the consumer's usage names the same stage, model and route;
-- the lower of the two token counts is used. Units are layer-tokens.
CREATE OR REPLACE VIEW credit_entries AS
SELECT w.session, w.signer AS worker, w.consumer, w.model_hash,
       LEAST(w.tokens, u.tokens)
         * ((w.stages->0->>'end')::bigint - (w.stages->0->>'start')::bigint) AS units
FROM credit_receipts w
JOIN credit_receipts u ON u.session = w.session AND u.signer = w.consumer AND u.kind = 'usage'
WHERE w.kind = 'work' AND u.model_hash = w.model_hash AND u.layers = w.layers
  AND u.stages @> jsonb_build_array(w.stages->0);
-- Every credit a worker earns is debited from its consumer, so balances sum to zero.
CREATE OR REPLACE VIEW credit_balances AS
SELECT holder, sum(delta)::bigint AS balance FROM (
    SELECT coalesce('account:' || l.account_id, 'peer:' || e.worker) AS holder, e.units AS delta
    FROM credit_entries e LEFT JOIN credit_links l ON l.peer_id = e.worker
    UNION ALL
    SELECT coalesce('account:' || l.account_id, 'peer:' || e.consumer), -e.units
    FROM credit_entries e LEFT JOIN credit_links l ON l.peer_id = e.consumer
) t GROUP BY holder;

-- Member-issued network invitations. Operator-issued rows have no inviter.
ALTER TABLE network_invitations ADD COLUMN IF NOT EXISTS invited_by BIGINT REFERENCES accounts(id) ON DELETE SET NULL;
ALTER TABLE network_invitations ADD COLUMN IF NOT EXISTS own_device BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE network_invitations ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE INDEX IF NOT EXISTS network_invitations_inviter ON network_invitations(invited_by, created_at);
ALTER TABLE network_members ADD COLUMN IF NOT EXISTS invited_by BIGINT REFERENCES accounts(id) ON DELETE SET NULL;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS can_invite BOOLEAN NOT NULL DEFAULT true;
