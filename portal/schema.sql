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
