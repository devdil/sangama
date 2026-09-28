//! Contribution credit ledger: accepts signed session receipts and publishes a signed
//! standing that nodes use to refuse new reservations from members past their allowance.
use super::*;
use axum::Json;
use sangama_network_auth::{
    credits::{Balance, ReceiptKind, SignedReceipt, SignedStanding, Standing},
    now,
};

/// Receipts one peer may submit per day; far above honest use.
const DAILY_RECEIPTS: i64 = 5000;
/// Layer-tokens per displayed credit.
pub const UNITS_PER_CREDIT: i64 = 1000;

pub fn format(units: i64) -> String {
    format!("{:.1}", units as f64 / UNITS_PER_CREDIT as f64)
}
/// The balance holder for a peer: its linked account, else the peer itself.
const HOLDER: &str = "coalesce('account:' || l.account_id, 'peer:' || m.peer_id)";

fn refuse(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error":message}))).into_response()
}

pub async fn receipt(State(app): State<App>, Json(input): Json<SignedReceipt>) -> Response {
    if app.authority.is_none() {
        return refuse(StatusCode::SERVICE_UNAVAILABLE, "membership not configured");
    }
    if input.verify(&app.network, now()).is_err() {
        return refuse(StatusCode::BAD_REQUEST, "invalid receipt");
    }
    let r = &input.receipt;
    let (kind, role) = match r.kind {
        ReceiptKind::Work => ("work", "worker"),
        ReceiptKind::Usage => ("usage", "client"),
    };
    let checks = app
        .db
        .query_one(
            "SELECT EXISTS (SELECT 1 FROM network_members WHERE peer_id=$1 AND role=$2 AND NOT revoked AND expires_at>now()), \
             EXISTS (SELECT 1 FROM network_members WHERE peer_id=$3 AND role='client'), \
             (SELECT count(*) FROM credit_receipts WHERE signer=$1 AND received_at>now()-interval '1 day')",
            &[&r.signer, &role, &r.consumer],
        )
        .await;
    let Ok(checks) = checks else {
        return refuse(StatusCode::SERVICE_UNAVAILABLE, "ledger unavailable");
    };
    if !checks.get::<_, bool>(0) || !checks.get::<_, bool>(1) {
        return refuse(
            StatusCode::FORBIDDEN,
            "signer or consumer is not an admitted member",
        );
    }
    if checks.get::<_, i64>(2) >= DAILY_RECEIPTS {
        return refuse(StatusCode::TOO_MANY_REQUESTS, "daily receipt limit reached");
    }
    let (Ok(stages), Ok(signed)) = (
        serde_json::to_string(&r.stages),
        serde_json::to_string(&input),
    ) else {
        return refuse(StatusCode::BAD_REQUEST, "invalid receipt");
    };
    // A session has one receipt per signer; a resubmission changes nothing.
    let inserted = app
        .db
        .execute(
            "INSERT INTO credit_receipts (session,signer,kind,consumer,model_hash,layers,tokens,stages,signed) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,($8::text)::jsonb,($9::text)::jsonb) ON CONFLICT (session,signer) DO NOTHING",
            &[
                &r.session,
                &r.signer,
                &kind,
                &r.consumer,
                &r.model_hash,
                &(r.layers as i32),
                &(r.tokens as i64),
                &stages,
                &signed,
            ],
        )
        .await;
    match inserted {
        Ok(n) => Json(serde_json::json!({"accepted":n==1})).into_response(),
        Err(_) => refuse(StatusCode::SERVICE_UNAVAILABLE, "ledger unavailable"),
    }
}

pub async fn standing(State(app): State<App>) -> Response {
    let (Some(key), Some(allowance)) = (&app.authority, app.credit_allowance) else {
        return refuse(
            StatusCode::NOT_FOUND,
            "credits are not enforced on this network",
        );
    };
    let query = format!(
        "SELECT m.peer_id, coalesce(b.balance,0) FROM network_members m \
         LEFT JOIN credit_links l ON l.peer_id=m.peer_id \
         LEFT JOIN credit_balances b ON b.holder={HOLDER} \
         WHERE NOT m.revoked AND m.expires_at>now() AND coalesce(b.balance,0)<>0 \
         ORDER BY m.peer_id LIMIT 1025"
    );
    let Ok(rows) = app.db.query(&query, &[]).await else {
        return refuse(StatusCode::SERVICE_UNAVAILABLE, "ledger unavailable");
    };
    if rows.len() > 1024 {
        return refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "standing capacity exceeded",
        );
    }
    let issued = now();
    let standing = Standing {
        network: app.network.clone(),
        issued,
        expires: issued + 30,
        allowance,
        balances: rows
            .into_iter()
            .map(|r| Balance {
                peer: r.get(0),
                balance: r.get(1),
            })
            .collect(),
    };
    match SignedStanding::sign(standing, key) {
        Ok(signed) => Json(signed).into_response(),
        Err(_) => refuse(StatusCode::SERVICE_UNAVAILABLE, "could not sign standing"),
    }
}

/// Links a peer to an account so its credits join the account's balance.
pub async fn link(db: &Client, peer: &str, username: &str) -> Result<bool> {
    let _: sangama_network_auth::libp2p_identity::PeerId = peer.parse()?;
    Ok(db
        .execute(
            "INSERT INTO credit_links (peer_id,account_id) SELECT $1,id FROM accounts WHERE username=$2 \
             ON CONFLICT (peer_id) DO UPDATE SET account_id=EXCLUDED.account_id",
            &[&peer, &username],
        )
        .await?
        == 1)
}

pub struct Summary {
    /// Holder label, balance in layer-tokens.
    pub balances: Vec<(String, i64)>,
    /// Consumers with work claims that found no matching usage after an hour.
    pub unmatched: Vec<(String, i64)>,
}

pub async fn summary(db: &Client) -> Result<Summary> {
    let balances = db
        .query(
            "SELECT coalesce(a.username || ' (account)', substr(b.holder,6)), b.balance \
             FROM credit_balances b LEFT JOIN accounts a ON b.holder='account:' || a.id \
             ORDER BY b.balance DESC LIMIT 100",
            &[],
        )
        .await?
        .into_iter()
        .map(|r| (r.get(0), r.get(1)))
        .collect();
    let unmatched = db
        .query(
            "SELECT w.consumer, count(*) FROM credit_receipts w WHERE w.kind='work' \
             AND w.received_at<now()-interval '1 hour' AND NOT EXISTS \
             (SELECT 1 FROM credit_entries e WHERE e.session=w.session AND e.worker=w.signer) \
             GROUP BY w.consumer ORDER BY count(*) DESC LIMIT 100",
            &[],
        )
        .await?
        .into_iter()
        .map(|r| (r.get(0), r.get(1)))
        .collect();
    Ok(Summary {
        balances,
        unmatched,
    })
}

/// An account's balance and how many peers contribute to it.
pub async fn account(db: &Client, username: &str) -> Result<(i64, i64)> {
    let row = db
        .query_one(
            "SELECT coalesce((SELECT balance FROM credit_balances WHERE holder='account:' || a.id),0), \
             (SELECT count(*) FROM credit_links WHERE account_id=a.id) FROM accounts a WHERE username=$1",
            &[&username],
        )
        .await?;
    Ok((row.get(0), row.get(1)))
}

#[cfg(test)]
mod tests {
    #[test]
    fn credits_display_in_thousands_of_layer_tokens() {
        assert_eq!(super::format(12_345), "12.3");
        assert_eq!(super::format(-500), "-0.5");
    }
}
