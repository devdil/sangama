use super::*;
use axum::Json;
use sangama_network_auth::{Join, Member, SignedSnapshot, Snapshot, now};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Challenge {
    invitation: String,
}
fn denied() -> Response {
    (StatusCode::FORBIDDEN, "Membership denied").into_response()
}
pub async fn challenge(State(app): State<App>, Json(input): Json<Challenge>) -> Response {
    if app.authority.is_none() || input.invitation.len() != 64 {
        return denied();
    }
    let nonce = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    // Invitation holders can rotate a challenge; only the most recent is usable.
    let rows = app.db.execute("UPDATE network_invitations SET nonce=$2, nonce_expires=now()+interval '60 seconds' WHERE token_hash=$1 AND used_at IS NULL AND expires_at>now()", &[&hash(&input.invitation), &nonce]).await;
    match rows {
        Ok(1) => Json(serde_json::json!({"nonce":nonce,"network":app.network})).into_response(),
        _ => denied(),
    }
}
pub async fn redeem(State(app): State<App>, Json(input): Json<Join>) -> Response {
    if app.authority.is_none() {
        return denied();
    }
    let Ok(peer) = input.peer(&app.network) else {
        return denied();
    };
    // A single statement atomically consumes both nonce and invitation, and creates membership.
    // It records the inviting account; a member's invitation cannot restore a revoked peer, and
    // one for the member's own device joins that account's credit balance.
    let result = app.db.query("WITH claimed AS (UPDATE network_invitations SET used_at=now(),nonce=NULL WHERE token_hash=$1 AND nonce=$2 AND nonce_expires>now() AND expires_at>now() AND used_at IS NULL RETURNING role,invited_by,own_device), \
        member AS (INSERT INTO network_members (peer_id,role,invited_by) SELECT $3,role,invited_by FROM claimed ON CONFLICT (peer_id) DO UPDATE SET role=EXCLUDED.role,expires_at=now()+interval '24 hours',revoked=false,invited_by=EXCLUDED.invited_by WHERE EXCLUDED.invited_by IS NULL OR NOT network_members.revoked RETURNING peer_id), \
        linked AS (INSERT INTO credit_links (peer_id,account_id) SELECT m.peer_id,c.invited_by FROM member m, claimed c WHERE c.own_device AND c.invited_by IS NOT NULL ON CONFLICT (peer_id) DO UPDATE SET account_id=EXCLUDED.account_id) \
        SELECT peer_id FROM member", &[&hash(&input.invitation), &input.nonce, &peer]).await;
    match result {
        Ok(rows) if rows.len() == 1 => {
            Json(serde_json::json!({"peer":peer,"network":app.network})).into_response()
        }
        _ => denied(),
    }
}
pub async fn snapshot(State(app): State<App>) -> Response {
    let Some(key) = &app.authority else {
        return (StatusCode::SERVICE_UNAVAILABLE, "Membership not configured").into_response();
    };
    let Ok(rows) = app.db.query("SELECT peer_id,role,extract(epoch from expires_at)::bigint FROM network_members WHERE NOT revoked AND expires_at>now() ORDER BY peer_id LIMIT 1025", &[]).await else { return denied(); };
    if rows.len() > 1024 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Membership capacity exceeded",
        )
            .into_response();
    }
    let issued = now();
    let members = rows
        .into_iter()
        .map(|r| Member {
            peer: r.get(0),
            role: r.get(1),
            expires: r.get::<_, i64>(2) as u64,
        })
        .collect();
    match SignedSnapshot::sign(
        Snapshot {
            network: app.network,
            issued,
            expires: issued + 15,
            members,
        },
        key,
    ) {
        Ok(signed) => Json(signed).into_response(),
        Err(_) => denied(),
    }
}
