//! Opt-in operator forms: no cookies, no stored browser session, exact-Origin POSTs.
use super::*;
use subtle::ConstantTimeEq;
static ATTEMPTS: std::sync::Mutex<Vec<std::time::Instant>> = std::sync::Mutex::new(Vec::new());

pub async fn form() -> Html<String> {
    page(
        "Network administration",
        r#"<h1>Network administration</h1><p>Operator access only. Enter the separate portal admin credential for each action. It is never a worker token or peer private key. Actions require HTTPS outside localhost tests.</p><form action="/admin" method="post"><label for="admin_token">Admin credential</label><input id="admin_token" type="password" name="admin_token" maxlength="64" required autocomplete="off"><label for="action">Action</label><select id="action" name="action"><option value="inspect">View memberships</option><option value="invite">Issue network invitation</option><option value="directory">Issue directory invitation</option><option value="revoke">Revoke member</option></select><label for="role">Invitation role</label><select id="role" name="role"><option>worker</option><option>client</option><option>relay</option></select><label for="peer">Peer ID to revoke</label><input id="peer" name="peer" maxlength="128" autocomplete="off"><label><input type="checkbox" name="confirm" value="yes"> Confirm revocation of this peer (required only for revocation)</label><p><button type="submit">Submit operator action</button></p></form><p>Invitation codes are shown once. Copy them privately; refreshing an invitation result may issue another invitation. Membership does not prove online status.</p>"#,
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    admin_token: String,
    action: String,
    role: String,
    peer: String,
    confirm: Option<String>,
}
fn authorized(expected: Option<&str>, supplied: &str) -> bool {
    let Some(expected) = expected else {
        return false;
    };
    // Compare fixed-length hashes in constant time, even for a wrong-length input.
    bool::from(Sha256::digest(expected.as_bytes()).ct_eq(&Sha256::digest(supplied.as_bytes())))
}
pub async fn submit(State(app): State<App>, Form(input): Form<Action>) -> Response {
    {
        let mut attempts = ATTEMPTS.lock().unwrap();
        attempts.retain(|t| t.elapsed().as_secs() < 60);
        if attempts.len() >= 20 {
            return failure(
                StatusCode::TOO_MANY_REQUESTS,
                "Operator request limit reached. Try again in one minute.",
            );
        }
        attempts.push(std::time::Instant::now());
    }
    if !authorized(app.admin_token.as_deref(), &input.admin_token) {
        return failure(
            StatusCode::FORBIDDEN,
            "Operator access denied or not configured.",
        );
    }
    if input.action == "invite" || input.action == "directory" {
        if !["worker", "client", "relay"].contains(&input.role.as_str())
            || (input.action == "invite" && app.authority.is_none())
        {
            return failure(
                StatusCode::BAD_REQUEST,
                "Invalid role or membership authority unavailable.",
            );
        }
        let token = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let digest = hash(&token);
        let result = if input.action == "invite" {
            app.db
                .execute(
                    "INSERT INTO network_invitations (token_hash,role) VALUES ($1,$2)",
                    &[&digest, &input.role],
                )
                .await
        } else {
            app.db
                .execute(
                    "INSERT INTO invitations (token_hash) VALUES ($1)",
                    &[&digest],
                )
                .await
        };
        if result.is_err() {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Could not issue invitation.",
            );
        }
        return page("Invitation issued", &format!("<h1>Invitation issued</h1><p>Type: {}. Valid once, for 24 hours. Deliver privately and do not include it in URLs or logs.</p><pre>{}</pre><p><a href=\"/admin\">Return to administration</a></p>", if input.action == "invite" {format!("network / {}",escape(&input.role))} else {"directory registration".into()}, token)).into_response();
    }
    if input.action == "revoke" {
        if input.confirm.as_deref() != Some("yes")
            || input.peer.len() > 128
            || input
                .peer
                .parse::<sangama_network_auth::libp2p_identity::PeerId>()
                .is_err()
        {
            return failure(
                StatusCode::BAD_REQUEST,
                "Enter a valid peer ID and confirm revocation.",
            );
        }
        return match app.db.execute("UPDATE network_members SET revoked=true WHERE peer_id=$1", &[&input.peer]).await {
            Ok(1) => page("Member revoked", "<h1>Membership revoked</h1><p>Workers enforce revocation as signed snapshots refresh or expire. An active request may be interrupted. This does not delete the directory entry.</p><a href=\"/admin\">Return to administration</a>").into_response(),
            Ok(_) => failure(StatusCode::NOT_FOUND,"Member not found."),
            Err(_) => failure(StatusCode::SERVICE_UNAVAILABLE,"Could not revoke member."),
        };
    }
    if input.action != "inspect" {
        return failure(StatusCode::BAD_REQUEST, "Unknown action.");
    }
    let Ok(rows) = app.db.query("SELECT peer_id,role,expires_at::text,CASE WHEN revoked THEN 'Revoked' WHEN expires_at<=now() THEN 'Expired' ELSE 'Admitted' END FROM network_members ORDER BY expires_at DESC LIMIT 100", &[]).await else { return failure(StatusCode::SERVICE_UNAVAILABLE,"Membership unavailable."); };
    let mut body = String::from(
        "<h1>Memberships</h1><p>Latest 100 memberships. Availability, model readiness and connection path are checked locally, not inferred from this list.</p><div class=\"table-wrap\"><table><tr><th>Peer</th><th>Role</th><th>Expires (server time)</th><th>Admission</th></tr>",
    );
    for row in rows {
        body.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape(row.get(0)),
            escape(row.get(1)),
            escape(row.get(2)),
            escape(row.get(3))
        ));
    }
    body.push_str("</table></div><p><a href=\"/admin\">Return to administration</a></p>");
    page("Memberships", &body).into_response()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operator_credential_required() {
        assert!(!authorized(None, ""));
        assert!(!authorized(Some("secret"), "wrong"));
        assert!(authorized(Some("secret"), "secret"));
    }
}
