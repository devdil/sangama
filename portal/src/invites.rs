//! Member-issued network invitations: a signed-in member invites a worker or client peer
//! within a monthly quota. The portal records who invited each peer, so an operator can
//! stop an inviter and revoke everyone they brought in.
use super::*;
use axum::{http::HeaderMap, response::Redirect};

/// Serializes quota checks with issuance within this portal process.
static ISSUING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
/// Invitations a member may issue in any 30 days unless `MEMBER_INVITES` says otherwise.
pub const DEFAULT_QUOTA: i64 = 3;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    role: String,
    purpose: String,
}

/// Invitations this account has issued in the last 30 days, and whether it may issue more.
async fn used(db: &Client, username: &str) -> Result<(i64, bool)> {
    let row = db
        .query_one(
            "SELECT (SELECT count(*) FROM network_invitations WHERE invited_by=a.id AND created_at>now()-interval '30 days'), a.can_invite \
             FROM accounts a WHERE username=$1",
            &[&username],
        )
        .await?;
    Ok((row.get(0), row.get(1)))
}

/// The account page section: quota, the form and recent invitations.
pub async fn section(app: &App, username: &str) -> String {
    if app.authority.is_none() || app.member_invites == 0 {
        return String::new();
    }
    let Ok((used, allowed)) = used(&app.db, username).await else {
        return "<h2>Invite a peer</h2><p>Invitations are temporarily unavailable.</p>".into();
    };
    let mut body = String::from("<h2>Invite a peer</h2>");
    if !allowed {
        body.push_str("<p>The operator has paused invitations from this account.</p>");
    } else if used >= app.member_invites {
        body.push_str(&format!(
            "<p>You have used all {} invitations for the last 30 days.</p>",
            app.member_invites
        ));
    } else {
        body.push_str(&format!(
            r#"<p>{} of {} invitations left in the last 30 days. Each code works once, for 24 hours. You are recorded as the inviter, and the operator can revoke peers you invite.</p><form action="/account/invite" method="post"><label for="role">Role</label><select id="role" name="role"><option value="worker">Worker: contributes memory and compute</option><option value="client">Client: runs inference</option></select><label for="purpose">Who is it for?</label><select id="purpose" name="purpose"><option value="own">My own device: its credits count toward my account</option><option value="other">Someone else: they keep their own balance</option></select><p><button type="submit">Create invitation</button></p></form>"#,
            app.member_invites - used,
            app.member_invites
        ));
    }
    if let Ok(rows) = app
        .db
        .query(
            "SELECT i.role, i.own_device, CASE WHEN i.used_at IS NOT NULL THEN 'Redeemed' WHEN i.expires_at<=now() THEN 'Expired' ELSE 'Waiting' END, \
             to_char(i.created_at,'YYYY-MM-DD HH24:MI') FROM network_invitations i JOIN accounts a ON a.id=i.invited_by \
             WHERE a.username=$1 ORDER BY i.created_at DESC LIMIT 10",
            &[&username],
        )
        .await
        && !rows.is_empty()
    {
        body.push_str("<div class=\"table-wrap\"><table><tr><th>Created (server time)</th><th>Role</th><th>For</th><th>Status</th></tr>");
        for r in rows {
            body.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                escape(r.get(3)),
                escape(r.get(0)),
                if r.get::<_, bool>(1) { "My device" } else { "Someone else" },
                escape(r.get(2))
            ));
        }
        body.push_str("</table></div>");
    }
    body
}

pub async fn issue(
    State(app): State<App>,
    headers: HeaderMap,
    Form(input): Form<Request>,
) -> Response {
    let username = match accounts::member(&app, &headers).await {
        Ok(Some(name)) => name,
        Ok(None) => return Redirect::to("/signin").into_response(),
        Err(response) => return response,
    };
    if app.authority.is_none() || app.member_invites == 0 {
        return failure(
            StatusCode::FORBIDDEN,
            "Member invitations are not enabled on this network.",
        );
    }
    if !["worker", "client"].contains(&input.role.as_str())
        || !["own", "other"].contains(&input.purpose.as_str())
    {
        return failure(
            StatusCode::BAD_REQUEST,
            "Choose a worker or client role and who it is for.",
        );
    }
    let _issuing = ISSUING.lock().await;
    match used(&app.db, &username).await {
        Ok((_, false)) => {
            return failure(
                StatusCode::FORBIDDEN,
                "The operator has paused invitations from this account.",
            );
        }
        Ok((used, true)) if used >= app.member_invites => {
            return failure(
                StatusCode::TOO_MANY_REQUESTS,
                "You have used all your invitations for the last 30 days.",
            );
        }
        Ok(_) => {}
        Err(_) => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Invitations are temporarily unavailable.",
            );
        }
    }
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let own = input.purpose == "own";
    let inserted = app
        .db
        .execute(
            "INSERT INTO network_invitations (token_hash,role,invited_by,own_device) SELECT $1,$2,id,$3 FROM accounts WHERE username=$4",
            &[&hash(&token), &input.role, &own, &username],
        )
        .await;
    if !matches!(inserted, Ok(1)) {
        return failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Could not create the invitation.",
        );
    }
    member_page(
        "Invitation created",
        &format!(
            "<h1>Invitation created</h1><p>{} invitation, {}. Valid once, for 24 hours. It is shown only now: copy it and send it privately, not in a public channel, URL or screenshot.</p><pre>{token}</pre><p>The invited computer saves it to a private file and runs <code>sangama mesh-join --config &lt;config&gt; --invitation-file &lt;file&gt;</code>. It also needs the network's authority public key and node configuration.</p><p><a href=\"/account\">Return to your account</a></p>",
            if input.role == "worker" { "Worker" } else { "Client" },
            if own { "for your own device" } else { "for someone else" }
        ),
    )
    .into_response()
}

/// Stops an account from inviting, revokes the peers it invited and cancels its unused
/// invitations. Returns the number of peers revoked, or `None` for an unknown account.
pub async fn stop_inviter(db: &Client, username: &str) -> Result<Option<u64>> {
    let Some(row) = db
        .query_opt(
            "UPDATE accounts SET can_invite=false WHERE username=$1 RETURNING id",
            &[&username],
        )
        .await?
    else {
        return Ok(None);
    };
    let id: i64 = row.get(0);
    db.execute(
        "UPDATE network_invitations SET expires_at=now() WHERE invited_by=$1 AND used_at IS NULL AND expires_at>now()",
        &[&id],
    )
    .await?;
    let revoked = db
        .execute(
            "UPDATE network_members SET revoked=true WHERE invited_by=$1 AND NOT revoked",
            &[&id],
        )
        .await?;
    Ok(Some(revoked))
}
