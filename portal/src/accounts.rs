//! Invite-only accounts; portal sessions never authorize inference membership.
use super::*;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, password_hash::SaltString};
use axum::{http::HeaderMap, response::Redirect};
static ATTEMPTS: std::sync::Mutex<Vec<std::time::Instant>> = std::sync::Mutex::new(Vec::new());
static HASHING: Semaphore = Semaphore::const_new(2);

pub async fn signup_form() -> Html<String> {
    page(
        "Sign up",
        r#"<h1>Sign up</h1><p>Join Sangama with an invitation.</p><form action="/join" method="post"><label for="invitation">Invite code</label><input id="invitation" name="invitation" type="password" maxlength="64" required autocomplete="off"><label for="username">Username</label><input id="username" name="username" minlength="3" maxlength="32" pattern="[A-Za-z0-9_]{3,32}" required autocomplete="username"><small>3–32 letters, numbers or underscores.</small><label for="password">Password</label><input id="password" name="password" type="password" minlength="15" maxlength="128" required autocomplete="new-password"><small>Use at least 15 characters.</small><p><button type="submit">Sign up</button> <a class="button" href="/signin">Sign in</a></p></form>"#,
    )
}
pub async fn signin_form() -> Html<String> {
    page(
        "Sign in",
        r#"<h1>Sign in</h1><form action="/signin" method="post"><label for="username">Username</label><input id="username" name="username" maxlength="32" required autocomplete="username"><label for="password">Password</label><input id="password" name="password" type="password" maxlength="128" required autocomplete="current-password"><p><button type="submit">Sign in</button> <a href="/join">Sign up with an invite code</a></p></form>"#,
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signup {
    invitation: String,
    username: String,
    password: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signin {
    username: String,
    password: String,
}
fn username(raw: &str) -> Option<String> {
    let value = raw.trim().to_ascii_lowercase();
    ((3..=32).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'))
    .then_some(value)
}
fn limited() -> bool {
    let mut attempts = ATTEMPTS.lock().unwrap();
    attempts.retain(|t| t.elapsed().as_secs() < 60);
    if attempts.len() >= 20 {
        return true;
    }
    attempts.push(std::time::Instant::now());
    false
}
fn busy() -> Response {
    failure(
        StatusCode::TOO_MANY_REQUESTS,
        "Too many attempts. Try again in one minute.",
    )
}
fn unavailable() -> Response {
    failure(
        StatusCode::SERVICE_UNAVAILABLE,
        "Account service unavailable. Please try again.",
    )
}
fn random() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}
fn password_hash(password: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::encode_b64(uuid::Uuid::new_v4().as_bytes())?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|p| p.to_string())
}
pub async fn signup(State(app): State<App>, Form(form): Form<Signup>) -> Response {
    if limited() {
        return busy();
    }
    let Some(name) = username(&form.username) else {
        return failure(
            StatusCode::BAD_REQUEST,
            "Use 3–32 letters, numbers or underscores for your username.",
        );
    };
    if !(15..=128).contains(&form.password.chars().count())
        || form.password.len() > 512
        || form.invitation.len() != 64
        || !form.invitation.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return failure(
            StatusCode::BAD_REQUEST,
            "Check your invite code and use a password of 15–128 characters.",
        );
    }
    let Ok(permit) = HASHING.try_acquire() else {
        return busy();
    };
    let hashed = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        password_hash(&form.password)
    })
    .await;
    let Ok(Ok(hashed)) = hashed else {
        return unavailable();
    };
    let result = app.db.query("WITH claimed AS (UPDATE invitations SET used_at=now() WHERE token_hash=$1 AND used_at IS NULL AND expires_at>now() RETURNING id) INSERT INTO accounts (username,password_hash,invitation_id) SELECT $2,$3,id FROM claimed RETURNING id", &[&hash(&form.invitation), &name, &hashed]).await;
    match result {
        Ok(rows) if !rows.is_empty() => (StatusCode::CREATED,page("Account created", "<h1>Your account is ready.</h1><p><a class=\"button\" href=\"/signin\">Sign in</a></p>")).into_response(),
        Ok(_) => failure(StatusCode::FORBIDDEN,"Invite code is invalid, expired or already used."),
        Err(e) if e.code() == Some(&tokio_postgres::error::SqlState::UNIQUE_VIOLATION) => failure(StatusCode::CONFLICT,"Unable to create account. Try a different username."),
        Err(_) => unavailable(),
    }
}
fn cookie_name(app: &App) -> &'static str {
    if app.origin.starts_with("https://") {
        "__Host-sangama_session"
    } else {
        "sangama_session"
    }
}
fn cookie(app: &App, token: &str, age: u32) -> String {
    format!(
        "{}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={age}{}",
        cookie_name(app),
        if app.origin.starts_with("https://") {
            "; Secure"
        } else {
            ""
        }
    )
}
fn token(app: &App, headers: &HeaderMap) -> Option<String> {
    headers
        .get("cookie")?
        .to_str()
        .ok()?
        .split(';')
        .find_map(|part| {
            let (name, value) = part.trim().split_once('=')?;
            (name == cookie_name(app)
                && value.len() == 64
                && value.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| value.to_owned())
        })
}
pub async fn signin(State(app): State<App>, Form(form): Form<Signin>) -> Response {
    if limited() {
        return busy();
    }
    let name = username(&form.username).unwrap_or_default();
    if form.password.len() > 512 {
        return failure(StatusCode::UNAUTHORIZED, "Invalid username or password.");
    }
    let row = match app
        .db
        .query_opt(
            "SELECT id,password_hash FROM accounts WHERE username=$1",
            &[&name],
        )
        .await
    {
        Ok(row) => row,
        Err(_) => return unavailable(),
    };
    let stored = row.as_ref().map(|r| r.get::<_, String>(1));
    let Ok(permit) = HASHING.try_acquire() else {
        return busy();
    };
    let verified = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        match stored {
            Some(stored) => PasswordHash::new(&stored).is_ok_and(|p| {
                Argon2::default()
                    .verify_password(form.password.as_bytes(), &p)
                    .is_ok()
            }),
            None => {
                let _ = password_hash(&form.password);
                false
            }
        }
    })
    .await
    .unwrap_or(false);
    if !verified {
        return failure(StatusCode::UNAUTHORIZED, "Invalid username or password.");
    }
    let id: i64 = row.unwrap().get(0);
    let session = random();
    // One session per account; successful sign-in invalidates older sessions.
    if app.db.execute("INSERT INTO account_sessions (account_id,token_hash) VALUES ($1,$2) ON CONFLICT (account_id) DO UPDATE SET token_hash=EXCLUDED.token_hash, expires_at=now()+interval '8 hours'", &[&id,&hash(&session)]).await.is_err() { return unavailable(); }
    let mut response = Redirect::to("/account").into_response();
    response
        .headers_mut()
        .insert("set-cookie", cookie(&app, &session, 28800).parse().unwrap());
    response
}
/// The signed-in username, if the request carries a live session.
pub async fn member(app: &App, headers: &HeaderMap) -> Result<Option<String>, Response> {
    let Some(session) = token(app, headers) else {
        return Ok(None);
    };
    match app.db.query_opt("SELECT a.username FROM accounts a JOIN account_sessions s ON s.account_id=a.id WHERE s.token_hash=$1 AND s.expires_at>now()", &[&hash(&session)]).await {
        Ok(row) => Ok(row.map(|r| r.get(0))),
        Err(_) => Err(unavailable()),
    }
}
/// Member-only pages send signed-out visitors to sign in.
pub async fn members_only(app: &App, headers: &HeaderMap, title: &str, body: &str) -> Response {
    match member(app, headers).await {
        Ok(Some(_)) => member_page(title, body).into_response(),
        Ok(None) => Redirect::to("/signin").into_response(),
        Err(response) => response,
    }
}
pub async fn account(State(app): State<App>, headers: HeaderMap) -> Response {
    match member(&app, &headers).await {
        Ok(Some(name)) => member_page("Your account", &format!("<h1>Welcome, {}.</h1><p>You are signed in.</p><p><a class=\"button\" href=\"/\">Network directory</a> <a class=\"button\" href=\"/connect\">Connect a worker</a></p><p>Your account does not automatically admit a worker to the network.</p><form action=\"/signout\" method=\"post\"><button type=\"submit\">Sign out</button></form>",escape(&name))).into_response(),
        Ok(None) => Redirect::to("/signin").into_response(),
        Err(response) => response,
    }
}
pub async fn signout(State(app): State<App>, headers: HeaderMap) -> Response {
    if let Some(session) = token(&app, &headers)
        && app
            .db
            .execute(
                "DELETE FROM account_sessions WHERE token_hash=$1",
                &[&hash(&session)],
            )
            .await
            .is_err()
    {
        return unavailable();
    }
    let mut response = Redirect::to("/signin").into_response();
    response
        .headers_mut()
        .insert("set-cookie", cookie(&app, "", 0).parse().unwrap());
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_and_bounds_usernames() {
        assert_eq!(username(" Alice_123 ").as_deref(), Some("alice_123"));
        for value in ["ab", "<script>", "a b", "éclair", &"a".repeat(33)] {
            assert!(username(value).is_none());
        }
    }
    #[test]
    fn password_hashes_are_salted_and_verifiable() {
        let first = password_hash("a long test passphrase").unwrap();
        let second = password_hash("a long test passphrase").unwrap();
        assert_ne!(first, second);
        assert!(first.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"));
        let parsed = PasswordHash::new(&first).unwrap();
        assert!(
            Argon2::default()
                .verify_password(b"a long test passphrase", &parsed)
                .is_ok()
        );
        assert!(
            Argon2::default()
                .verify_password(b"wrong", &parsed)
                .is_err()
        );
    }
}
