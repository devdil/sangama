# Portal accounts

Open `/join` to sign up with an **invite code, username and password**. Open `/signin`
to sign in. The account page links to worker setup and provides a Sign out button.
No device specifications, peer ID, email address or publication consent are requested at signup.
Existing device directory rows are preserved. New accounts are not public directory entries.

Operators issue signup codes through `/admin` → **Issue signup invitation**, or the existing
`sangama-portal invite` command. Codes expire after 24 hours and work once. Network membership
invitations remain separate: creating an account never authorizes a worker or grants admin access.

Usernames are case-insensitive, normalized to lowercase, and contain 3–32 ASCII letters, numbers
or underscores. Passwords require 15–128 characters (maximum 512 UTF-8 bytes). There is no password
recovery or password-change UI yet; do not promise these flows to testers.

PostgreSQL stores accounts and sessions. Passwords use independently salted Argon2id hashes
(19 MiB, two iterations, one lane). Session tokens are random; only SHA-256 token hashes are stored.
Sessions expire after eight hours, rotate on sign-in, and are revoked by sign-out. Each account has
one active session. HTTPS uses a `__Host-` cookie with Secure, HttpOnly, SameSite=Strict and Path=/.
The localhost test stack uses a separate non-Secure cookie name. All browser POSTs require the
configured exact Origin, including signup, signin and signout. Responses disable caching.

Authentication has a conservative process-wide limit of 20 attempts/minute shared by signup and
signin, with at most two password hashing operations at a time. This is suitable for the invited
preview, but can cause other users to be throttled. A multi-instance public service needs a shared
rate limiter and operational abuse monitoring. Keep the existing private PostgreSQL network and
HTTPS proxy configuration in production.

The startup schema adds `accounts` and `account_sessions` without deleting legacy registrations.
Run `python3 scripts/test-portal.py` against the local Docker test stack for invitation, password,
session, CSRF and throttle checks. Tests create temporary accounts and remove them afterward.

References: [OWASP password storage](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
and [session management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html).
