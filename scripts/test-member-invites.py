#!/usr/bin/env python3
"""Exercise member-issued network invitations against a throwaway local portal.

Starts PostgreSQL (TCP only, private port) and the portal, creates two member
accounts through the real signup and sign-in forms, then checks that members can
invite worker and client peers within their quota, that redemption records the
inviter and links own-device peers to the inviter's credit balance, that an
operator can stop an inviter and revoke its peers, and that a member's invitation
cannot restore a revoked peer. No inference runs. Requires the release binaries.
"""
import http.cookiejar, json, os, re, secrets, shutil, socket, subprocess, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'target/release/sangama'
PORTAL_BIN = ROOT / 'portal/target/release/sangama-portal'
NETWORK = 'invites-test'
QUOTA = 2


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait(check, seconds, what):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except Exception as e:  # noqa: BLE001 - report the final failure below
            last = e
        time.sleep(.5)
    raise AssertionError(f'Timed out waiting for {what}: {last}')


def main():
    work = ROOT / 'runs' / f'invites-{uuid.uuid4().hex[:8]}'
    work.mkdir(parents=True)
    pg, pg_port, portal_port = work / 'pg', free_port(), free_port()
    portal = None
    checks = {}
    try:
        subprocess.run(['initdb', '-D', str(pg), '-U', 'postgres', '-A', 'trust'], check=True, capture_output=True)
        subprocess.run(['pg_ctl', '-D', str(pg), '-o', f"-p {pg_port} -k '' -c listen_addresses=127.0.0.1",
                        '-l', str(work / 'pg.log'), 'start'], check=True, capture_output=True)

        def psql(query, db='sangama'):
            return subprocess.run(['psql', '-h', '127.0.0.1', '-p', str(pg_port), '-U', 'postgres', '-d', db,
                                   '-v', 'ON_ERROR_STOP=1', '-Atc', query],
                                  check=True, capture_output=True, text=True).stdout.strip()
        wait(lambda: psql('SELECT 1', 'postgres') == '1', 30, 'PostgreSQL')
        dbpass = secrets.token_hex(16)
        psql(f"CREATE USER sangama PASSWORD '{dbpass}'", 'postgres')
        psql('CREATE DATABASE sangama OWNER sangama', 'postgres')
        (work / 'dbpass').write_text(dbpass)
        authority = work / 'authority'
        subprocess.run([str(BIN), 'mesh-identity', '--state-dir', str(authority)], check=True, capture_output=True)
        origin = f'http://127.0.0.1:{portal_port}'
        env = os.environ | {
            'DATABASE_PASSWORD_FILE': str(work / 'dbpass'), 'DATABASE_HOST': '127.0.0.1',
            'DATABASE_PORT': str(pg_port), 'PUBLIC_ORIGIN': origin, 'LISTEN_ADDR': f'127.0.0.1:{portal_port}',
            'NETWORK_ID': NETWORK, 'MEMBERSHIP_KEY_FILE': str(authority / 'identity.key'),
            'MEMBER_INVITES': str(QUOTA),
        }

        def cli(*args):
            return subprocess.run([str(PORTAL_BIN), *args], env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()
        portal = subprocess.Popen([str(PORTAL_BIN)], env=env, stdout=open(work / 'portal.log', 'w'),
                                  stderr=subprocess.STDOUT, start_new_session=True)

        def healthy():
            with urllib.request.urlopen(origin + '/healthz', timeout=2) as r:
                return r.status == 200
        wait(healthy, 30, 'portal')

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                return None

        def browser():
            jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar), NoRedirect)

            def request(path, fields=None, origin_header=origin):
                data = None if fields is None else urllib.parse.urlencode(fields).encode()
                req = urllib.request.Request(origin + path, data=data, headers={'Origin': origin_header})
                try:
                    r = opener.open(req, timeout=10)
                except urllib.error.HTTPError as e:
                    r = e
                with r:
                    return r.status, r.read().decode()
            return request

        def member(name):
            b = browser()
            password = 'a test passphrase ' + secrets.token_hex(8)
            assert b('/join', {'invitation': cli('invite'), 'username': name, 'password': password})[0] == 201
            assert b('/signin', {'username': name, 'password': password})[0] == 303
            return b
        alice, bob = member('alice'), member('bob')
        assert 'Invite a peer' in alice('/account')[1] and f'{QUOTA} of {QUOTA} invitations left' in alice('/account')[1]

        def invite(who, role, purpose, expect=200):
            status, body = who('/account/invite', {'role': role, 'purpose': purpose})
            assert status == expect, (status, body)
            if status == 200:
                return re.search(r'<pre>([0-9a-f]{64})</pre>', body).group(1)

        def peer(name):
            state = work / name
            peer_id = subprocess.run([str(BIN), 'mesh-identity', '--state-dir', str(state)],
                                     check=True, capture_output=True, text=True).stdout.strip()
            (state / 'token').write_text(secrets.token_hex(32))
            (state / 'config.json').write_text(json.dumps({
                'state_dir': str(state), 'authority_file': str(authority / 'public.key'), 'network': NETWORK,
                'portal': origin, 'test_http': True, 'listen': '/ip4/127.0.0.1/tcp/0',
                'token_file': str(state / 'token')}))
            return peer_id

        def join(name, code):
            state = work / name
            (state / 'invite').write_text(code)
            (state / 'invite').chmod(0o600)
            return subprocess.run([str(BIN), 'mesh-join', '--config', str(state / 'config.json'),
                                   '--invitation-file', str(state / 'invite')], capture_output=True, text=True)

        # Cross-site form posts and relay or unknown roles are refused.
        assert alice('/account/invite', {'role': 'worker', 'purpose': 'own'}, origin_header='https://evil.example')[0] == 403
        invite(alice, 'relay', 'own', 400)
        checks['cross_site_and_privileged_roles_refused'] = True

        laptop, friend = peer('alice-laptop'), peer('bob-friend')
        assert join('alice-laptop', invite(alice, 'worker', 'own')).returncode == 0
        assert join('bob-friend', invite(alice, 'client', 'other')).returncode == 0
        rows = psql("SELECT m.peer_id||'|'||m.role||'|'||a.username FROM network_members m JOIN accounts a ON a.id=m.invited_by ORDER BY m.role DESC")
        assert rows.splitlines() == [f'{laptop}|worker|alice', f'{friend}|client|alice'], rows
        links = psql("SELECT l.peer_id||'|'||a.username FROM credit_links l JOIN accounts a ON a.id=l.account_id")
        assert links == f'{laptop}|alice', links
        checks['members_invite_within_quota_and_inviter_recorded'] = True
        checks['own_device_joins_inviter_credit_balance'] = True

        # The quota counts issued invitations, redeemed or not.
        invite(alice, 'worker', 'own', 429)
        assert 'used all 2 invitations' in alice('/account')[1]
        spare = invite(bob, 'worker', 'other')
        checks['quota_enforced_per_account'] = True

        # Stopping an inviter revokes its peers, cancels unused codes and blocks new ones.
        assert 'revoked 2 invited peer' in cli('stop-inviter', 'alice')
        assert psql("SELECT count(*) FROM network_members WHERE revoked") == '2'
        assert 'paused invitations' in alice('/account')[1]
        cli('stop-inviter', 'bob')
        assert join('bob-friend', spare).returncode != 0
        checks['operator_stops_inviter_and_revokes_its_peers'] = True

        # Another member cannot readmit a revoked peer; the operator still can.
        cli('resume-inviter', 'bob')
        assert join('bob-friend', invite(bob, 'client', 'other')).returncode != 0
        assert psql(f"SELECT revoked FROM network_members WHERE peer_id='{friend}'") == 't'
        assert join('bob-friend', cli('network-invite', 'client')).returncode == 0
        assert psql(f"SELECT revoked::text||'|'||coalesce(invited_by::text,'operator') FROM network_members WHERE peer_id='{friend}'") == 'false|operator'
        checks['member_invite_cannot_restore_revoked_peer'] = True
        print(json.dumps({'checks': checks}, indent=2))
    finally:
        if portal and portal.poll() is None:
            os.killpg(portal.pid, 15)
            portal.wait(timeout=10)
        if pg.exists():
            subprocess.run(['pg_ctl', '-D', str(pg), '-m', 'fast', 'stop'], capture_output=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
