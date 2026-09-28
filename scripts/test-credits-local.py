#!/usr/bin/env python3
"""Exercise contribution credits end to end with native processes on this computer.

Starts a throwaway PostgreSQL (TCP only, private port), the portal, a relay, one
real Qwen worker serving every layer, and a client, all admitted with invitations.
Checks that both sides' receipts reach the ledger and agree, that a client past its
allowance is refused new sessions, and that linking both peers to one account settles
the debt. Everything runs under runs/ and is removed afterwards.

Two-worker routes need distinct loopback aliases per node; use the Docker simulation
for those. Requires release builds, PostgreSQL server binaries and the prepared model.
"""
import json, os, secrets, shutil, socket, subprocess, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'target/release/sangama'
PORTAL_BIN = ROOT / 'portal/target/release/sangama-portal'
MODEL = ROOT / '.models/qwen2.5-0.5b-instruct'
NETWORK = 'credits-test'
LAYERS = 24


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait(check, seconds, what):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except Exception as e:  # noqa: BLE001 - report the final failure below
            last = e
        time.sleep(1)
    raise AssertionError(f'Timed out waiting for {what}: {last}')


def main():
    work = ROOT / 'runs' / f'credits-{uuid.uuid4().hex[:8]}'
    work.mkdir(parents=True)
    processes = []
    pg = work / 'pg'
    pg_port, portal_port, relay_port = free_port(), free_port(), free_port()
    report = {'checks': {}}
    checks = report['checks']
    try:
        # One shard covering every layer; the checkpoint is hard-linked, not copied.
        model = work / 'model'
        model.mkdir()
        for name in ['model.safetensors', 'config.json', 'tokenizer.json', 'LICENSE']:
            os.link(MODEL / name, model / name)
        subprocess.run(['python3', str(ROOT / 'scripts/fetch-qwen.py'), '--model-dir', str(model), '--shards', '1'],
                       check=True, capture_output=True)

        subprocess.run(['initdb', '-D', str(pg), '-U', 'postgres', '-A', 'trust'], check=True, capture_output=True)
        subprocess.run(['pg_ctl', '-D', str(pg), '-o', f"-p {pg_port} -k '' -c listen_addresses=127.0.0.1",
                        '-l', str(work / 'pg.log'), 'start'], check=True, capture_output=True)

        def psql(query, db='sangama'):
            return subprocess.run(['psql', '-h', '127.0.0.1', '-p', str(pg_port), '-U', 'postgres', '-d', db,
                                   '-v', 'ON_ERROR_STOP=1', '-Atc', query],
                                  check=True, capture_output=True, text=True).stdout.strip()
        dbpass = secrets.token_hex(16)
        wait(lambda: psql('SELECT 1', 'postgres') == '1', 30, 'PostgreSQL')
        psql(f"CREATE USER sangama PASSWORD '{dbpass}'", 'postgres')
        psql('CREATE DATABASE sangama OWNER sangama', 'postgres')
        (work / 'dbpass').write_text(dbpass)

        authority = work / 'authority'
        subprocess.run([str(BIN), 'mesh-identity', '--state-dir', str(authority)], check=True, capture_output=True)
        origin = f'http://127.0.0.1:{portal_port}'
        portal_env = os.environ | {
            'DATABASE_PASSWORD_FILE': str(work / 'dbpass'), 'DATABASE_HOST': '127.0.0.1',
            'DATABASE_PORT': str(pg_port), 'PUBLIC_ORIGIN': origin, 'LISTEN_ADDR': f'127.0.0.1:{portal_port}',
            'NETWORK_ID': NETWORK, 'MEMBERSHIP_KEY_FILE': str(authority / 'identity.key'),
            # One credit: 1,000 layer-tokens. One generation below (40+ tokens x 24 layers) exceeds it.
            'CREDIT_ALLOWANCE': '1',
        }

        def portal_cli(*args):
            return subprocess.run([str(PORTAL_BIN), *args], env=portal_env, check=True,
                                  capture_output=True, text=True).stdout.strip()

        def spawn(name, cmd, env=None):
            log = open(work / f'{name}.log', 'w')
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            processes.append(p)
            return p

        spawn('portal', [str(PORTAL_BIN)], portal_env)

        def portal_up():
            import urllib.request
            with urllib.request.urlopen(origin + '/healthz', timeout=2) as r:
                return r.status == 200
        wait(portal_up, 30, 'portal')

        ids, state = {}, {}
        for role in ['relay', 'worker', 'client']:
            state[role] = work / role
            ids[role] = subprocess.run([str(BIN), 'mesh-identity', '--state-dir', str(state[role])],
                                       check=True, capture_output=True, text=True).stdout.strip()
            token = state[role] / 'token'
            token.write_text(secrets.token_hex(32))
            token.chmod(0o600)
        relay_addr = f'/ip4/127.0.0.1/tcp/{relay_port}/p2p/{ids["relay"]}'
        worker_http, bridge = f'127.0.0.1:{free_port()}', f'127.0.0.1:{free_port()}'
        for role in ['relay', 'worker', 'client']:
            port = relay_port if role == 'relay' else free_port()
            config = {
                'state_dir': str(state[role]), 'authority_file': str(authority / 'public.key'),
                'network': NETWORK, 'portal': origin, 'test_http': True,
                'listen': f'/ip4/127.0.0.1/tcp/{port}',
                'external': [f'/ip4/127.0.0.1/tcp/{relay_port}'] if role == 'relay' else [],
                'relay_server': role == 'relay', 'relay': None if role == 'relay' else relay_addr,
                'worker': worker_http if role == 'worker' else None,
                'token_file': str(state[role] / 'token'), 'force_relay': True,
                'bridges': [{'listen': bridge, 'peer': ids['worker']}] if role == 'client' else [],
            }
            (state[role] / 'config.json').write_text(json.dumps(config))
            invitation = state[role] / 'invite'
            invitation.write_text(portal_cli('network-invite', role))
            invitation.chmod(0o600)
            subprocess.run([str(BIN), 'mesh-join', '--config', str(state[role] / 'config.json'),
                            '--invitation-file', str(invitation)], check=True, capture_output=True)
        checks['all_roles_admitted'] = True

        spawn('relay', [str(BIN), 'mesh', '--config', str(state['relay'] / 'config.json')])
        time.sleep(1)
        spawn('qwen-worker', [str(BIN), '--token-file', str(state['worker'] / 'token'), 'qwen-worker',
                              '--model-dir', str(model), '--device', 'cpu', '--shard', '0', '--listen', worker_http])
        worker_mesh = spawn('worker', [str(BIN), 'mesh', '--config', str(state['worker'] / 'config.json')])
        client_mesh = spawn('client', [str(BIN), 'mesh', '--config', str(state['client'] / 'config.json')])

        client_token = state['client'] / 'token'

        def reachable():
            import urllib.request
            req = urllib.request.Request(f'http://{bridge}/v1/qwen/info',
                                         headers={'Authorization': 'Bearer ' + client_token.read_text()})
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        wait(reachable, 180, 'worker through the relay')
        checks['worker_reachable_through_relay'] = True

        def generate():
            return subprocess.run([str(BIN), '--token-file', str(client_token), 'generate', '--model-dir', str(model),
                                   '--device', 'cpu', '--peers', bridge, '--max-tokens', '40',
                                   '--prompt', 'Explain peer-to-peer computing in one short sentence.'],
                                  capture_output=True, text=True, timeout=300)
        first = generate()
        assert first.returncode == 0, first.stderr
        report['generation'] = json.loads(first.stdout)['distributed_text']
        checks['generation_before_allowance_is_used'] = True

        # Receipts are sent 90 s after a session goes idle, on a 30 s cycle.
        rows = wait(lambda: (r := psql("SELECT kind||':'||tokens||':'||signer FROM credit_receipts ORDER BY kind"))
                    and len(r.splitlines()) == 2 and r, 240, 'both receipts')
        (usage, work_row) = [line.split(':') for line in rows.splitlines()]
        assert usage[0] == 'usage' and usage[2] == ids['client'], usage
        assert work_row[0] == 'work' and work_row[2] == ids['worker'], work_row
        assert usage[1] == work_row[1], 'worker and client counted different tokens'
        tokens = int(usage[1])
        report['tokens'] = tokens
        checks['both_sides_receipts_accepted_and_agree'] = True

        units = int(psql('SELECT units FROM credit_entries'))
        assert units == tokens * LAYERS, (units, tokens)
        balances = dict(line.split('|') for line in psql('SELECT holder, balance FROM credit_balances').splitlines())
        assert int(balances['peer:' + ids['worker']]) == units
        assert int(balances['peer:' + ids['client']]) == -units
        checks['worker_credited_client_debited_zero_sum'] = True

        # The worker refreshes its standing every 10 s. Should the first session fall short of the
        # allowance, later sessions' receipts push the client past it.
        def refused():
            r = generate()
            return r.returncode != 0 and '402' in r.stderr
        wait(refused, 300, 'reservation refused past allowance')
        checks['reservation_refused_past_allowance'] = True

        # Linking both peers to one account nets their balances to zero.
        psql("INSERT INTO invitations (token_hash) VALUES ('credits-test') ")
        psql("INSERT INTO accounts (username,password_hash,invitation_id) "
             "SELECT 'credits_test','unused',id FROM invitations WHERE token_hash='credits-test'")
        for role in ['worker', 'client']:
            portal_cli('link-peer', ids[role], 'credits_test')
        assert 'credits_test (account)\t0.0' in portal_cli('credits')
        second = wait(lambda: (r := generate()).returncode == 0 and r, 60, 'reservation after linking')
        assert second
        checks['linked_account_settles_and_restores_access'] = True

        # A replayed receipt changes nothing; a forged one is refused.
        signed = psql("SELECT signed FROM credit_receipts WHERE kind='work'")
        import urllib.error, urllib.request

        def post(body):
            req = urllib.request.Request(origin + '/v1/credits/receipts', data=body.encode(),
                                         headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e:
                return e.code, None
        assert post(signed) == (200, {'accepted': False})
        forged = json.loads(signed)
        forged['receipt']['tokens'] += 1000
        assert post(json.dumps(forged))[0] == 400
        checks['replay_ignored_and_forgery_refused'] = True

        # Stopping nodes reports sessions still inside the idle window instead of dropping them.
        received = int(psql('SELECT count(*) FROM credit_receipts'))
        assert generate().returncode == 0
        for node in (worker_mesh, client_mesh):
            node.terminate()
            node.wait(timeout=15)
        unmatched = psql("SELECT count(*) FROM credit_receipts a WHERE NOT EXISTS (SELECT 1 FROM credit_receipts b "
                         "WHERE b.session=a.session AND b.kind<>a.kind)")
        assert int(psql('SELECT count(*) FROM credit_receipts')) >= received + 2 and unmatched == '0'
        checks['shutdown_sends_pending_receipts'] = True
        print(json.dumps(report, indent=2))
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, 15)
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
        if pg.exists():
            subprocess.run(['pg_ctl', '-D', str(pg), '-m', 'fast', 'stop'], capture_output=True)
        if os.environ.get('KEEP_CREDITS_TEST') != '1':
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f'Kept {work}')


if __name__ == '__main__':
    main()
