#!/usr/bin/env python3
"""Simulate a fleet of home devices serving the test model through the admitted mesh.

Each device runs in its own container on its own isolated Docker network, as if in its own
home, and can reach the others only through the relay. Device profiles set its memory, CPUs,
network shaping (delay, jitter, bandwidth, loss) and extra compute time, and make it sleep and
wake on a schedule (docker pause/unpause). While devices come and go, the client keeps choosing
a ready route and generating, and the report records what succeeded, how fast, and on which
devices.

Needs Docker, the prepared model (scripts/fetch-qwen.py), a debug build (for mesh identities),
and images built from the current tree:
  docker build -f docker/Dockerfile -t sangama:sim .
  docker build -f deploy/portal/Dockerfile -t sangama-portal:sim .
  docker build -f docker/Dockerfile.netem -t sangama-netem:test .
"""
import argparse, json, random, secrets, shutil, statistics, subprocess, threading, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = [30888, 4686, 78597, 24231, 6147, 3847, 311, 4564, 323, 4332, 4963, 2041, 279, 1184,
             369, 264, 8622, 3538, 13, 151645]


def run(*args, **kw):
    return subprocess.run(list(args), check=True, text=True, capture_output=True, **kw)


def docker(*args, **kw):
    return run('docker', *args, **kw)


def wait(fn, seconds=90, what='condition'):
    deadline, last = time.monotonic() + seconds, None
    while time.monotonic() < deadline:
        try:
            value = fn()
            if value:
                return value
        except Exception as e:  # probes fail until the node is up
            last = e
        time.sleep(1)
    raise RuntimeError(f'timed out waiting for {what}: {last}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fleet', type=Path, default=ROOT/'docs/simulation/fleet-mixed.json')
    parser.add_argument('--profiles', type=Path, default=ROOT/'docs/simulation/device-profiles.json')
    parser.add_argument('--duration', type=int, help='seconds of load; defaults to the fleet file')
    parser.add_argument('--image', default='sangama:sim')
    parser.add_argument('--portal-image', default='sangama-portal:sim')
    parser.add_argument('--netem-image', default='sangama-netem:test')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/fleet-simulation')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    random.seed(args.seed)
    profiles = json.loads(args.profiles.read_text())['profiles']
    fleet = json.loads(args.fleet.read_text())
    devices = fleet['devices']
    duration = args.duration or fleet.get('duration_s', 300)
    model = ROOT/'.models/qwen2.5-0.5b-instruct'
    manifest = json.loads((model/'manifest.json').read_text())
    shards = len(manifest['shards'])
    assert 1 <= len(devices) <= 8, 'a client bridges at most 8 devices'
    assert {d['shard'] for d in devices} == set(range(shards)), 'every shard needs a device'
    binary = ROOT/'target/debug/sangama'
    assert binary.exists(), 'build first: ./scripts/cargo build'

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    private = out/'private'; private.mkdir(mode=0o700, exist_ok=True); private.chmod(0o700)
    prefix = 'sangama-fleet-' + uuid.uuid4().hex[:8]
    networks, containers, volumes = [], [], []
    report = {'fleet': fleet.get('description'), 'profiles_note': json.loads(args.profiles.read_text())['note'],
              'devices': [], 'events': [], 'generations': []}
    stop = threading.Event()
    started = time.monotonic()

    def now():
        return round(time.monotonic() - started, 1)

    def create(name, net, extra, command):
        cname = f'{prefix}-{name}'
        docker('create', '--name', cname, '--network', net, '--network-alias', name, '--label', 'sangama.test=true',
               '--read-only', '--cap-drop=ALL', '--security-opt', 'no-new-privileges=true', '--pids-limit', '128',
               '--tmpfs', '/tmp:rw,nosuid,noexec,size=32m,mode=1777', '--entrypoint', 'python3', *extra, args.image, *command)
        containers.append(cname)
        return cname

    def execpy(name, code, timeout=60):
        return docker('exec', name, 'python3', '-c', code, timeout=timeout)

    try:
        for n in [d['name'] for d in devices] + ['client', 'db']:
            net = f'{prefix}-net-{n}'
            docker('network', 'create', '--internal', net)
            networks.append(net)
        net = {n.removeprefix(prefix + '-net-'): n for n in networks}

        # Membership authority and portal, as in scripts/test-mesh-containers.py.
        authority = private/'authority'
        run(str(binary), 'mesh-identity', '--state-dir', str(authority))
        (authority/'identity.key').chmod(0o444)
        dbpass = secrets.token_hex(32); (private/'dbpass').write_text(dbpass); (private/'dbpass').chmod(0o444)
        init = private/'init.sql'
        init.write_text(f"CREATE USER sangama PASSWORD '{dbpass}';\nCREATE DATABASE sangama OWNER sangama;\n"); init.chmod(0o444)
        db = f'{prefix}-postgres'
        docker('run', '-d', '--name', db, '--network', net['db'], '--network-alias', 'postgres',
               '--tmpfs', '/var/lib/postgresql/data:rw,size=512m',
               '--mount', f'type=bind,src={private}/dbpass,dst=/run/dbpass,readonly',
               '--mount', f'type=bind,src={init},dst=/docker-entrypoint-initdb.d/10-init.sql,readonly',
               '-e', 'POSTGRES_PASSWORD_FILE=/run/dbpass', 'postgres:17-bookworm')
        containers.append(db)
        wait(lambda: docker('exec', db, 'pg_isready', '-U', 'postgres').returncode == 0, what='postgres')
        portal = f'{prefix}-portal'
        docker('create', '--name', portal, '--network', net['db'], '--label', 'sangama.test=true', '--read-only',
               '--cap-drop=ALL', '--security-opt', 'no-new-privileges=true', '--entrypoint', '/usr/local/bin/sangama-portal',
               '--mount', f'type=bind,src={private}/dbpass,dst=/run/dbpass,readonly',
               '--mount', f'type=bind,src={authority}/identity.key,dst=/run/authority.key,readonly',
               '-e', 'DATABASE_PASSWORD_FILE=/run/dbpass', '-e', 'DATABASE_HOST=postgres',
               '-e', 'PUBLIC_ORIGIN=http://portal:8080', '-e', 'SIMULATION_HTTP=1', '-e', 'NETWORK_ID=simulation',
               '-e', 'MEMBERSHIP_KEY_FILE=/run/authority.key', '-e', 'CREDIT_ALLOWANCE=1000000', args.portal_image)
        containers.append(portal)
        for n, full in net.items():
            if n != 'db':
                docker('network', 'connect', '--alias', 'portal', full, portal)
        docker('start', portal)

        # Identities, tokens and state volumes for the relay, the client and every device.
        roles = ['relay', 'client'] + [d['name'] for d in devices]
        ids, state, names, vol = {}, {}, {}, {}
        for role in roles:
            state[role] = private/role
            ids[role] = run(str(binary), 'mesh-identity', '--state-dir', str(state[role])).stdout.strip()
            (state[role]/'token').write_text(secrets.token_hex(32)); (state[role]/'token').chmod(0o600)
            (state[role]/'authority.pub').write_bytes((authority/'public.key').read_bytes())
            vol[role] = f'{prefix}-{role}-state'; docker('volume', 'create', vol[role]); volumes.append(vol[role])

        keep_alive = ['-c', 'import time; time.sleep(86400)']
        common = lambda role: ['--mount', f'type=bind,src={ROOT}/scripts/mesh-node.py,dst=/app/mesh-node.py,readonly',
                               '--mount', f'type=volume,src={vol[role]},dst=/state']
        names['relay'] = create('relay', net['client'], common('relay') + ['--memory', '512m', '--cpus', '1'], keep_alive)
        client_files = ['config.json', 'manifest.json', 'tokenizer.json']
        names['client'] = create('client', net['client'], common('client') + ['--memory', '768m', '--cpus', '2'] +
                                 sum((['--mount', f'type=bind,src={model/f},dst=/model/{f},readonly'] for f in client_files), []), keep_alive)
        for d in devices:
            p = profiles[d['profile']]
            files = ['config.json', 'manifest.json', manifest['shards'][d['shard']]['file']]
            extra = common(d['name']) + ['--memory', f"{p['memory_mib']}m", '--cpus', str(p['cpus'])]
            extra += sum((['--mount', f'type=bind,src={model/f},dst=/model/{f},readonly'] for f in files), [])
            if p['compute_ms_per_layer_token']:
                extra += ['-e', f"SANGAMA_SIMULATE_MS_PER_LAYER_TOKEN={p['compute_ms_per_layer_token']}"]
            names[d['name']] = create(d['name'], net[d['name']], extra, keep_alive)
            report['devices'].append({'name': d['name'], 'profile': d['profile'], 'shard': d['shard'], 'represents': p['represents']})
        for n, full in net.items():
            if n not in ('client', 'db'):
                docker('network', 'connect', full, names['relay'])
        for role in roles:
            docker('start', names[role])

        relay_ips = json.loads(docker('inspect', names['relay']).stdout)[0]['NetworkSettings']['Networks']
        bridge = {d['name']: f'127.0.0.1:{7901 + i}' for i, d in enumerate(devices)}
        bridges = [{'listen': bridge[d['name']], 'peer': ids[d['name']]} for d in devices]
        for role in roles:
            home = net['client'] if role in ('relay', 'client') else net[role]
            config = {'state_dir': '/state', 'authority_file': '/state/authority.pub', 'network': 'simulation',
                      'portal': 'http://portal:8080', 'test_http': True, 'listen': '/ip4/0.0.0.0/tcp/9000',
                      'external': [], 'relay_server': role == 'relay', 'token_file': '/state/token', 'force_relay': True,
                      'relay': None if role == 'relay' else f"/ip4/{relay_ips[home]['IPAddress']}/tcp/9000/p2p/{ids['relay']}",
                      'worker': None if role in ('relay', 'client') else '127.0.0.1:7900',
                      'bridges': [] if role == 'relay' else bridges}
            if role == 'relay':
                config['external'] = [f"/ip4/{e['IPAddress']}/tcp/9000" for e in relay_ips.values()]
            elif role != 'client':
                p = profiles[next(d['profile'] for d in devices if d['name'] == role)]
                config['managed'] = {'model_dir': '/model', 'device': 'cpu', 'memory_budget_mib': p['memory_budget_mib']}
            (state[role]/'config.json').write_text(json.dumps(config))
            kind = 'worker' if role not in ('relay', 'client') else role
            invite = wait(lambda: docker('exec', portal, 'sangama-portal', 'network-invite', kind).stdout.strip(), what='invitation')
            (state[role]/'invite').write_text(invite); (state[role]/'invite').chmod(0o600)
            docker('run', '--rm', '--network', 'none', '--user', '0:0', '--entrypoint', 'python3',
                   '--mount', f'type=bind,src={state[role]},dst=/input,readonly',
                   '--mount', f'type=volume,src={vol[role]},dst=/output', args.image, '-c',
                   "import pathlib,shutil,os; src=pathlib.Path('/input');dst=pathlib.Path('/output');"
                   "[shutil.copyfile(p,dst/p.name) for p in src.iterdir() if p.is_file() and p.name!='node.lock'];"
                   "os.chown(dst,10001,10001);os.chmod(dst,0o700);[(os.chown(p,10001,10001),os.chmod(p,0o600)) for p in dst.iterdir()]")
            docker('exec', names[role], 'sangama', 'mesh-join', '--config', '/state/config.json', '--invitation-file', '/state/invite')

        for d in devices:
            cmd = ['python3', '/app/mesh-node.py', '--binary', '/usr/local/bin/sangama', '--config', '/state/config.json',
                   '--model-dir', '/model', '--device', 'cpu']
            execpy(names[d['name']], f"import subprocess; subprocess.Popen({cmd!r},stdout=open('/state/mesh.log','w'),stderr=subprocess.STDOUT)")
        for role in ('relay', 'client'):
            execpy(names[role], "import subprocess;subprocess.Popen(['sangama','mesh','--config','/state/config.json'],"
                                "stdout=open('/state/mesh.log','w'),stderr=subprocess.STDOUT)")

        # Shape each device's traffic from a short-lived NET_ADMIN helper in its namespace only.
        for d in devices:
            n = profiles[d['profile']]['network']
            cmd = ['qdisc', 'replace', 'dev', 'eth0', 'root', 'netem', 'delay', f"{n['delay_ms']}ms", f"{n['jitter_ms']}ms",
                   'loss', f"{n['loss_percent']}%", 'rate', f"{n['rate_mbit']}mbit"]
            docker('run', '--rm', '--network', 'container:' + names[d['name']], '--cap-drop=ALL', '--cap-add=NET_ADMIN', args.netem_image, *cmd)

        def probe(address, path):
            code = ("import urllib.request;print(urllib.request.urlopen(urllib.request.Request("
                    f"'http://{address}{path}',headers={{'Authorization':'Bearer '+open('/state/token').read()}}),timeout=5).status)")
            return execpy(names['client'], code).stdout.strip() == '200'
        for d in devices:
            wait(lambda: probe(bridge[d['name']], '/v1/node/capacity'), 180, f"{d['name']} through the relay")
        report['events'].append({'t': now(), 'event': 'all devices reachable through the relay'})

        # Load a complete route per replica, so a sleeping device has a warm stand-in.
        by_shard = [[d for d in devices if d['shard'] == s] for s in range(shards)]
        for replica in range(max(len(group) for group in by_shard)):
            group = [g[replica % len(g)] for g in by_shard]
            candidates = ','.join(bridge[d['name']] for d in group)
            docker('exec', names['client'], 'sangama', '--token-file', '/state/token', 'mesh-allocate',
                   '--model-dir', '/model', '--candidates', candidates, timeout=300)
        report['events'].append({'t': now(), 'event': 'all devices loaded their shards'})

        # Devices sleep and wake on their profile's schedule, starting at a random phase.
        def churn(d):
            a = profiles[d['profile']]['availability']
            if a['kind'] != 'sleeps':
                return
            stop.wait(random.uniform(0, a['awake_s']))
            while not stop.is_set():
                docker('pause', names[d['name']]); report['events'].append({'t': now(), 'device': d['name'], 'event': 'asleep'})
                stop.wait(a['asleep_s'])
                docker('unpause', names[d['name']]); report['events'].append({'t': now(), 'device': d['name'], 'event': 'awake'})
                stop.wait(a['awake_s'])
        threads = [threading.Thread(target=churn, args=(d,), daemon=True) for d in devices]
        load_start = now()
        for t in threads:
            t.start()
        port_to_device = {v: k for k, v in bridge.items()}
        while now() - load_start < duration:
            record = {'t': now()}
            plan = subprocess.run(['docker', 'exec', names['client'], 'sangama', '--token-file', '/state/token', 'mesh-plan',
                                   '--model-dir', '/model', '--candidates', ','.join(bridge.values())],
                                  capture_output=True, text=True, timeout=120)
            if plan.returncode:
                record |= {'ok': False, 'stage': 'plan', 'error': (plan.stderr.strip().splitlines() or [''])[-1][:200]}
                report['generations'].append(record); time.sleep(2); continue
            peers = json.loads(plan.stdout)['peers']
            record['route'] = [port_to_device[p] for p in peers]
            gen = subprocess.run(['docker', 'exec', names['client'], 'sangama', '--token-file', '/state/token', 'generate',
                                  '--model-dir', '/model', '--device', 'cpu', '--peers', ','.join(peers), '--max-tokens', '20'],
                                 capture_output=True, text=True, timeout=300)
            if gen.returncode:
                record |= {'ok': False, 'stage': 'generate', 'error': (gen.stderr.strip().splitlines() or [''])[-1][:200]}
            else:
                g = json.loads(gen.stdout)
                record |= {'ok': True, 'exact': g['distributed_token_ids'] == REFERENCE,
                           'first_token_ms': round(g['distributed']['first_token_ms']),
                           'decode_tokens_per_second': round(g['distributed']['decode_tokens_per_second'], 2)}
            report['generations'].append(record)
            print(json.dumps(record), flush=True)
        stop.set()
        for t in threads:
            t.join(timeout=60)

        gens = report['generations']
        ok = [g for g in gens if g.get('ok')]
        routes = {}
        for g in ok:
            routes.setdefault(' -> '.join(g['route']), []).append(g['decode_tokens_per_second'])
        report['summary'] = {
            'duration_s': duration, 'attempts': len(gens), 'succeeded': len(ok),
            'exact': sum(1 for g in ok if g['exact']),
            'failed_at_plan': sum(1 for g in gens if g.get('stage') == 'plan'),
            'failed_mid_generation': sum(1 for g in gens if g.get('stage') == 'generate'),
            'median_first_token_ms': statistics.median([g['first_token_ms'] for g in ok]) if ok else None,
            'routes': {r: {'generations': len(v), 'median_decode_tokens_per_second': statistics.median(v)} for r, v in routes.items()},
            'sleep_events': sum(1 for e in report['events'] if e.get('event') == 'asleep'),
        }
        print(json.dumps(report['summary'], indent=2))
    finally:
        stop.set()
        (out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
        for c in containers:
            subprocess.run(['docker', 'unpause', c], capture_output=True)
            logs = subprocess.run(['docker', 'logs', c], capture_output=True, text=True)
            (out/(c.removeprefix(prefix + '-') + '.log')).write_text(logs.stdout + logs.stderr)
            subprocess.run(['docker', 'cp', c + ':/state/mesh.log', str(out/(c.removeprefix(prefix + '-') + '-mesh.log'))], capture_output=True)
            subprocess.run(['docker', 'rm', '-f', c], capture_output=True)
        for v in volumes:
            subprocess.run(['docker', 'volume', 'rm', v], capture_output=True)
        for n in reversed(networks):
            subprocess.run(['docker', 'network', 'rm', n], capture_output=True)
        shutil.rmtree(private)  # invitations and identities never stay in reports
    print(f'report: {out/"report.json"}')


if __name__ == '__main__':
    main()
