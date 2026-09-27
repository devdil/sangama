#!/usr/bin/env python3
"""Open a pinned-key SSH tunnel to a peer over its Tailscale IPv4 address.

No credentials are generated, copied, or transmitted by this helper.
"""
import argparse
import ipaddress
import os
from pathlib import Path
import re
import subprocess
import sys


def command(args):
    host = ipaddress.ip_address(args.host)
    if host not in ipaddress.ip_network('100.64.0.0/10'):
        raise ValueError('use the peer Tailscale IPv4 address (100.64.0.0/10)')
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_-]*', args.user):
        raise ValueError('invalid SSH username')
    if not (args.ssh_port == 22 or 1024 <= args.ssh_port <= 65535):
        raise ValueError("use SSH port 22 or 1024..65535")
    if not all(1024 <= port <= 65535 for port in (args.local_port, args.remote_port)):
        raise ValueError("use worker ports 1024..65535")
    identity = Path(args.identity).resolve(strict=True)
    known = Path(args.known_hosts).resolve(strict=True)
    if identity.stat().st_mode & 0o077:
        raise ValueError('SSH private key must have mode 600 or stricter')
    if not identity.is_file() or not known.is_file() or not known.stat().st_size:
        raise ValueError('private key and nonempty verified known_hosts file are required')
    return ['ssh', '-F', '/dev/null', '-N', '-T', '-p', str(args.ssh_port),
            '-i', str(identity), '-o', f'UserKnownHostsFile={known}',
            '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UpdateHostKeys=no', '-o', 'VerifyHostKeyDNS=no',
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
            '-o', 'PasswordAuthentication=no', '-o', 'KbdInteractiveAuthentication=no',
            '-o', 'PreferredAuthentications=publickey', '-o', 'ForwardAgent=no',
            '-o', 'ForwardX11=no', '-o', 'GatewayPorts=no', '-o', 'ControlMaster=no',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-L', f'127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}',
            f'{args.user}@{host}']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--identity', default='.secrets/peer_ed25519')
    parser.add_argument('--known-hosts', default='.secrets/known_hosts')
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--local-port', type=int, default=7902)
    parser.add_argument('--remote-port', type=int, default=7902)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        argv = command(args)
        if args.dry_run:
            import shlex
            print(shlex.join(argv))
        else:
            print('Opening authenticated tunnel; keep this terminal open. Ctrl-C closes it.', flush=True)
            os.execvp(argv[0], argv)
    except (ValueError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
