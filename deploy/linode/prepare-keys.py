#!/usr/bin/env python3
"""Generate dedicated deployment and server host keys. Does not provision resources."""
import os
from pathlib import Path
import secrets
import subprocess
root=Path(__file__).resolve().parents[2]/'.secrets/linode'
os.umask(0o077)
root.mkdir(mode=0o700,parents=True,exist_ok=True)
for name in ['id_ed25519','host_ed25519']:
    path=root/name
    if not path.exists():subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(path)],check=True)
password=root/'root-password'
if not password.exists():password.write_text(secrets.token_hex(32)+'Aa1!\n');password.chmod(0o600)
print('Dedicated deployment and pinned server-host keys are prepared.')
