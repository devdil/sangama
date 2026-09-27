#!/usr/bin/env python3
"""Create a fresh 256-bit test token without printing it or overwriting an existing one."""
import os
from pathlib import Path
import secrets

folder = Path('.secrets')
folder.mkdir(mode=0o700, exist_ok=True)
if folder.is_symlink() or folder.stat().st_mode & 0o077:
    raise SystemExit('.secrets must be a real private directory (chmod 700 .secrets)')
path = folder / 'peer.token'
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as output:
    output.write(secrets.token_hex(32) + '\n')
print('Created .secrets/peer.token (mode 600). Transfer privately; never paste it into chat or commit it.')
