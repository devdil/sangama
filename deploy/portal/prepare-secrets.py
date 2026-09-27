#!/usr/bin/env python3
"""Create dedicated database secrets in a private directory; never print values."""
from pathlib import Path
import os
import secrets
root=Path(__file__).resolve().parent/'secrets'
root.mkdir(mode=0o700,exist_ok=True)
if root.is_symlink():raise RuntimeError('Secret directory must not be a symlink')
root.chmod(0o700)
for name in ['postgres_password','app_password','admin_token']:
    file=root/name
    if file.exists():
        if file.is_symlink():raise RuntimeError('Secret file must not be a symlink')
        continue
    fd=os.open(file,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o400)
    with os.fdopen(fd,'w') as stream:stream.write(secrets.token_hex(32)+'\n')
    # Docker mounts only the named secret into authorized containers. The host parent is 0700.
    file.chmod(0o444)
print('Database and operator secrets prepared in a private directory.')
