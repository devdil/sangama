#!/usr/bin/env python3
"""Extract the integrity-pinned Linux ARM64 OpenCode binary for Docker tests."""
import base64,hashlib,io,json,tarfile,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
pkg=json.loads((root/'integrations/opencode/package-lock.json').read_text())['packages']['node_modules/opencode-linux-arm64']
archive=root/'.tools/opencode-linux/package.tgz';archive.parent.mkdir(parents=True,exist_ok=True)
if not archive.exists():
    with urllib.request.urlopen(pkg['resolved'],timeout=60) as r:archive.write_bytes(r.read())
data=archive.read_bytes()
assert 'sha512-'+base64.b64encode(hashlib.sha512(data).digest()).decode()==pkg['integrity'],'OpenCode integrity mismatch'
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
    member=tar.getmember('package/bin/opencode');assert member.isfile()
    binary=archive.parent/'opencode';binary.write_bytes(tar.extractfile(member).read());binary.chmod(0o755)
print('Pinned OpenCode Linux ARM64 binary prepared.')
