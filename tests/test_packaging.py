"""Installer tests use only disposable local fixtures; no network or real install."""
import hashlib,io,os,platform,subprocess,tarfile,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipIf(os.name=='nt','POSIX installer; PowerShell separately smoke-tested on Windows CI')
class Installer(unittest.TestCase):
 def fixture(self,root):
  target={('Darwin','arm64'):'aarch64-apple-darwin',('Linux','x86_64'):'x86_64-unknown-linux-gnu',('Linux','aarch64'):'aarch64-unknown-linux-gnu'}.get((platform.system(),platform.machine()))
  if not target:self.skipTest('not a release target')
  release=root/'release';release.mkdir();asset=release/f'sangama-v0.1.0-preview.1-{target}.tar.gz'
  with tarfile.open(asset,'w:gz') as t:
   data=b'#!/bin/sh\necho sangama-test\n';entry=tarfile.TarInfo('sangama');entry.size=len(data);entry.mode=0o755;t.addfile(entry,io.BytesIO(data))
  (release/'SHA256SUMS').write_text(hashlib.sha256(asset.read_bytes()).hexdigest()+'  '+asset.name+'\n')
  return release,asset
 def run_install(self,root,release):
  env=os.environ|{'SANGAMA_RELEASE_DIR':str(release),'SANGAMA_BIN_DIR':str(root/'bin with spaces'),'SANGAMA_VERSION':'v0.1.0-preview.1'}
  return subprocess.run(['sh',str(ROOT/'packaging/install.sh')],env=env,capture_output=True,text=True)
 def test_verified_install_and_tampering_preserves_existing_binary(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);release,asset=self.fixture(root)
   result=self.run_install(root,release);self.assertEqual(result.returncode,0,result.stderr)
   binary=root/'bin with spaces/sangama';before=binary.read_bytes()
   asset.write_bytes(b'corrupted')
   result=self.run_install(root,release);self.assertNotEqual(result.returncode,0);self.assertIn('Checksum mismatch',result.stderr);self.assertEqual(binary.read_bytes(),before)
 def test_missing_checksum_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);release,_=self.fixture(root);(release/'SHA256SUMS').write_text('')
   self.assertNotEqual(self.run_install(root,release).returncode,0)
   self.assertFalse((root/'bin with spaces/sangama').exists())
if __name__=='__main__':unittest.main()
