#!/usr/bin/env python3
"""Package a natively built worker. Does not build, enroll, publish or embed secrets."""
import argparse, hashlib, os, plistlib, re, shutil, subprocess, tarfile, tempfile, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGETS={'aarch64-apple-darwin','x86_64-unknown-linux-gnu','aarch64-unknown-linux-gnu','x86_64-pc-windows-msvc'}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--version',required=True);p.add_argument('--target',required=True,choices=sorted(TARGETS));p.add_argument('--binary',required=True,type=Path);p.add_argument('--output',type=Path,default=ROOT/'dist');p.add_argument('--dmg',action='store_true')
 a=p.parse_args();assert re.fullmatch(r'v[0-9][A-Za-z0-9._-]*',a.version),'Invalid version'
 binary=a.binary.resolve();assert binary.is_file(),'Build the native executable first'
 assert not a.dmg or a.target=='aarch64-apple-darwin','DMG requires macOS target'
 a.output.mkdir(parents=True,exist_ok=True)
 stem=f'sangama-{a.version}-{a.target}'
 with tempfile.TemporaryDirectory(prefix='sangama-package-') as tmp:
  tmp=Path(tmp);name='sangama.exe' if 'windows' in a.target else 'sangama'
  shutil.copyfile(binary,tmp/name);(tmp/name).chmod(0o755)
  shutil.copyfile(ROOT/'docs/distribution.md',tmp/'README.md')
  if 'windows' in a.target:
   shutil.copyfile(ROOT/'packaging/start-worker.cmd',tmp/'start-worker.cmd')
   with zipfile.ZipFile(a.output/(stem+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(tmp.iterdir()):z.write(f,f.name)
  else:
   with tarfile.open(a.output/(stem+'.tar.gz'),'w:gz') as t:
    for f in sorted(tmp.iterdir()):t.add(f,arcname=f.name)
  if a.dmg:
   volume=tmp/'volume';volume.mkdir();app=volume/'Sangama.app'
   subprocess.run(['osacompile','-o',str(app),str(ROOT/'packaging/launcher.applescript')],check=True)
   shutil.copyfile(binary,app/'Contents/Resources/sangama');(app/'Contents/Resources/sangama').chmod(0o755)
   info=app/'Contents/Info.plist'
   data=plistlib.loads(info.read_bytes());data.update(CFBundleIdentifier='net.sangama.worker',CFBundleShortVersionString='0.1.0',CFBundleVersion='1',NSAppleEventsUsageDescription='Open Terminal to run the worker you selected.')
   info.write_bytes(plistlib.dumps(data))
   identity=os.environ.get('SANGAMA_CODESIGN_IDENTITY','-')
   for path in [app/'Contents/Resources/sangama',app]:
    subprocess.run(['codesign','--force','--options','runtime','--sign',identity]+(['--entitlements',str(ROOT/'packaging/entitlements.plist')] if path==app else [])+[str(path)],check=True)
   (volume/'Applications').symlink_to('/Applications',target_is_directory=True)
   shutil.copyfile(ROOT/'docs/distribution.md',volume/'Read me.md')
   dest=a.output/(stem+'.dmg')
   subprocess.run(['hdiutil','create','-ov','-format','UDZO','-volname','Sangama','-srcfolder',str(volume),str(dest)],check=True)
   if identity!='-':subprocess.run(['codesign','--sign',identity,str(dest)],check=True)
 for name in ['install.sh','install.ps1']:shutil.copyfile(ROOT/'packaging'/name,a.output/name)
 checks=[]
 for path in sorted(a.output.iterdir()):
  if path.is_file() and path.name!='SHA256SUMS':checks.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name)
 (a.output/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
 print('Packages written to',a.output)
if __name__=='__main__':main()
