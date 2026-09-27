#!/usr/bin/env python3
"""Install the prepared Sangama/OpenCode runtime as a user-level CLI (macOS/Linux)."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=Path.home()/'.local/share/sangama-code')
    parser.add_argument('--bin-dir', type=Path, default=Path.home()/'.local/bin')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'.models/qwen2.5-0.5b-instruct')
    parser.add_argument('--device', choices=['metal', 'cpu'], default='metal' if platform.system() == 'Darwin' and platform.machine() == 'arm64' else 'cpu')
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('This installer currently supports macOS/Linux; Windows CLI installation is not yet supported.')
    home, bin_dir, model = (p.expanduser().resolve() for p in (args.home, args.bin_dir, args.model_dir))
    sources = {
        'target/release/sangama': ROOT/'target/release/sangama',
        '.tools/opencode/node_modules/opencode-ai/bin/opencode.exe': ROOT/'.tools/opencode/node_modules/opencode-ai/bin/opencode.exe',
        'scripts/opencode.py': ROOT/'scripts/opencode.py',
        'integrations/opencode/opencode.json': ROOT/'integrations/opencode/opencode.json',
    }
    for source in sources.values():
        if not source.is_file():
            parser.error(f'Missing {source}; build Sangama and run ./scripts/install-opencode.sh first.')
    if not (model/'config.json').is_file():
        parser.error('Prepare the model first using python3 scripts/fetch-qwen.py, or pass --model-dir.')
    if home == ROOT or ROOT in home.parents:
        parser.error('--home must be outside the checkout.')
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    for relative, source in sources.items():
        destination = home/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name+'.new')
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    (home/'settings.json').write_text(json.dumps({'device': args.device, 'model_dir': str(model)}, indent=2)+'\n')
    bin_dir.mkdir(parents=True, exist_ok=True)
    command = bin_dir/'sangama-code'
    temporary = command.with_name(command.name+'.new')
    temporary.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(home/'scripts/opencode.py'))+' "$@"\n')
    temporary.chmod(0o755)
    temporary.replace(command)
    print(f'Installed {command}\nRuntime: {home}\nModel (shared, not copied): {model}\n\nFrom any project: sangama-code\nOne prompt: sangama-code -- run "Explain this code: ..."')
    if str(bin_dir) not in os.environ.get('PATH', '').split(os.pathsep):
        print('\nAdd this to your shell profile, then open a new terminal:\nexport PATH='+shlex.quote(str(bin_dir))+':"$PATH"')


if __name__ == '__main__':
    main()
