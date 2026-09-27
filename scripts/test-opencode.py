#!/usr/bin/env python3
"""Run a real OpenCode coding suggestion through Sangama; no model-generated code is executed."""
import ast
import json
import os
import tempfile
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
prompt='The function add must return the SUM of a and b. It currently contains a bug: def add(a, b): return a - b. Replace subtraction with addition. Output only the corrected Python code.'
command = [os.environ['SANGAMA_CODE_TEST_CLI']] if os.environ.get('SANGAMA_CODE_TEST_CLI') else [sys.executable, str(ROOT/'scripts/opencode.py')]
with tempfile.TemporaryDirectory(prefix='sangama-code-project-') as directory:
    project = Path(directory)
    # Project config must not load plugins, change providers, or enable tools.
    (project/'opencode.json').write_text(json.dumps({'plugin': ['file:///nonexistent/sangama-test-plugin.js'], 'model': 'invalid/no-cloud', 'permission': {'*': 'allow'}}))
    plugins = project/'.opencode/plugins'
    plugins.mkdir(parents=True)
    (plugins/'blocked.js').write_text('throw new Error("Project plugin must not execute");')
    result=subprocess.run([*command,'--','run','--format','json',prompt],cwd=project,capture_output=True,text=True,timeout=180)
(ROOT/'runs/opencode-code-fix.jsonl').write_text(result.stdout)
(ROOT/'runs/opencode-code-fix.log').write_text(result.stderr)
assert result.returncode==0,result.stderr
records=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
assert not any(r['type']=='error' for r in records),records
text=''.join(r['part']['text'] for r in records if r['type']=='text')
match=re.search(r'```(?:python)?\n(.*?)```',text,re.S)
assert match,'Expected a Python code block'
module=ast.parse(match.group(1))
assert len(module.body)==1 and isinstance(module.body[0],ast.FunctionDef)
function=module.body[0]
assert function.name=='add' and len(function.body)==1 and isinstance(function.body[0],ast.Return)
value=function.body[0].value
assert isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add)
assert isinstance(value.left,ast.Name) and value.left.id=='a'
assert isinstance(value.right,ast.Name) and value.right.id=='b'
report={'transport_passed':True,'explicit_code_suggestion_passed':True,'client_version':json.loads((ROOT/'integrations/opencode/package.json').read_text())['dependencies']['opencode-ai'], 'model':'Qwen/Qwen2.5-0.5B-Instruct','topology':'OpenCode -> localhost Rust gateway -> two separate Metal shard worker processes on one Mac','prompt':prompt,'response':text,'automatic_file_edits':False,'tool_calling':False,'validation':'Parsed returned code with Python AST; asserted add returns a + b. Did not execute model-generated code.', 'events':records}
(ROOT/'runs/opencode-integration.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'client_version':report['client_version'],'corrected_code':match.group(1)},indent=2))
