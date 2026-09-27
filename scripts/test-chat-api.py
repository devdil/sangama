#!/usr/bin/env python3
"""Exercise a running gateway started by scripts/opencode.py --serve-only."""
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
TOKEN=(ROOT/'.secrets/opencode/api.token').read_text().strip()
BASE='http://127.0.0.1:8090'
MODEL='qwen2.5-0.5b-instruct'
def request(path,body=None,headers=None):
    h={'Authorization':'Bearer '+TOKEN,'Content-Type':'application/json'}
    h.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=h),timeout=90)
def rejected(status,path='/v1/models',body=None,headers=None):
    try:
        request(path,body,headers)
        raise AssertionError('Expected rejection')
    except urllib.error.HTTPError as e:assert e.code==status,(e.code,e.read())

assert json.load(request('/v1/models'))['data'][0]['id']==MODEL
rejected(401,headers={'Authorization':'Bearer wrong'})
rejected(403,headers={'Origin':'https://example.com'})
rejected(403,headers={'Host':'evil.example'})
base={'model':MODEL,'messages':[{'role':'user','content':'What is 2 + 2? Answer with only the number.'}],'max_tokens':8,'temperature':0}
for patch in [{'tools':[{'type':'function','function':{'name':'bash'}}]},{'max_tokens':129},{'temperature':1},{'model':'unknown'}]:
    rejected(400,'/v1/chat/completions',base|patch)
plain=json.load(request('/v1/chat/completions',base))
assert plain['choices'][0]['message']['content'].strip()=='4'
started=time.monotonic();pieces=[];events=[];arrival=[];done=False
with request('/v1/chat/completions',base|{'stream':True}) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        data=line[6:].strip()
        if data==b'[DONE]':
            done=True
            break
        event=json.loads(data);assert 'error' not in event,event
        events.append(event)
        for choice in event['choices']:
            delta=choice['delta'].get('content','')
            if delta:pieces.append(delta);arrival.append(time.monotonic()-started)
assert done and len({event['id'] for event in events})==1
assert ''.join(pieces)==plain['choices'][0]['message']['content']
assert events[-1]['choices'][0]['finish_reason']=='stop'
# This exceeds the old 512-token prefill ceiling while keeping the answer deterministic.
long=base|{'messages':[{'role':'system','content':'You are a helpful assistant. '+('The following is background context. '*110)},{'role':'user','content':'What is 2 + 2? Answer with only the number.'}]}
long_result=json.load(request('/v1/chat/completions',long))
assert long_result['usage']['prompt_tokens']>512
assert long_result['choices'][0]['message']['content'].strip()=='4'
conversation=base|{'messages':[{'role':'user','content':'Remember the number 17.'},{'role':'assistant','content':'I will remember 17.'},{'role':'user','content':'What number did I ask you to remember? Answer with only the number.'}]}
recalled=json.load(request('/v1/chat/completions',conversation))
assert recalled['choices'][0]['message']['content'].strip()=='17'
report={'passed':True,'checks':['authenticated model listing','bad credentials rejected','browser Origin rejected','unexpected Host rejected','unsupported tools/options rejected','real JSON completion','SSE matches JSON','chunked prefill beyond 512 tokens','conversation history preserved'], 'arithmetic':plain,'long_context':long_result,'conversation':recalled,'stream_first_content_seconds':arrival[0]}
path=ROOT/'runs/opencode-api-test.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'long_prompt_tokens':long_result['usage']['prompt_tokens'],'recall':'17','report':str(path)},indent=2))
