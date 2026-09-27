'use strict';
const $ = id => document.getElementById(id);
let key = /^[a-f0-9]{64}$/.test(location.hash.slice(1)) ? location.hash.slice(1) : '';
if (key) { sessionStorage.setItem('sangama-ui-key', key); history.replaceState(null, '', '/'); }
key ||= sessionStorage.getItem('sangama-ui-key') || '';
const operation = () => $('operation').value;
let mode = 'local', config = null, report = null, busy = false, pending = false, readyPeers = [];
const notice = message => { $('notice').textContent = message; $('notice').hidden = !message; };
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers:{'x-ui-key':key, ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function setMode(value) {
  if (busy) return;
  mode = value;
  ['local','peer'].forEach(id => {
    const selected = (id === 'local') === (mode === 'local');
    $(id+'-mode').classList.toggle('selected', selected);
    $(id+'-mode').setAttribute('aria-pressed', selected);
  });
  $('peer-a').textContent = mode === 'local' ? 'Auto-assigned localhost' : readyPeers[0] || 'Route not checked';
  $('peer-b').textContent = mode === 'local' ? 'Auto-assigned localhost' : readyPeers[1] || 'Route not checked';
  $('mode-hint').textContent = mode === 'local'
    ? 'Both shard workers run on this computer. The client does not load a full model during generation.'
    : 'Start with --peers and --token-file to connect configured admitted-mesh bridges or private SSH endpoints. Physical locations are not verified by this app.';
  updateButton();
}
function updateButton() {
  $('run').disabled = !config?.model_ready || busy || pending || (mode === 'peers' && !config?.peer_enabled);
  $('run').textContent = busy || pending ? 'Working…' : operation() === 'verify' ? 'Run verification ↗' : 'Generate text ↗';
  $('operation').disabled = busy || pending;
  ['mesh-plan','mesh-allocate'].forEach(id => { $(id).disabled = busy || pending || !config?.peer_enabled; });
  $('decode-note').textContent = operation() === 'verify' ? 'Same prompt, both routes' : 'No local baseline required';
  $('task-note').textContent = operation() === 'verify' ? 'Loads the full baseline first, then checks the split model.' : 'Loads only assigned shards in workers. Remote mode needs only client metadata.';
  $('local-mode').disabled = busy; $('peer-mode').disabled = busy;
}
function rate(id, value) {
  $(id).replaceChildren(document.createTextNode(value == null ? '—' : value.toFixed(1)));
  const unit = document.createElement('small'); unit.textContent = 'tok/s'; $(id).append(unit);
}
function render(data) {
  config = data; busy = data.job.phase === 'running';
  $('model-state').textContent = data.model_ready ? 'Files available' : 'Model not found';
  $('device').textContent = data.device === 'metal' ? 'Apple Metal' : 'CPU backend';
  $('phase').textContent = ({idle:'Ready',running:'Running',complete:'Complete',error:'Failed'})[data.job.phase] || data.job.phase;
  $('output').classList.toggle('running', busy);
  if (['allocate','plan'].includes(data.job.operation)) {
    readyPeers = data.job.plan?.peers || [];
    if (mode === 'peers') { $('peer-a').textContent=readyPeers[0] || 'Route not checked'; $('peer-b').textContent=readyPeers[1] || 'Route not checked'; }
    $('mesh-message').textContent = busy ? (data.job.operation === 'allocate' ? 'Reserving workers, loading prepared shards and checking readiness…' : 'Checking the complete model route…') : data.job.phase === 'error' ? `Operation failed: ${data.job.error}. Check workers and retry; abandoned leases expire.` : `Ready route: ${data.job.plan.peers.join(' → ')}. Inference will reserve and recheck workers.`;
    updateButton(); return;
  }
  if (busy) {
    report = null; $('download').disabled = true;
    $('output').classList.remove('has-text');
    $('output').textContent = data.job.operation === 'verify' ? 'Loading the full baseline, then checking the split route…' : 'Connecting workers and generating through the split model…';
    $('verification').textContent = data.job.operation === 'verify' ? '○ Verification in progress' : '○ Generation in progress · no baseline loaded'; $('verification').className = 'verification';
    rate('split-speed',null); rate('local-speed',null); $('ttft').textContent='—'; $('generated').textContent='—';
  } else if (data.job.phase === 'complete') {
    report = data.job.report; $('download').disabled = false;
    $('output').classList.add('has-text'); $('output').textContent = report.distributed_text || '(No visible text; inspect the report.)';
    rate('split-speed',report.distributed.decode_tokens_per_second); rate('local-speed',report.local?.decode_tokens_per_second);
    $('ttft').textContent = report.distributed.first_token_ms.toFixed(1)+' ms'; $('generated').textContent = report.generated_tokens+' tokens';
    $('verification').className = 'verification '+(report.operation === 'generate' ? '' : report.passed ? 'passed' : 'failed');
    $('verification').textContent = report.operation === 'generate' ? `Generated · ${report.finish_reason === 'eos' ? 'end of response' : 'token limit reached'} · not baseline-verified` : report.passed ? `✓ Tokens match · Max logit error ${report.maximum_logit_absolute_error}` : '✕ Verification mismatch — inspect the report';
  } else if (data.job.phase === 'error') {
    $('output').textContent = data.job.error; $('output').classList.add('has-text');
    $('verification').textContent='✕ Request interrupted. Recover workers, allocate/check the route, then generate again as a new session. Previous KV state is not resumed.'; $('verification').className='verification failed';
  }
  updateButton();
}
$('operation').onchange = updateButton;
$('local-mode').onclick = () => setMode('local'); $('peer-mode').onclick = () => setMode('peers');
$('example').onclick = () => { $('prompt').value='Write a Python function called add that returns the sum of two numbers. Output only the code.'; };
$('run').onclick = async () => {
  pending = true; updateButton(); notice('');
  try { await api('/api/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:$('prompt').value,max_tokens:Number($('tokens').value),mode,operation:operation()})}); await refresh(); }
  catch(error) { notice(error.message); }
  finally { pending=false; updateButton(); }
};
$('download').onclick = () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  const a = document.createElement('a'); a.href=url; a.download='sangama-qwen-report.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
};
async function refresh() {
  try { render(await api('/api/status')); notice(config.model_ready ? '' : 'Prepare the pinned checkpoint first: python3 scripts/fetch-qwen.py'); }
  catch(error) { notice(error.message); config=null; updateButton(); }
}
let meshPollAt = 0;
async function poll() { await refresh(); await refreshDht(); if (Date.now() - meshPollAt > 10000) { await refreshMesh(); meshPollAt = Date.now(); } setTimeout(poll,1500); }

let nodeAddress = '';
async function refreshDht() {
  try {
    const state = await api('/api/dht');
    nodeAddress = state.address || '';
    $('node-address').textContent = nodeAddress || 'Starting…';
    $('dht-state').textContent = `${state.connected_peers || 0} connected · ${state.stored_records || 0} records`;
    if (!$('model-hash').value && config?.manifest_hash) $('model-hash').value = config.manifest_hash;
    $('dht-message').textContent = state.last_error || (state.search === 'searching' ? 'Looking up providers and verifying their signed records…' : state.search === 'complete' ? `Search complete: ${state.discoveries.length} verified provider(s). Discovery does not authorize inference.` : state.published ? 'Shard advertisement replicated. Signed records expire in five minutes and renew while this node runs.' : state.offer ? 'Advertisement saved locally; waiting for replication to another node.' : 'Join another node to replicate records. Your private identity is stored only on this device.');
    $('providers').replaceChildren();
    for (const peer of state.discoveries || []) {
      const row = document.createElement('div'); row.className = 'provider-row';
      const title = document.createElement('strong'); title.textContent = `Verified signature · layers ${peer.start}–${peer.end-1} · approval required`;
      const id = document.createElement('code'); id.textContent = peer.peer_id;
      const address = document.createElement('code'); address.textContent = peer.addresses.join(' · ');
      row.append(title,id,address); $('providers').append(row);
    }
  } catch(error) { $('dht-message').textContent=error.message; }
}
async function dhtAction(button, payload) {
  button.disabled = true;
  try { await api('/api/dht',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)}); await refreshDht(); }
  catch(error) { $('dht-message').textContent=error.message; }
  finally { button.disabled=false; }
}
$('copy-node').onclick=async()=>{try {await navigator.clipboard.writeText(nodeAddress);$('copy-node').textContent='Copied';setTimeout(()=>$('copy-node').textContent='Copy',1800);}catch{ $('dht-message').textContent='Select and copy the node address above.'; }};
$('join-network').onclick=()=>dhtAction($('join-network'),{action:'join',address:$('bootstrap').value.trim()});
$('publish-shard').onclick=()=>{const [start,end]=$('shard-range').value.split(',').map(Number);dhtAction($('publish-shard'),{action:'publish',start,end});};
$('find-providers').onclick=()=>dhtAction($('find-providers'),{action:'find',model_hash:$('model-hash').value.trim()});

poll();

async function refreshMesh() {
  try {
    const state = await api('/api/mesh');
    $('mesh-workers').replaceChildren();
    for (const worker of state.workers) {
      const row = document.createElement('tr');
      const capacity = worker.capacity, info = worker.info;
      const values = [
        worker.address + ((capacity?.peer || worker.mesh?.peer) ? ` / ${capacity?.peer || worker.mesh.peer}` : ''),
        info ? (info.busy ? 'Loaded · busy' : 'Loaded · responds; route check required') : capacity ? (capacity.busy ? 'Placement busy' : 'Reachable · unloaded') : 'Unavailable / access denied',
        capacity ? `${(capacity.budget_bytes / 1073741824).toFixed(2)} GiB budget / ${(capacity.available_bytes / 1073741824).toFixed(2)} GiB available` : info?.memory ? `${(info.memory.budget_bytes / 1073741824).toFixed(2)} GiB at load` : 'Not reported',
        info ? `${info.shard.index}: layers ${info.shard.start}–${info.shard.end-1}` : capacity ? `Prepared: ${capacity.shards.map(s => s.index).join(', ')}` : 'Unknown',
        `${worker.probe_ms.toFixed(1)} ms (combined checks)`, worker.mesh ? `${worker.mesh.admitted ? 'Admitted' : 'Not admitted / snapshot expired'}; ${worker.mesh.paths.join(' + ') || 'no active connection'}; membership expires ${worker.mesh.member_expires ? new Date(worker.mesh.member_expires*1000).toLocaleString() : 'unknown'}` : 'Path and membership not reported by bridge'
      ];
      for (const value of values) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      $('mesh-workers').append(row);
    }
    $('mesh-state').textContent = `${state.workers.filter(w => w.reachable).length}/${state.workers.length} reachable · checked ${new Date(state.checked_at*1000).toLocaleTimeString()}`;
    if (!state.workers.length) $('mesh-message').textContent = 'No candidate bridges configured. Restart the local UI with --peers and --token-file to manage remote workers.';
  } catch(error) { $('mesh-state').textContent='Status unavailable'; $('mesh-message').textContent=error.message; }
}
$('mesh-refresh').onclick = refreshMesh;
for (const action of ['plan','allocate']) {
  $('mesh-'+action).onclick = async () => {
    pending=true; updateButton();
    try { await api('/api/mesh',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({action})}); await refresh(); }
    catch(error) { $('mesh-message').textContent=error.message; }
    finally { pending=false; updateButton(); }
  };
}
