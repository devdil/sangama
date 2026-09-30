#!/bin/sh
# run on relay box: [SANGAMA_MTP=K] bench.sh NAME MAXTOK PROMPT
cd /root/run
/root/sangama-mesh --token-file /root/mesh/client.token generate --model-dir model --peers 127.0.0.1:7901,127.0.0.1:7902,127.0.0.1:7903 --prompt "$3" --max-tokens "$2" --output /root/$1.json > /root/$1.log 2>&1
python3 -c "
import json;d=json.load(open(\"/root/$1.json\"));b=d[\"breakdown\"];x=d[\"distributed\"];s=d.get(\"speculation\") or {};n=d[\"generated_tokens\"];st=s.get(\"steps\") or n
print(\"$1\", d[\"prompt_tokens\"],\"->\",n, d[\"finish_reason\"], \"ttft_ms\",round(x[\"first_token_ms\"]),\"tok/s\",round(x[\"decode_tokens_per_second\"],2),\"ms/pass\",round(n/x[\"decode_tokens_per_second\"]/st*1000,1),\"passes\",st,\"accepted\",s.get(\"accepted\"),\"/\",s.get(\"drafted\"),\"| per token p50: total\",round(b[\"decode_ms_p50\"],1),\"compute\",round(b[\"decode_compute_ms_p50\"],1),\"wait\",round(b.get(\"decode_wait_ms_p50\",0),1),\"transit\",round(b.get(\"decode_transit_ms_p50\",0),1))
print(\"  \", d[\"distributed_text\"][:160].replace(chr(10),\" \"))" 2>&1 || tail -3 /root/$1.log
