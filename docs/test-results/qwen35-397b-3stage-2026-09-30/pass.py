import json, sys
for name in sys.argv[1:]:
    d = json.load(open(f"/root/{name}.json")); b = d["breakdown"]; n = d["generated_tokens"]; st = d["speculation"]["steps"] or n
    k = n / st  # tokens per pass: per-token figures times k give per-pass figures
    t = d["last_trace"]
    comp = " ".join(f"{x['forward_ms']:.1f}" for x in t)
    wait = " ".join(f"{x['started_ms'] - x['received_ms']:.2f}" for x in t)
    print(f"{name}: per pass {b['decode_ms_p50']*k:.1f} ms = compute {b['decode_compute_ms_p50']*k:.1f} + wait {b['decode_wait_ms_p50']*k:.1f} + transit {b['decode_transit_ms_p50']*k:.1f} | last pass per stage: compute {comp} | wait {wait}")
