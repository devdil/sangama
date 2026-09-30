import json, glob, sys
# leak.py PREFIX: compare each request with its prompt's solo run; flag text from other prompts.
solo = {k: json.load(open(f"/root/solo-c4-{k}.json")) for k in range(4)}
words = {0: ["fibonacci", "canberra", "volunteer"], 1: ["canberra", "pipeline parallelism", "volunteer"],
         2: ["fibonacci", "pipeline parallelism", "volunteer"], 3: ["fibonacci", "canberra", "pipeline parallelism"]}
for prefix in sys.argv[1:]:
    n = same = leaks = short = 0; firsts = []
    for f in sorted(glob.glob(f"/root/{prefix}-[0-9]*.json")):
        i = int(f.split("-")[-1].split(".")[0]); d = json.load(open(f)); ref = solo[i % 4]; n += 1
        a, b = ref["distributed_token_ids"], d["distributed_token_ids"]; m = min(len(a), len(b))
        j = next((j for j in range(m) if a[j] != b[j]), None); same += j is None; firsts.append(j)
        t = d["distributed_text"].lower()
        if any(w in t and w not in ref["distributed_text"].lower() for w in words[i % 4]): leaks += 1
        if d["finish_reason"] == "eos" and d["generated_tokens"] < 40: short += 1
    print(f"{prefix}: requests={n} same_as_solo={same} foreign_text={leaks} ended_under_40_tokens={short} first_diffs={sorted(x for x in firsts if x is not None)[:10]}")
