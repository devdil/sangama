#!/bin/sh
# Prepare one worker: metrics recorder, Python deps, then this stage's slices.
set -e
NAME=$1; STAGE=$2
cd /root/sangama
M="python3 scripts/sysmetrics.py"
pgrep -f "sysmetrics.py record" >/dev/null || setsid -f python3 scripts/sysmetrics.py record --dir /root/metrics --label "$NAME" > /root/metrics-recorder.log 2>&1 < /dev/null
$M event --dir /root/metrics setup-start
[ -x /root/py/bin/python ] || python3 -m venv /root/py; /root/py/bin/pip install -q numpy pyyaml tqdm
$M event --dir /root/metrics setup-end
$M event --dir /root/metrics stage-download-start --detail "stage $STAGE"
/root/py/bin/python scripts/sliced-model.py stage --model-dir /root/sangama/model --shard "$STAGE"
$M event --dir /root/metrics stage-download-end --detail "stage $STAGE"
echo PREP_OK
