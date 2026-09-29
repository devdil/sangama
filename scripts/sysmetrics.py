#!/usr/bin/env python3
"""Record system metrics on a Linux worker for later analysis (standard library only).

  record     sample once per second into DIR/metrics.jsonl until stopped: CPU, load, memory,
             network and disk bytes, and per-GPU utilisation, memory, power, temperature and
             clocks (from nvidia-smi when present)
  event      append a timestamped phase marker to DIR/events.jsonl (e.g. download-start)
  summarize  per-phase averages, 95th percentiles, peaks and byte totals as JSON

Phases run from one `NAME-start` event to the matching `NAME-end`.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

GPU_FIELDS = ['index', 'name', 'utilization.gpu', 'utilization.memory', 'memory.used', 'memory.total',
              'power.draw', 'temperature.gpu', 'clocks.sm']


def cpu_times():
    with open('/proc/stat') as f:
        parts = [int(v) for v in f.readline().split()[1:]]
    idle = parts[3] + parts[4]
    return sum(parts), idle


def meminfo():
    values = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, rest = line.split(':', 1)
            values[key] = int(rest.split()[0]) * 1024
    return values


def net_bytes():
    rx = tx = 0
    with open('/proc/net/dev') as f:
        for line in f.readlines()[2:]:
            name, data = line.split(':', 1)
            if name.strip() == 'lo':
                continue
            fields = data.split()
            rx += int(fields[0])
            tx += int(fields[8])
    return rx, tx


def disk_bytes():
    read = written = 0
    with open('/proc/diskstats') as f:
        for line in f:
            fields = line.split()
            name = fields[2]
            # Whole devices only, so partitions are not counted twice.
            if name.startswith(('loop', 'ram')) or name[-1].isdigit() and not name.startswith('nvme'):
                continue
            if name.startswith('nvme') and 'p' in name[4:]:
                continue
            read += int(fields[5]) * 512
            written += int(fields[9]) * 512
    return read, written


def cgroup_memory():
    for path in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            return int(Path(path).read_text())
        except (OSError, ValueError):
            continue
    return None


def gpus():
    if not shutil.which('nvidia-smi'):
        return []
    out = subprocess.run(['nvidia-smi', f"--query-gpu={','.join(GPU_FIELDS)}", '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=10)
    rows = []
    for line in out.stdout.strip().splitlines():
        values = [v.strip() for v in line.split(',')]
        row = {}
        for key, value in zip(GPU_FIELDS, values):
            try:
                row[key] = float(value) if key != 'name' else value
            except ValueError:
                row[key] = None
        rows.append(row)
    return rows


def record(args):
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {'host': args.label or socket.gethostname(), 'cpus': os.cpu_count(),
            'memory_bytes': meminfo().get('MemTotal'), 'gpus': [g.get('name') for g in gpus()],
            'started': time.time()}
    (out/'machine.json').write_text(json.dumps(info) + '\n')
    prev_cpu, prev_net, prev_disk, prev_t = cpu_times(), net_bytes(), disk_bytes(), time.time()
    with open(out/'metrics.jsonl', 'a') as f:
        while True:
            time.sleep(args.interval)
            now = time.time()
            cpu, net, disk = cpu_times(), net_bytes(), disk_bytes()
            dt = max(now - prev_t, 1e-6)
            busy = (cpu[0] - prev_cpu[0]) - (cpu[1] - prev_cpu[1])
            mem = meminfo()
            sample = {
                't': round(now, 3),
                'cpu_percent': round(100.0 * busy / max(cpu[0] - prev_cpu[0], 1), 1),
                'load1': os.getloadavg()[0],
                'mem_used_bytes': mem['MemTotal'] - mem['MemAvailable'],
                'mem_available_bytes': mem['MemAvailable'],
                'cgroup_mem_bytes': cgroup_memory(),
                'net_rx_bytes_per_s': round((net[0] - prev_net[0]) / dt),
                'net_tx_bytes_per_s': round((net[1] - prev_net[1]) / dt),
                'net_rx_total': net[0], 'net_tx_total': net[1],
                'disk_read_bytes_per_s': round((disk[0] - prev_disk[0]) / dt),
                'disk_write_bytes_per_s': round((disk[1] - prev_disk[1]) / dt),
                'gpus': gpus(),
            }
            f.write(json.dumps(sample) + '\n')
            f.flush()
            prev_cpu, prev_net, prev_disk, prev_t = cpu, net, disk, now


def event(args):
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    entry = {'t': round(time.time(), 3), 'event': args.name}
    if args.detail:
        entry['detail'] = args.detail
    with open(out/'events.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda p: values[round((len(values) - 1) * p)]
    return {'avg': round(sum(values) / len(values), 2), 'p50': pick(0.5), 'p95': pick(0.95), 'max': values[-1]}


def summarize(args):
    out = Path(args.dir)
    samples = [json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (out/'events.jsonl').read_text().splitlines()] if (out/'events.jsonl').exists() else []
    phases = {}
    for e in events:
        name = e['event']
        if name.endswith('-start'):
            phases.setdefault(name[:-6], {})['start'] = e['t']
        elif name.endswith('-end'):
            phases.setdefault(name[:-4], {})['end'] = e['t']
    report = {'machine': json.loads((out/'machine.json').read_text()), 'phases': {}}
    for name, span in phases.items():
        if 'start' not in span or 'end' not in span:
            continue
        window = [s for s in samples if span['start'] <= s['t'] <= span['end']]
        if not window:
            continue
        gpu = [g for s in window for g in s['gpus']]
        report['phases'][name] = {
            'seconds': round(span['end'] - span['start'], 1),
            'cpu_percent': stats([s['cpu_percent'] for s in window]),
            'mem_used_gb': stats([s['mem_used_bytes'] / 1e9 for s in window]),
            'net_rx_mbit_s': stats([s['net_rx_bytes_per_s'] * 8 / 1e6 for s in window]),
            'net_tx_mbit_s': stats([s['net_tx_bytes_per_s'] * 8 / 1e6 for s in window]),
            'net_rx_gb_total': round((window[-1]['net_rx_total'] - window[0]['net_rx_total']) / 1e9, 3),
            'net_tx_gb_total': round((window[-1]['net_tx_total'] - window[0]['net_tx_total']) / 1e9, 3),
            'disk_write_mb_s': stats([s['disk_write_bytes_per_s'] / 1e6 for s in window]),
            'gpu_util_percent': stats([g.get('utilization.gpu') for g in gpu]),
            'gpu_mem_used_mib': stats([g.get('memory.used') for g in gpu]),
            'gpu_power_w': stats([g.get('power.draw') for g in gpu]),
            'gpu_temp_c': stats([g.get('temperature.gpu') for g in gpu]),
        }
    json.dump(report, sys.stdout, indent=1)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('record')
    r.add_argument('--dir', required=True)
    r.add_argument('--interval', type=float, default=1.0)
    r.add_argument('--label')
    r.set_defaults(func=record)
    e = sub.add_parser('event')
    e.add_argument('--dir', required=True)
    e.add_argument('name')
    e.add_argument('--detail')
    e.set_defaults(func=event)
    s = sub.add_parser('summarize')
    s.add_argument('--dir', required=True)
    s.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
