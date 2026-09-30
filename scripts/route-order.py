#!/usr/bin/env python3
"""Choose which machine runs which pipeline stage, and flag rentals that end soon.

Each token passes through stage 0, 1, ..., N-1 and returns to the client, so the time per token
is mostly the sum of the network delays along that path. `order` finds a short path from measured
delays, `check` lists machines whose rental ends soon, and `relay-matrix` builds the delay matrix
for a mesh where every peer talks through one relay.

Input file for `order` and `check` (JSON):
  {"client": "laptop" or null,
   "nodes": ["a", "b", "c"],
   "delay_ms": {"laptop": {"a": 4.0}, "a": {"b": 12.3, "c": 40.1}},
   "fixed_last": "c",
   "ends_at": {"a": 1790750000}}

delay_ms[x][y] is the one-way delay from x to y in milliseconds. If only one direction is given it
is used for both. A missing pair is unknown and costs --unknown-ms. ends_at is in unix seconds.

Input file for `relay-matrix` (JSON): {"rtt_ms": {"a": 17.1, "b": 30.0}, "client_rtt_ms": 1.0}

Examples:
  python3 scripts/route-order.py order fleet.json --fixed-last big-memory-box
  python3 scripts/route-order.py check fleet.json --min-hours 48
  python3 scripts/route-order.py relay-matrix rtt.json > fleet.json
"""
import argparse
import json
import sys
import time

EPSILON = 1e-9  # ignore gains smaller than this so float noise cannot cause endless swapping
RELAY_CLIENT = 'client'


class Delays:
    """Delay lookup that fills in the reverse direction and penalises unknown pairs."""

    def __init__(self, delay_ms, unknown_ms):
        self.delay_ms = delay_ms
        self.unknown_ms = unknown_ms

    def known(self, source, target):
        for a, b in ((source, target), (target, source)):
            value = self.delay_ms.get(a, {}).get(b)
            if value is not None:
                return float(value)
        return None

    def __call__(self, source, target):
        value = self.known(source, target)
        return self.unknown_ms if value is None else value


def hops(order, delays, client=None):
    """Delay of each hop: client to first, between consecutive nodes, last to client."""
    path = list(order) if client is None else [client, *order, client]
    return [delays(a, b) for a, b in zip(path, path[1:])]


def cost(order, delays, client=None):
    return sum(hops(order, delays, client))


def nearest_neighbour(start, nodes, delays):
    """Greedy path from `start`; ties go to the node listed first in the input."""
    order, remaining = [start], [node for node in nodes if node != start]
    while remaining:
        nearest = min(remaining, key=lambda node: delays(order[-1], node))
        remaining.remove(nearest)
        order.append(nearest)
    return order


def neighbours(order):
    """Every order one 2-opt reversal or one single-node move away, in a fixed sequence."""
    size = len(order)
    for i in range(size - 1):
        for j in range(i + 1, size):
            yield order[:i] + order[i:j + 1][::-1] + order[j + 1:]
    for i in range(size):
        rest = order[:i] + order[i + 1:]
        for j in range(size):
            if j != i:
                yield rest[:j] + [order[i]] + rest[j:]


def improve(order, tail, delays, client):
    """Apply the best available move until none shortens the path. `tail` stays at the end."""
    def total(candidate):
        # The whole path is re-costed because delays may differ by direction, so a reversal
        # changes more than its two end hops. Fleets are tens of nodes at most.
        return cost(candidate + tail, delays, client)

    best = total(order)
    while True:
        # Taking the best move rather than the first improving one reaches the optimum far
        # more often on small fleets; min() keeps the earliest candidate on ties.
        candidate = min(neighbours(order), key=total, default=order)
        candidate_cost = total(candidate)
        if candidate_cost >= best - EPSILON:
            return order, best
        order, best = candidate, candidate_cost


def best_order(nodes, delays, client=None, fixed_last=None):
    """Shortest order found from every nearest-neighbour start, improved by 2-opt and or-opt."""
    tail = [] if fixed_last is None else [fixed_last]
    movable = [node for node in nodes if node != fixed_last]
    if not movable:
        return tail, cost(tail, delays, client)
    starts = [nearest_neighbour(start, movable, delays) for start in movable]
    if tail:
        # Forward construction ignores the fixed end, so also grow one path back from it.
        starts.append(nearest_neighbour(fixed_last, nodes, delays)[:0:-1])
    best, best_cost = None, None
    for start in starts:
        order, order_cost = improve(start, tail, delays, client)
        if best is None or order_cost < best_cost - EPSILON:
            best, best_cost = order, order_cost
    return best + tail, best_cost


def ending_soon(ends_at, now, min_hours):
    """Nodes whose rental ends within `min_hours` of `now`, soonest first."""
    soon = [(end, node) for node, end in ends_at.items() if end - now < min_hours * 3600]
    return [{'node': node, 'ends_at': end, 'hours_left': round((end - now) / 3600, 2)}
            for end, node in sorted(soon)]


def relay_matrix(rtt_ms, client_rtt_ms=None):
    """Delay matrix for a relay-only mesh, where x to y costs half of each round trip."""
    rtt = dict(rtt_ms)
    client = None
    if client_rtt_ms is not None:
        client = RELAY_CLIENT
        if client in rtt:
            raise ValueError(f'a node may not be named {client!r} when client_rtt_ms is set')
        rtt[client] = client_rtt_ms
    delay_ms = {a: {b: (rtt[a] + rtt[b]) / 2 for b in rtt if b != a} for a in rtt}
    return {'client': client, 'nodes': list(rtt_ms), 'delay_ms': delay_ms}


def load(path):
    if path == '-':
        return json.load(sys.stdin)
    with open(path) as file:
        return json.load(file)


def run_order(args):
    data = load(args.file)
    nodes = data.get('nodes') or []
    client = data.get('client')
    fixed_last = args.fixed_last or data.get('fixed_last')
    if len(set(nodes)) != len(nodes):
        raise ValueError('nodes contains a name more than once')
    if client is not None and client in nodes:
        raise ValueError(f'client {client!r} is also listed in nodes')
    if fixed_last is not None and fixed_last not in nodes:
        raise ValueError(f'fixed last node {fixed_last!r} is not in nodes')
    delays = Delays(data.get('delay_ms') or {}, args.unknown_ms)
    order, order_cost = best_order(nodes, delays, client, fixed_last)
    path = order if client is None else [client, *order, client]
    unknown = [[a, b] for a, b in zip(path, path[1:]) if delays.known(a, b) is None]
    result = {
        'order': order,
        'cost_ms': round(order_cost, 3),
        'hops_ms': [round(hop, 3) for hop in hops(order, delays, client)],
        'baseline_cost_ms': round(cost(nodes, delays, client), 3),
    }
    if unknown:
        # Without this a path through unmeasured pairs would look like a real measurement.
        result['unknown_hops'] = unknown
    print(json.dumps(result, indent=2))
    return 0


def run_check(args):
    data = load(args.file)
    now = time.time() if args.now is None else args.now
    ends_at = data.get('ends_at') or {}
    soon = ending_soon(ends_at, now, args.min_hours)
    result = {
        'min_hours': args.min_hours,
        'ending_soon': soon,
        'no_end_date': [node for node in data.get('nodes') or [] if node not in ends_at],
    }
    print(json.dumps(result, indent=2))
    return 1 if soon else 0


def run_relay_matrix(args):
    data = load(args.file)
    print(json.dumps(relay_matrix(data['rtt_ms'], data.get('client_rtt_ms')), indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    order = commands.add_parser('order', help='find a short order of all nodes')
    order.add_argument('file', help='fleet JSON file, or - for standard input')
    order.add_argument('--fixed-last', metavar='NAME',
                       help='node that must run the last stage; overrides fixed_last in the file')
    order.add_argument('--unknown-ms', type=float, default=1000.0,
                       help='delay assumed for a pair with no measurement (default 1000)')
    order.set_defaults(run=run_order)

    check = commands.add_parser('check', help='list nodes whose rental ends soon; exit 1 if any')
    check.add_argument('file', help='fleet JSON file, or - for standard input')
    check.add_argument('--min-hours', type=float, default=24.0,
                       help='hours of rental a node must have left (default 24)')
    check.add_argument('--now', type=float, help='unix seconds to use instead of the current time')
    check.set_defaults(run=run_check)

    relay = commands.add_parser('relay-matrix', help='build the delay matrix for a relay-only mesh')
    relay.add_argument('file', help='relay round-trip JSON file, or - for standard input')
    relay.set_defaults(run=run_relay_matrix)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError) as error:
        print(f'route-order: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
