import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('route_order', ROOT / 'scripts/route-order.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def line(positions):
    """Delays between points on a line, one direction only."""
    names = list(positions)
    return {a: {b: abs(positions[a] - positions[b]) for b in names[i + 1:]} for i, a in enumerate(names)}


def random_instance(seed, size, with_client):
    """Random points in a plane, so delays are symmetric and obey the triangle inequality."""
    rng = random.Random(seed)
    names = [f'n{i}' for i in range(size)] + (['client'] if with_client else [])
    points = {name: (rng.uniform(0, 100), rng.uniform(0, 100)) for name in names}
    delay_ms = {a: {b: ((points[a][0] - points[b][0]) ** 2 + (points[a][1] - points[b][1]) ** 2) ** 0.5
                    for b in names if b != a} for a in names}
    return names[:size], delay_ms, 'client' if with_client else None


def brute_force(nodes, delays, client, fixed_last=None):
    movable = [node for node in nodes if node != fixed_last]
    tail = [] if fixed_last is None else [fixed_last]
    return min(module.cost(list(order) + tail, delays, client) for order in itertools.permutations(movable))


class RouteOrder(unittest.TestCase):
    def run_cli(self, data, *args):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            path.write_text(json.dumps(data))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = module.main([args[0], str(path), *args[1:]])
        return status, json.loads(output.getvalue())

    def test_line_of_points_is_visited_in_position_order(self):
        positions = {'c': 20, 'a': 0, 'e': 40, 'b': 10, 'd': 30}
        delays = module.Delays(line(positions), 1000)
        order, cost = module.best_order(list(positions), delays)
        self.assertIn(order, (list('abcde'), list('edcba')))
        self.assertAlmostEqual(cost, 40)
        self.assertEqual(module.hops(order, delays), [10, 10, 10, 10])

    def test_client_is_the_start_and_end_of_the_path(self):
        positions = {'client': 0, 'c': 30, 'a': 10, 'b': 20}
        status, result = self.run_cli(
            {'client': 'client', 'nodes': ['c', 'a', 'b'], 'delay_ms': line(positions)}, 'order')
        self.assertEqual(status, 0)
        # Out to the far end and back is 60 ms; the input order c, a, b doubles back for 80 ms.
        self.assertIn(result['order'], (['a', 'b', 'c'], ['c', 'b', 'a']))
        self.assertAlmostEqual(result['cost_ms'], 60)
        self.assertAlmostEqual(sum(result['hops_ms']), result['cost_ms'])
        self.assertEqual(len(result['hops_ms']), 4)
        self.assertAlmostEqual(result['baseline_cost_ms'], 80)

    def test_two_opt_uncrosses_a_path(self):
        # Corners of a 10 by 1 rectangle. a-c-b-d crosses the long diagonal twice.
        points = {'a': (0, 0), 'b': (10, 0), 'c': (10, 1), 'd': (0, 1)}
        delay_ms = {x: {y: ((points[x][0] - points[y][0]) ** 2 + (points[x][1] - points[y][1]) ** 2) ** 0.5
                        for y in points if y != x} for x in points}
        delays = module.Delays(delay_ms, 1000)
        crossed = ['a', 'c', 'b', 'd']
        improved, cost = module.improve(crossed, [], delays, None)
        self.assertLess(cost, module.cost(crossed, delays))
        self.assertAlmostEqual(cost, brute_force(list(points), delays, None))
        self.assertAlmostEqual(cost, 12)
        self.assertAlmostEqual(module.cost(improved, delays), cost)

    def test_fixed_last_from_flag_overrides_file(self):
        positions = {'a': 0, 'b': 10, 'c': 20, 'd': 30}
        data = {'client': None, 'nodes': list(positions), 'delay_ms': line(positions), 'fixed_last': 'a'}
        _, from_file = self.run_cli(data, 'order')
        self.assertEqual(from_file['order'], ['d', 'c', 'b', 'a'])
        _, from_flag = self.run_cli(data, 'order', '--fixed-last', 'b')
        self.assertEqual(from_flag['order'][-1], 'b')
        delays = module.Delays(data['delay_ms'], 1000)
        self.assertAlmostEqual(from_flag['cost_ms'], brute_force(data['nodes'], delays, None, 'b'))
        self.assertGreater(from_flag['cost_ms'], from_file['cost_ms'])

    def test_fixed_last_must_be_a_node(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            status = module.main(['order', '/nonexistent/input.json'])
            self.assertEqual(status, 2)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'input.json'
                path.write_text(json.dumps({'nodes': ['a'], 'delay_ms': {}}))
                self.assertEqual(module.main(['order', str(path), '--fixed-last', 'z']), 2)
        self.assertIn("'z'", errors.getvalue())

    def test_one_direction_fills_the_other(self):
        delays = module.Delays({'a': {'b': 12.5}, 'b': {'c': 3}, 'c': {'b': 7}}, 1000)
        self.assertEqual(delays('a', 'b'), 12.5)
        self.assertEqual(delays('b', 'a'), 12.5)
        # When both directions are measured each keeps its own value.
        self.assertEqual(delays('b', 'c'), 3)
        self.assertEqual(delays('c', 'b'), 7)

    def test_unknown_pair_costs_the_penalty(self):
        data = {'nodes': ['a', 'b', 'c'], 'delay_ms': {'a': {'b': 5}}}
        _, default = self.run_cli(data, 'order')
        self.assertAlmostEqual(default['cost_ms'], 1005)
        self.assertEqual(len(default['unknown_hops']), 1)
        _, custom = self.run_cli(data, 'order', '--unknown-ms', '200')
        self.assertAlmostEqual(custom['cost_ms'], 205)
        # A measured but slow pair is preferred to an unmeasured one.
        data['delay_ms']['b'] = {'c': 900}
        _, measured = self.run_cli(data, 'order')
        self.assertEqual(measured['order'], ['a', 'b', 'c'])
        self.assertNotIn('unknown_hops', measured)

    def test_same_input_gives_same_output(self):
        nodes, delay_ms, client = random_instance(7, 8, True)
        delays = module.Delays(delay_ms, 1000)
        first = module.best_order(nodes, delays, client)
        for _ in range(3):
            self.assertEqual(module.best_order(nodes, delays, client), first)
        # All delays equal: every order ties, so the input order must be kept.
        equal = module.Delays({a: {b: 5 for b in 'abcd' if b != a} for a in 'abcd'}, 1000)
        self.assertEqual(module.best_order(list('abcd'), equal)[0], list('abcd'))

    def seeded_instances(self, seeds):
        for seed in seeds:
            for size in (2, 5, 7, 8):
                for with_client in (False, True):
                    nodes, delay_ms, client = random_instance(seed * 100 + size, size, with_client)
                    delays = module.Delays(delay_ms, 1000)
                    for fixed_last in (None, nodes[0]):
                        with self.subTest(seed=seed, size=size, client=client, fixed_last=fixed_last):
                            order, cost = module.best_order(nodes, delays, client, fixed_last)
                            self.assertEqual(sorted(order), sorted(nodes))
                            if fixed_last:
                                self.assertEqual(order[-1], fixed_last)
                            self.assertAlmostEqual(cost, module.cost(order, delays, client))
                            yield cost, brute_force(nodes, delays, client, fixed_last)

    def test_matches_brute_force_on_seeded_instances(self):
        for cost, optimum in self.seeded_instances(range(12)):
            self.assertAlmostEqual(cost, optimum)

    def test_stays_close_to_brute_force_when_not_optimal(self):
        # The search is a heuristic. Seed 18 (7 nodes, client, fixed last) is a known miss, 0.9%
        # above the optimum, so these seeds assert a bound instead of equality.
        misses = 0
        for cost, optimum in self.seeded_instances(range(12, 24)):
            self.assertLessEqual(cost, optimum * 1.02)
            misses += cost > optimum + 1e-6
        self.assertLessEqual(misses, 1)

    def test_check_exit_status(self):
        now = 1790000000
        data = {'nodes': ['a', 'b', 'c'], 'ends_at': {'a': now + 10 * 3600, 'b': now + 30 * 3600}}
        status, result = self.run_cli(data, 'check', '--now', str(now))
        self.assertEqual(status, 1)
        self.assertEqual([entry['node'] for entry in result['ending_soon']], ['a'])
        self.assertAlmostEqual(result['ending_soon'][0]['hours_left'], 10)
        self.assertEqual(result['no_end_date'], ['c'])
        status, result = self.run_cli(data, 'check', '--now', str(now), '--min-hours', '5')
        self.assertEqual((status, result['ending_soon']), (0, []))
        status, result = self.run_cli(data, 'check', '--now', str(now), '--min-hours', '48')
        self.assertEqual(status, 1)
        self.assertEqual([entry['node'] for entry in result['ending_soon']], ['a', 'b'])
        # A rental that has already ended is reported too.
        status, result = self.run_cli(data, 'check', '--now', str(now + 11 * 3600), '--min-hours', '1')
        self.assertEqual((status, result['ending_soon'][0]['node']), (1, 'a'))

    def test_relay_matrix_output(self):
        rtt = {'a': 17.0, 'b': 30.0, 'c': 5.0, 'd': 44.0}
        status, matrix = self.run_cli({'rtt_ms': rtt, 'client_rtt_ms': 1.0}, 'relay-matrix')
        self.assertEqual(status, 0)
        self.assertEqual(matrix['client'], 'client')
        self.assertEqual(matrix['nodes'], ['a', 'b', 'c', 'd'])
        self.assertAlmostEqual(matrix['delay_ms']['a']['b'], 23.5)
        self.assertAlmostEqual(matrix['delay_ms']['b']['a'], 23.5)
        self.assertAlmostEqual(matrix['delay_ms']['client']['c'], 3.0)
        # The output is valid input for `order`.
        status, result = self.run_cli(matrix, 'order')
        self.assertEqual(status, 0)
        self.assertNotIn('unknown_hops', result)
        self.assertAlmostEqual(result['cost_ms'], result['baseline_cost_ms'])

    def test_relay_only_mesh_makes_every_order_cost_the_same(self):
        # Through a relay each node's round trip is paid once whatever the order, so ordering
        # only starts to matter when peers connect to each other directly.
        rtt = {'a': 17.0, 'b': 30.0, 'c': 5.0, 'd': 44.0}
        matrix = module.relay_matrix(rtt, 1.0)
        delays = module.Delays(matrix['delay_ms'], 1000)
        for order in itertools.permutations(rtt):
            self.assertAlmostEqual(module.cost(order, delays, matrix['client']), 1.0 + sum(rtt.values()))
        # Without a client the two ends are paid half, and nothing else depends on the order.
        matrix = module.relay_matrix(rtt)
        self.assertIsNone(matrix['client'])
        delays = module.Delays(matrix['delay_ms'], 1000)
        for order in itertools.permutations(rtt):
            ends = (rtt[order[0]] + rtt[order[-1]]) / 2
            self.assertAlmostEqual(module.cost(order, delays) + ends, sum(rtt.values()))


if __name__ == '__main__':
    unittest.main()
