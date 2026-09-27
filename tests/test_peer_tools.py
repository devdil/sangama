import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('peer_tunnel', ROOT / 'scripts/peer-tunnel.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PeerTools(unittest.TestCase):
    def test_tunnel_policy_and_input_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / 'key'
            known = Path(folder) / 'known_hosts'
            key.write_text('test fixture, not an actual key')
            key.chmod(0o600)
            known.write_text('verified-host-key-fixture')
            args = argparse.Namespace(host='100.64.1.2', user='p2ptest', ssh_port=22,
                local_port=7902, remote_port=7902, identity=str(key), known_hosts=str(known))
            cmd = module.command(args)
            for required in ['StrictHostKeyChecking=yes', 'BatchMode=yes',
                             'PasswordAuthentication=no', 'ForwardAgent=no',
                             'ExitOnForwardFailure=yes', '127.0.0.1:7902:127.0.0.1:7902']:
                self.assertIn(required, cmd)
            for field, value in [('host', '8.8.8.8'), ('host', '127.0.0.1'),
                                 ('user', '-oProxyCommand=evil'), ('local_port', 22),
                                 ('remote_port', 0)]:
                changed = argparse.Namespace(**vars(args)); setattr(changed, field, value)
                with self.assertRaises(ValueError): module.command(changed)
            key.chmod(0o644)
            with self.assertRaises(ValueError): module.command(args)

    def test_token_generation_private_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, str(ROOT / 'scripts/new-peer-token.py')]
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            path = Path(folder) / '.secrets/peer.token'
            value = path.read_text().strip()
            self.assertEqual(len(value), 64)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(value, result.stdout + result.stderr)
            self.assertNotEqual(subprocess.run(command, cwd=folder, capture_output=True).returncode, 0)
            self.assertEqual(path.read_text().strip(), value)


if __name__ == '__main__':
    unittest.main()
