"""Fault-injection tests for endpoint-only continuation and lost producer identity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finish_e1 import completed,wait_for_endpoint


class FinishTests(unittest.TestCase):
    def test_exited_producer_without_endpoint_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch('finish_e1.process_signature',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'exited without a verified'):
                    wait_for_endpoint(root,123,'original',root,interval=0)
            self.assertFalse((root/'status.json').exists())

    def test_reused_pid_is_not_treated_as_original_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch('finish_e1.process_signature',return_value='new process identity'):
                with self.assertRaisesRegex(RuntimeError,'identity changed'):
                    wait_for_endpoint(root,123,'original',root,interval=0)

    def test_wrong_budget_or_missing_checkpoint_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'training_complete.json'
            p.write_text(json.dumps(dict(steps=5000)))
            with self.assertRaises(AssertionError):completed(root)
            p.write_text(json.dumps(dict(steps=10000,final={a:dict(adam_count=110000) for a in ['pair','goal']})))
            with self.assertRaises(AssertionError):completed(root)

    def test_partial_marker_from_live_writer_is_rechecked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'training_complete.json';p.write_text('{')
            for arm in ['pair','goal']:
                (root/arm).mkdir();(root/arm/'params_110000.pkl').touch()
            def finish_write(_):
                p.write_text(json.dumps(dict(steps=10000,final={a:dict(adam_count=110000) for a in ['pair','goal']})))
            with patch('finish_e1.process_signature',return_value='original'),patch('finish_e1.time.sleep',side_effect=finish_write):
                wait_for_endpoint(root,123,'original',root,interval=0)
            self.assertTrue(completed(root))
            self.assertEqual(json.loads((root/'status.json').read_text())['phase'],'waiting_for_fixed_endpoint')


if __name__=='__main__':unittest.main()
