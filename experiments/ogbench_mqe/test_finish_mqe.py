"""Fault-injection tests: no restarts, wrong endpoints and launcher-exit race."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finish_mqe import completed,wait_for_endpoint


def make_endpoint(root,step=100000):
    directory=root/'OGBench/group/run';directory.mkdir(parents=True)
    flags=dict(seed=0,train_steps=100000,eval_episodes=50,agent=dict(agent_name='mqe'),restore_path=None)
    (directory/'flags.json').write_text(json.dumps(flags));(directory/'params_100000.pkl').write_bytes(b'state fixture')
    (directory/'eval.csv').write_text(f'evaluation/overall_success,step\n0,{step}\n')
    (root/'runtime.json').write_text(json.dumps(dict(exit_code=0)))
    (root/'TRAINING_COMPLETE').write_text('finished')
    return directory


class FinishMQETests(unittest.TestCase):
    def test_wrong_evaluation_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.assertFalse(completed(root));directory=make_endpoint(root,50000)
            with self.assertRaises(AssertionError):completed(root)
            (directory/'eval.csv').write_text('step\n100000\n')
            self.assertEqual(completed(root),directory)
            (directory/'params_100000.pkl').unlink()
            with self.assertRaises(AssertionError):completed(root)

    def test_failed_main_is_not_a_wait_or_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'runtime.json').write_text(json.dumps(dict(exit_code=1)))
            with self.assertRaisesRegex(RuntimeError,'failed with exit 1'):completed(root)

    def test_exited_and_reused_identities_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch('finish_mqe.process_signature',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'exited without'):
                    wait_for_endpoint(root,1,'producer',2,'launcher',root,interval=0)
            with patch('finish_mqe.process_signature',return_value='reused'):
                with self.assertRaisesRegex(RuntimeError,'identity changed'):
                    wait_for_endpoint(root,1,'producer',2,'launcher',root,interval=0)

    def test_live_launcher_can_finish_after_main_exits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch('finish_mqe.process_signature',side_effect=lambda pid:'launcher' if pid==2 else None),\
                 patch('finish_mqe.time.sleep',side_effect=lambda _:make_endpoint(root)):
                result=wait_for_endpoint(root,1,'producer',2,'launcher',root,interval=0)
            self.assertEqual(result,root/'OGBench/group/run')
            self.assertEqual(json.loads((root/'status.json').read_text())['phase'],'waiting_for_fixed_endpoint')


if __name__=='__main__':unittest.main()
