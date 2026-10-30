"""Protect E2 analysis from wrong epochs/units and incomplete/reused producers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_multistep import UNIT,check_evaluation,digest
from finish_e2 import completed,wait_for_endpoint,freeze_sources


class EndpointTests(unittest.TestCase):
    def test_snapshot_rejects_changed_source_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);training=root/'run';source=root/'source';training.mkdir();source.mkdir()
            (source/'method.py').write_text('original method')
            (source/'MULTISTEP_PROTOCOL.md').write_text('fixed protocol')
            protocol=dict(source_sha256={'method.py':digest(source/'method.py')},
                          protocol_sha256=digest(source/'MULTISTEP_PROTOCOL.md'))
            (training/'protocol.json').write_text(json.dumps(protocol))
            with patch('finish_e2.SCRIPTS',source):
                (source/'method.py').write_text('changed method')
                with self.assertRaises(AssertionError):freeze_sources(training)
                (source/'method.py').write_text('original method');freeze_sources(training)
                (source/'method.py').write_text('changed method');freeze_sources(training)
            self.assertEqual((training/'source_snapshot/method.py').read_text(),'original method')

    def make_done(self,root):
        done=dict(steps=10000,source_transitions_per_arm=2560000,
                  final={a:dict(step=10000,adam_count=10000,value_digest='value') for a in ['one','multi']})
        for a in ['one','multi']:
            (root/a).mkdir();(root/a/'critic_10000.pkl').write_bytes(b'full state fixture')
        return done

    def test_evaluation_metadata_rejects_wrong_endpoint_and_scale(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);done=self.make_done(root)
            good=dict(checkpoint=str(root/'one'),checkpoint_kind='multistep',mode='one',epoch=10000,
                      training_extra_updates=10000,distance_unit='log-discount',unit=UNIT,
                      checkpoint_sha256=digest(root/'one/critic_10000.pkl'),critic_digest='value')
            check_evaluation(good,'one',root,done,UNIT)
            for key,value in [('epoch',5000),('training_extra_updates',110000),('checkpoint_kind','official'),
                              ('mode','multi'),('unit',1.),('critic_digest','wrong'),('checkpoint_sha256','wrong')]:
                bad=copy.deepcopy(good);bad[key]=value
                with self.assertRaises(AssertionError,msg=key):check_evaluation(bad,'one',root,done,UNIT)

    def test_incomplete_or_wrong_budget_marker_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.assertFalse(completed(root));done=self.make_done(root)
            marker=root/'training_complete.json';marker.write_text(json.dumps(done));self.assertTrue(completed(root))
            done['final']['multi']['adam_count']=5000;marker.write_text(json.dumps(done))
            with self.assertRaises(AssertionError):completed(root)
            done['final']['multi']['adam_count']=10000;marker.write_text(json.dumps(done))
            (root/'multi/critic_10000.pkl').unlink()
            with self.assertRaises(AssertionError):completed(root)

    def test_exited_or_reused_pid_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for signature,pattern in [(None,'exited without'),('different','identity changed')]:
                with patch('finish_e2.process_signature',return_value=signature):
                    with self.assertRaisesRegex(RuntimeError,pattern):
                        wait_for_endpoint(root,123,'original',root,interval=0)

    def test_live_partial_completion_marker_is_rechecked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);done=self.make_done(root);marker=root/'training_complete.json';marker.write_text('{')
            def complete(_):marker.write_text(json.dumps(done))
            with patch('finish_e2.process_signature',return_value='original'),patch('finish_e2.time.sleep',side_effect=complete):
                wait_for_endpoint(root,123,'original',root,interval=0)
            self.assertTrue(completed(root))


if __name__=='__main__':unittest.main()
