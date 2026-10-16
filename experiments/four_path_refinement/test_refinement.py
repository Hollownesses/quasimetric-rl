import unittest
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.four_path_refinement.run import (
    load_data, restore, advance, optimizer_digest, diagnose, learning_rate,
    training_objective, tensor_digest,
)
from experiments.four_path_refinement.repair import accept_candidate, apply_delta, actual, local_linearization, qp_direction, stopping_reason


class RestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Serial reductions make the equation-equivalence regression reproducible.
        torch.set_num_threads(1)
        cls.data = load_data()

    def test_restart_and_exact_original_update(self):
        model, solver, opt, saved = restore(self.data)
        other, dual, adam, _ = restore(self.data)
        self.assertEqual(tensor_digest(*model.state_dict().values()), tensor_digest(*saved['model'].values()))
        self.assertEqual(optimizer_digest(opt), optimizer_digest(adam))
        self.assertEqual(len(opt.param_groups), 1)
        self.assertTrue(all(int(s['step']) == 20000 for s in opt.state.values()))
        loss, h, _ = training_objective(other,self.data,dual,'phr')
        adam.zero_grad(set_to_none=True); loss.backward(); adam.step(); dual.update(h.detach())
        advance(model,self.data,solver,opt,'A',1)
        # CPU/JIT backward reductions can differ by a float32 ulp between calls.
        for a, b in zip(model.state_dict().values(), other.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=3e-8)
        self.assertTrue(torch.equal(solver.multiplier,dual.multiplier))
        for a, b in zip(opt.state.values(),adam.state.values()):
            for key in a:
                torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-7)

    def test_low_lr_preserves_moments_until_first_update(self):
        model, solver, opt, _ = restore(self.data)
        digest = optimizer_digest(opt)
        for group in opt.param_groups: group['lr'] = learning_rate('B',1)
        self.assertEqual(optimizer_digest(opt), digest)
        self.assertAlmostEqual(learning_rate('B',1),2e-6)
        self.assertAlmostEqual(learning_rate('B',5000),2e-7)

    def test_outer_update_uses_current_residual(self):
        model, solver, opt, _ = restore(self.data)
        initial = solver.multiplier.clone()
        advance(model,self.data,solver,opt,'C',1)
        self.assertTrue(torch.equal(initial,solver.multiplier))
        with torch.no_grad(): old = model.distances(self.data)[0]-self.data.cost
        advance(model,self.data,solver,opt,'C',20)
        with torch.no_grad(): new = model.distances(self.data)[0]-self.data.cost
        self.assertTrue(torch.equal(solver.multiplier,initial.clone().add_(new,alpha=20).relu()))
        self.assertGreater(float(((initial+20*old).relu()-solver.multiplier).abs().max()),1e-5)

    def test_full_diagnostics(self):
        model, solver, _, _ = restore(self.data)
        rec, _, arrays = diagnose(model,self.data,solver,0,0.)
        self.assertEqual(len(arrays['residual']),339)
        self.assertEqual(len(rec['terminal_entry_edges']),4)
        self.assertEqual([x['path'] for x in rec['terminal_entry_edges']],[0,1,2,3])
        self.assertAlmostEqual(rec['push'],80.9825668,places=4)
        self.assertFalse(rec['joint_pass'])
        self.assertAlmostEqual(sum(p['positive_residual_sum'] for p in rec['paths']),
                               float(torch.tensor(arrays['residual'])[self.data.physical].relu().sum()),places=5)

    def test_repair_guard_and_infeasible_restoration(self):
        physical = torch.tensor([True,True,False])
        old = torch.tensor([.1,.05,0.])
        better = torch.tensor([.05,.03,0.])
        self.assertTrue(accept_candidate(old,better,80.95,80.9,physical))
        self.assertFalse(accept_candidate(old,better,80.89,80.9,physical))
        self.assertFalse(accept_candidate(old,old*2,81.,80.9,physical))

    def test_reference_free_stop_rules(self):
        physical=torch.tensor([True,True,False])
        h=torch.tensor([-1.,0.,0.])
        self.assertEqual(stopping_reason(h,0.,-.15,physical,0,40,0),'feasible_with_value_guard')
        self.assertIsNone(stopping_reason(h,0.,.1,physical,0,40,0))
        self.assertIsNone(stopping_reason(torch.tensor([.01,0.,0.]),1.,0.,physical,0,40,0))
        self.assertIsNone(stopping_reason(torch.tensor([0.,0.,.01]),1.,0.,physical,0,40,0))
        self.assertEqual(stopping_reason(h,0.,.1,physical,40,40,0),'iteration_budget')
        self.assertEqual(stopping_reason(h,0.,.1,physical,3,40,3),'three_stalled_iterations')

    def test_feasible_but_poor_values_stop_before_reference_evaluation(self):
        from experiments.four_path_refinement import repair
        from experiments.four_path_refinement.run import START
        model,solver,opt,saved=restore(self.data)
        layer=getattr(model.projector.module,'2')
        with torch.no_grad(): layer.weight.zero_();layer.bias.zero_()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'repair'
            args=argparse.Namespace(checkpoint=str(START),output=str(out),iterations=40,delta=.15,push_step=0.)
            calls=[]
            def evaluate_after_save(*args):
                payload=torch.load(out/'latest.pt',weights_only=True)
                self.assertEqual(payload['stop_reason'],'feasible_with_value_guard')
                calls.append(payload['step'])
                return diagnose(*args)
            with patch.object(repair,'load_data',return_value=self.data), \
                 patch.object(repair,'restore',return_value=(model,solver,opt,saved)), \
                 patch.object(repair,'diagnose',side_effect=evaluate_after_save), \
                 patch.object(repair,'local_linearization',side_effect=AssertionError('must stop before QP')):
                repair.run(args)
            final=json.loads((out/'evaluation.json').read_text())
            self.assertEqual(calls,[0])
            self.assertFalse(final['joint_pass'])
            self.assertAlmostEqual(final['push_gap_percent'],100.)
            self.assertEqual(final['stop_reason'],'feasible_with_value_guard')
        torch.set_num_threads(1)

    def test_repair_subspace_is_actual_network_update(self):
        model, _, _, _ = restore(self.data)
        module = model.projector.module
        with torch.no_grad():
            hidden = getattr(module,'1')(getattr(module,'0')(model.encoder(model.nodes)))
            features = torch.cat((hidden,torch.ones(len(hidden),1)),dim=1)
            z0 = model.project()
            layer = getattr(module,'2')
            weight,bias = layer.weight.clone(),layer.bias.clone()
            delta = torch.randn(2048*513)*1e-5
            prediction = z0+features@delta.reshape(2048,513).T
            apply_delta(model,weight,bias,delta,1.)
            torch.testing.assert_close(model.project(),prediction,rtol=2e-6,atol=1e-4)
            actual_norm = torch.cat(((layer.weight-weight).flatten(),(layer.bias-bias).flatten())).norm()
            self.assertAlmostEqual(float(actual_norm),float(delta.norm()),places=6)
            apply_delta(model,weight,bias,delta,0.)
            self.assertTrue(torch.equal(layer.weight,weight))

    def test_full_layer_gram_matches_actual_parameter_jacobian(self):
        model,_,_,_=restore(self.data)
        linearization=local_linearization(model,self.data,80.88)
        A,h,*_=linearization
        self.assertEqual(tuple(A.shape),(340,340))
        layer=getattr(model.projector.module,'2')
        edge,task,pair=model.distances(self.data)
        push=.6*task.mean()+.4*pair.mean()
        targets=[edge[0],edge[80],-push]
        gradients=[]
        for target in targets:
            gw,gb=torch.autograd.grad(target,(layer.weight,layer.bias),retain_graph=True)
            gradients.append(torch.cat((gw,gb[:,None]),dim=1).detach().flatten().double())
        actual_gram=torch.stack(gradients)@torch.stack(gradients).T
        ids=torch.tensor([0,80,339])
        compressed_gram=A[ids]@A[ids].T
        torch.testing.assert_close(compressed_gram,actual_gram,rtol=1e-5,atol=1e-8)
        direction,_,slack=qp_direction(linearization,self.data,push_step=.1)
        actual_slack=(torch.stack(gradients)@direction.double()+h[ids]).relu()
        torch.testing.assert_close(actual_slack,slack[ids],rtol=1e-4,atol=1e-5)


if __name__ == '__main__': unittest.main()
