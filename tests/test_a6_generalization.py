"""CPU checks for the new algorithms and release/checkpoint interface."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.a6_generalization.common import Config
from experiments.a6_generalization import models,optimizers,training
from experiments.a6_generalization.analysis import ensemble_losses

class GeneralizationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_trajectory_text_distinguishes_a_late_rise_from_a_still_falling_curve(self):
        from experiments.a6_generalization.collect import trajectory_text

        falling = [
            {'epochs': 8, 'validation_nll': 1.30},
            {'epochs': 16, 'validation_nll': 1.28},
            {'epochs': 32, 'validation_nll': 1.27},
        ]
        rising = [
            {'epochs': 8, 'validation_nll': 1.30},
            {'epochs': 16, 'validation_nll': 1.26},
            {'epochs': 32, 'validation_nll': 1.29},
        ]
        from experiments.a6_generalization.collect import linear_lr_fraction

        # Epoch 24 of a 32-epoch run is completed step 11256. The multiplier
        # was set from index 11255, which is 25% of the peak.
        self.assertAlmostEqual(linear_lr_fraction(11256, 15008, 75), 0.251, places=3)
        self.assertLess(linear_lr_fraction(15008, 15008, 75), 1e-4)
        self.assertIn('best one', trajectory_text(falling))
        self.assertIn('rose after', trajectory_text(rising))
        self.assertIn('epoch 16', trajectory_text(rising))

    def test_diagnostic_text_records_active_parameters_and_executed_depth(self):
        from experiments.a6_generalization.collect import diagnostic_text

        moe = diagnostic_text(dict(
            stored_parameters=96514560, active_parameters=30454272, experts=4, top_k=1,
            routing=dict(load_cv=0.02, entropy=1.2, mean_max_probability=0.9),
        ))
        loop = diagnostic_text(dict(stored_parameters=11426688, executed_depth=8, passes=6))
        self.assertIn('30,454,272', moe)
        self.assertIn('load CV 0.020', moe)
        self.assertIn('Executed depth 8', loop)
        self.assertIn('6 core passes', loop)

    def test_scoring_does_not_redownload_a_run_that_already_has_an_outcome(self):
        from experiments.a6_generalization.collect import pending_downloads

        names = ('a6-moe-lr0.003-s42', 'a6-loop-lr0.001-s43')
        predictions = set(names)
        pending = pending_downloads(names, predictions, {'a6-moe-lr0.003-s42'})
        self.assertEqual(pending, ('a6-loop-lr0.001-s43',))
        self.assertEqual(pending_downloads(names, predictions, set(names)), ())

    def test_scoring_waits_for_diagnostics_after_the_final_history_point(self):
        from experiments.a6_generalization.collect import ready_to_score

        finished = [{'epochs': 32.0, 'validation_nll': 1.27}]
        partial = [{'epochs': 30.0, 'validation_nll': 1.26}]
        self.assertFalse(ready_to_score(finished, 32, False))
        self.assertFalse(ready_to_score(partial, 32, True))
        self.assertTrue(ready_to_score(finished, 32, True))

    def test_prescribed_score_is_the_endpoint_not_the_best_checkpoint(self):
        from experiments.a6_generalization.plot_trained import endpoint_record

        rose = [
            {'epochs': 22, 'validation_nll': 1.2638},
            {'epochs': 32, 'validation_nll': 1.2900},
        ]
        record = endpoint_record(rose, baseline=1.2627138263769448)
        self.assertAlmostEqual(record['delta'], 1.2900 - 1.2627138263769448)
        self.assertFalse(record['endpoint_is_minimum'])
        self.assertEqual(record['best_epoch'], 22)
        self.assertIsNone(endpoint_record(rose[:1], baseline=1.26))

    def test_a_partial_second_seed_is_not_drawn_as_a_finished_curve(self):
        from experiments.a6_generalization.plot_trained import finished_history

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history.json'
            path.write_text(json.dumps([{'epochs': 10, 'validation_nll': 1.28}]))
            self.assertIsNone(finished_history(path, 32))
            path.write_text(json.dumps([{'epochs': 32, 'validation_nll': 1.265}]))
            self.assertEqual(finished_history(path, 32)[-1]['validation_nll'], 1.265)
            self.assertEqual(finished_history(Path(tmp), 32)[-1]['validation_nll'], 1.265)
            self.assertIsNone(finished_history(Path(tmp) / 'missing.json', 32))

    def test_later_configs_do_not_collide_with_wave1_or_write_the_ledger(self):
        from experiments.a6_generalization.orchestrate import (
            dense_depth3,
            dropout_replicas,
            sgd_grid,
            wave1_configs,
        )

        ledger = Path('results/a6_generalization/prediction_ledger.jsonl')
        before = ledger.read_text()
        wave1, _ = wave1_configs()
        later = [dense_depth3(0.003), *sgd_grid(), *dropout_replicas(0.05)]
        names = [cfg.name for cfg in later]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(set(names).isdisjoint(cfg.name for cfg in wave1))
        dense = dense_depth3(0.003)
        self.assertEqual((dense.architecture, dense.depth, dense.width), ('dense', 3, 512))
        self.assertTrue(all(cfg.optimizer == 'sgd' and cfg.wd == 3.2 for cfg in sgd_grid()))
        replicas = dropout_replicas(0.15)
        self.assertTrue(all(cfg.data_seed == 42 and cfg.save_logits for cfg in replicas))
        self.assertEqual([cfg.seed for cfg in replicas], [43, 44, 45, 46])
        self.assertEqual(ledger.read_text(), before)
        from experiments.a6_generalization.orchestrate import wave2_configs

        wave2, notes = wave2_configs()
        wave2_names = [cfg.name for cfg in wave2]
        self.assertEqual(len(wave2_names), len(set(wave2_names)))
        self.assertTrue(set(wave2_names).isdisjoint(cfg.name for cfg in wave1))
        self.assertFalse(any('scratch' in name and '-e16-' in name for name in wave2_names))
        self.assertEqual(ledger.read_text(), before)
        self.assertTrue(all(note['interval'][0] < note['interval'][1] for note in notes))
        from experiments.a6_generalization.orchestrate import wave3_configs

        before_wave3 = ledger.read_text()
        wave3, wave3_notes = wave3_configs()
        wave3_names = [cfg.name for cfg in wave3]
        self.assertEqual(len(wave3_names), 21)
        self.assertEqual(len(wave3_names), len(set(wave3_names)))
        self.assertTrue(set(wave3_names).isdisjoint(cfg.name for cfg in wave1))
        self.assertTrue(set(wave3_names).isdisjoint(wave2_names))
        self.assertEqual(ledger.read_text(), before_wave3)
        self.assertEqual({note['problem'] for note in wave3_notes}, {'P4b', 'P4c', 'P4d'})
        from experiments.a6_generalization.collect import comparison_groups
        from experiments.a6_generalization.orchestrate import wave4_configs

        wave4, wave4_notes = wave4_configs()
        wave4_names = [cfg.name for cfg in wave4]
        self.assertEqual(len(wave4_names), 8)
        self.assertEqual(len(wave4_names), len(set(wave4_names)))
        self.assertTrue(set(wave4_names).isdisjoint(set(wave3_names) | set(wave2_names) | {cfg.name for cfg in wave1}))
        self.assertTrue(all(cfg.seed == 42 and cfg.data_seed == 43 for cfg in wave4))
        self.assertEqual({note['problem'] for note in wave4_notes}, {'P4a-e4-d43', 'P4a-e16-d43'})
        self.assertEqual(ledger.read_text(), before_wave3)
        routed = comparison_groups(wave4_names)
        self.assertEqual(len(routed['P4a-e4-d43']), 4)
        self.assertEqual(len(routed['P4a-e16-d43']), 4)
        self.assertNotIn(wave4_names[0], routed.get('P4a-d43', ()))
        from experiments.a6_generalization.orchestrate import adaptation_configs, sample_scaling_configs

        adaptations = adaptation_configs(3e-4)
        taken = {cfg.name for cfg in wave1} | set(wave2_names)
        self.assertTrue({cfg.name for cfg in adaptations}.isdisjoint(taken))
        self.assertTrue(all(cfg.initialization == 'related' and cfg.epochs == 16 for cfg in adaptations))
        self.assertFalse(any(cfg.adaptation == 'frozen_embedding' for cfg in adaptations))
        self.assertEqual(
            [cfg.adaptation for cfg in adaptations],
            ['full', 'full', 'full', 'toward_initialization', 'frozen_head', 'frozen_body'],
        )
        from experiments.a6_generalization.orchestrate import hyperball_finetune_configs

        hyperball = hyperball_finetune_configs()
        self.assertEqual([cfg.optimizer for cfg in hyperball], ['hyperball', 'hyperball', 'hyperball'])
        self.assertTrue(all(cfg.wd == 0 and cfg.initialization == 'related' and cfg.epochs == 16 for cfg in hyperball))
        self.assertTrue({cfg.name for cfg in hyperball}.isdisjoint(taken | {cfg.name for cfg in adaptations}))
        scaling = sample_scaling_configs({'related': 3e-4, 'scratch': 1e-3})
        self.assertTrue(all(cfg.ntrain < 30000 and cfg.epochs == 16 for cfg in scaling))
        self.assertTrue({cfg.name for cfg in scaling}.isdisjoint(taken))
        self.assertEqual(ledger.read_text(), before)

    def test_every_locked_prediction_is_in_exactly_one_scoring_group(self):
        from experiments.a6_generalization.collect import comparison_groups
        from experiments.a6_generalization.ledger import read_rows

        predictions = [row for row in read_rows() if row.get('record_type') == 'prediction']
        groups = comparison_groups(row['prediction_id'] for row in predictions)
        seen = [name for members in groups.values() for name in members]
        self.assertEqual(sorted(seen), sorted(row['prediction_id'] for row in predictions))
        self.assertEqual(len(seen), len(set(seen)))
        by_id = {row['prediction_id']: row for row in predictions}
        extra = comparison_groups(list(by_id) + [
            'a6-ft-related-n30000-e16-hyperball-lr0.0003-s42',
            'a6-ft-related-n30000-e16-lr0.0003-wd0-s42',
            'a6-ft-related-n1000-e16-lr0.0003-wd3.2-s42',
            'a6-ft-related-n30000-e16-lr0.0003-wd3.2-toward_initialization-s42',
        ])
        self.assertEqual(extra['P4d'], ('a6-ft-related-n30000-e16-hyperball-lr0.0003-s42',))
        self.assertIn('a6-ft-related-n30000-e16-lr0.0003-wd0-s42', extra['P4c'])
        self.assertIn('a6-ft-related-n1000-e16-lr0.0003-wd3.2-s42', extra['P4b'])
        self.assertIn(
            'a6-ft-related-n30000-e16-lr0.0003-wd3.2-toward_initialization-s42',
            extra['P4c'],
        )
        self.assertNotIn('a6-ft-related-n30000-e16-hyperball-lr0.0003-s42', extra.get('P4a-e16', ()))
        for name in seen:
            changed = by_id[name]['changed']
            if '-e1-' in name:
                self.assertEqual(changed.get('epochs'), 1)
            elif '-e4-' in name:
                self.assertEqual(changed.get('epochs'), 4)
            elif '-e16-' in name:
                self.assertEqual(changed.get('epochs'), 16)
            else:
                self.assertNotIn('epochs', changed)

    def test_recorded_outcomes_match_history_endpoints_and_interval_flags(self):
        import json
        from pathlib import Path

        from experiments.a6_generalization.collect import comparison_groups, _family_falsified
        from experiments.a6_generalization.ledger import audit, read_rows

        summary = audit()
        self.assertTrue(summary['ok'], summary['errors'])
        rows = list(read_rows())
        predictions = {row['prediction_id']: row for row in rows if row.get('record_type') == 'prediction'}
        outcomes = {row['prediction_id']: row for row in rows if row.get('record_type') == 'outcome'}
        root = Path('results/a6_generalization/runs')
        for name, outcome in outcomes.items():
            final = float(json.loads((root / name / 'history.json').read_text())[-1]['validation_nll'])
            self.assertAlmostEqual(final, float(outcome['measured_nll']))
            interval = predictions[name]['interval']
            self.assertEqual(bool(interval[0] <= final <= interval[1]), outcome['interval_hit'])
        groups = comparison_groups(predictions)
        for problem, members in groups.items():
            if not all(name in outcomes for name in members):
                continue
            measured = {name: float(outcomes[name]['measured_nll']) for name in members}
            flag = _family_falsified(problem, measured)
            self.assertTrue(all(outcomes[name]['falsified'] is flag for name in members))

    def test_second_seeds_keep_data_order_and_fine_tuning_replicates_keep_the_model_seed(self):
        import json
        from pathlib import Path

        rows = [
            json.loads(line)
            for line in Path('results/a6_generalization/run_manifest.jsonl').read_text().splitlines()
            if line.strip()
        ]
        spawned = {
            row['experiment_id']: row['config']
            for row in rows
            if row.get('record_type') == 'spawn'
        }
        for name, config in spawned.items():
            if name.endswith('-s43'):
                self.assertEqual(config['seed'], 43)
                self.assertEqual(config['data_seed'], 42, name)
            if name.endswith('-d43'):
                self.assertEqual(config['seed'], 42, name)
                self.assertEqual(config['data_seed'], 43, name)

    def test_sgd_uses_coupled_weight_decay_on_linear_weights_only(self):
        config = Config(width=32, depth=2, context=8, lr=0.1, wd=0.5, optimizer='sgd')
        model = models.build(config)
        optimizer = optimizers.build(model, config)
        linear = model.model.layers[0].self_attn.q_proj.weight
        embed = model.model.embed_tokens.weight
        linear.grad = torch.ones_like(linear)
        embed.grad = torch.ones_like(embed)
        before_linear = linear.detach().clone()
        before_embed = embed.detach().clone()
        optimizer.step()
        self.assertTrue(torch.allclose(linear, before_linear - 0.1 * (1 + 0.5 * before_linear)))
        self.assertTrue(torch.allclose(embed, before_embed - 0.1))

    def test_width512_loop_matches_dense_depth3_at_either_pass_count(self):
        base = Config(width=512, depth=8, context=8)
        six = models.build(replace(base, architecture='loop', loop_passes=6))
        two = models.build(replace(base, architecture='loop', loop_passes=2))
        dense = models.build(replace(base, architecture='dense', depth=3))
        stored = sum(p.numel() for p in six.parameters())
        self.assertEqual(stored, 11_426_688)
        self.assertEqual(stored, sum(p.numel() for p in two.parameters()))
        self.assertEqual(stored, sum(p.numel() for p in dense.parameters()))
        self.assertEqual(six.executed_depth, 8)
        self.assertEqual(two.executed_depth, 4)

    def test_loop_stores_exactly_a_dense_depth3_model(self):
        c = Config(width=64, depth=8, context=16)
        loop = models.build(replace(c, architecture='loop'))
        dense = models.build(replace(c, architecture='dense', depth=3))
        self.assertEqual(
            sum(p.numel() for p in loop.parameters()),
            sum(p.numel() for p in dense.parameters()),
        )
        self.assertEqual(loop.executed_depth, 8)

    def test_depth8_training_flops_match_the_release(self):
        from experiments.a5_capacity.p1_depth_width import architecture, training_flops
        import json
        from pathlib import Path

        released = json.loads(
            Path('experiments/a6_generalization/references/parameter_scaling.json').read_text()
        )
        recorded = next(row['training_flops'] for row in released['runs'] if row['depth'] == 8)
        cfg = architecture(8, 512, vocab=19, context=1024)
        tokens = 30_000 * 1024 * 32
        self.assertEqual(training_flops(cfg, tokens=tokens), recorded)
        router = 6 * 8 * 512 * 4 * tokens
        self.assertEqual(router, 96636764160000)
        self.assertLess(router / recorded, 0.001)

    def test_architecture_carryovers_and_attention_only_dropout(self):
        for kind in ('dense','moe','loop'):
            c=Config(architecture=kind,width=64,depth=8,context=16,attention_dropout=.15)
            m=models.build(c);ids=torch.randint(0,19,(2,16));loss=training.nll(m(ids),ids);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(b.self_attn.dropout==.15 and b.dropout==0 for b in m.model.layers))
            if hasattr(m,'finish_step'):m.finish_step()

    def test_frozen_steps_and_initialization_centered_decay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'splits').mkdir()
            (root / 'checkpoints').mkdir()
            rng = np.random.default_rng(7)
            np.save(root / 'splits/target_30k.npy', rng.integers(0, 19, (16, 16), dtype=np.uint8))
            np.save(root / 'splits/validation.npy', rng.integers(0, 19, (4, 16), dtype=np.uint8))
            base = Config(width=64, depth=2, context=16, ntrain=16, batch=8, micro_batch=8, epochs=2, lr=0.01)
            source = models.build(base)
            torch.save({'model': source.state_dict()}, root / 'checkpoints/related.pt')
            body_name = 'model.layers.0.self_attn.q_proj.weight'
            self.assertIn(body_name, source.state_dict())
            training.train(
                replace(base, initialization='related', adaptation='frozen_body', wd=3.2),
                root=root, output=root / 'frozen', device='cpu',
            )
            frozen = torch.load(root / 'frozen/model.pt', weights_only=False)['model']
            torch.testing.assert_close(frozen[body_name], source.state_dict()[body_name], rtol=0, atol=0)
            self.assertFalse(torch.equal(frozen['lm_head.weight'], source.state_dict()['lm_head.weight']))
            training.train(
                replace(base, initialization='related', adaptation='toward_initialization', wd=50),
                root=root, output=root / 'pull', device='cpu',
            )
            training.train(
                replace(base, initialization='related', adaptation='full', wd=0),
                root=root, output=root / 'free', device='cpu',
            )
            pulled = torch.load(root / 'pull/model.pt', weights_only=False)['model'][body_name]
            free = torch.load(root / 'free/model.pt', weights_only=False)['model'][body_name]
            start = source.state_dict()[body_name]
            self.assertLess((pulled - start).norm().item(), (free - start).norm().item())

    def test_dclm_drops_embedding_and_head(self):
        c = Config(width=64, depth=2, context=16, initialization='dclm')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoints').mkdir()
            donor = models.build(Config(width=64, depth=2, context=16, seed=1))
            state = donor.state_dict()
            state['model.embed_tokens.weight'] = torch.ones_like(state['model.embed_tokens.weight'])
            state['lm_head.weight'] = torch.ones_like(state['lm_head.weight'])
            torch.save({'model': state}, root / 'checkpoints/dclm.pt')
            torch.manual_seed(c.seed)
            model = models.build(c, root)
            self.assertFalse(torch.equal(model.model.embed_tokens.weight, torch.ones_like(model.model.embed_tokens.weight)))
            self.assertFalse(torch.equal(model.lm_head.weight, torch.ones_like(model.lm_head.weight)))
            for name, parameter in model.named_parameters():
                if name not in ('model.embed_tokens.weight', 'lm_head.weight'):
                    torch.testing.assert_close(parameter, donor.state_dict()[name])

    def test_fine_tuning_freezes_and_transfer(self):
        c=Config(width=64,depth=2,context=16)
        original=models.build(c)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir()
            torch.save({'model':original.state_dict()},root/'checkpoints/related.pt')
            for mode in ('full','toward_initialization','frozen_head','frozen_body'):
                m=models.build(replace(c,initialization='related',adaptation=mode),root)
                names={n for n,p in m.named_parameters() if p.requires_grad}
                if mode=='frozen_head':self.assertNotIn('lm_head.weight',names);self.assertIn('model.embed_tokens.weight',names)
                elif mode=='frozen_body':self.assertEqual(names,{'model.embed_tokens.weight','lm_head.weight'})
                else:self.assertEqual(len(names),len(list(m.parameters())))
                for n,p in m.named_parameters():torch.testing.assert_close(p,original.state_dict()[n])

    def test_hyperball_preserves_linear_norms_and_updates_embedding(self):
        m=models.build(Config(width=64,depth=2,context=16))
        weights=[x.weight for x in m.modules() if isinstance(x,torch.nn.Linear)]
        radii=[p.norm().item() for p in weights];embedding=m.model.embed_tokens.weight.detach().clone()
        ids=torch.randint(0,19,(2,16));training.nll(m(ids),ids).backward()
        optimizers.Hyperball(m,.03).step()
        np.testing.assert_allclose([p.norm().item() for p in weights],radii,rtol=2e-6)
        self.assertFalse(torch.equal(embedding,m.model.embed_tokens.weight))

    def test_hyperball_finetune_freezes_the_loaded_norm_and_can_still_rotate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'splits').mkdir()
            (root / 'checkpoints').mkdir()
            rng = np.random.default_rng(7)
            np.save(root / 'splits/target_30k.npy', rng.integers(0, 19, (16, 16), dtype=np.uint8))
            np.save(root / 'splits/validation.npy', rng.integers(0, 19, (4, 16), dtype=np.uint8))
            base = Config(width=64, depth=2, context=16, ntrain=16, batch=8, micro_batch=8, epochs=2, lr=0.01)
            source = models.build(replace(base, seed=7))
            torch.save({'model': source.state_dict()}, root / 'checkpoints/related.pt')
            body_name = 'model.layers.0.self_attn.q_proj.weight'
            training.train(
                replace(
                    base, initialization='related', optimizer='hyperball', wd=0.0, seed=99,
                ),
                root=root, output=root / 'sphere', device='cpu',
            )
            trained = torch.load(root / 'sphere/model.pt', weights_only=False)['model']
            start = source.state_dict()
            for name, tensor in trained.items():
                module_name = name.rsplit('.', 1)[0]
                parameter_name = name.rsplit('.', 1)[1]
                if parameter_name != 'weight':
                    continue
                module = dict(source.named_modules())[module_name]
                if not isinstance(module, torch.nn.Linear):
                    continue
                np.testing.assert_allclose(tensor.norm().item(), start[name].norm().item(), rtol=2e-6)
            self.assertFalse(torch.equal(trained[body_name], start[body_name]))
            self.assertGreater(
                (trained['model.embed_tokens.weight'] - start['model.embed_tokens.weight']).norm().item(),
                0,
            )

    def test_sam_zero_radius_matches_adamw_with_microbatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'splits').mkdir()
            rng=np.random.default_rng(4)
            np.save(root/'splits/target_30k.npy',rng.integers(0,19,(16,16),dtype=np.uint8))
            np.save(root/'splits/validation.npy',rng.integers(0,19,(4,16),dtype=np.uint8))
            base=Config(width=64,depth=2,context=16,ntrain=16,batch=8,micro_batch=4,epochs=1,attention_dropout=.15)
            for mode in ('adamw','sam'):
                training.train(replace(base,optimizer=mode,rho=0),root=root,output=root/mode,device='cpu')
            states=[torch.load(root/m/'model.pt',weights_only=False)['model'] for m in ('adamw','sam')]
            for n in states[0]:torch.testing.assert_close(states[0][n],states[1][n],rtol=1e-6,atol=1e-7)

    def test_sharpness_probe_restores_weights_and_mode(self):
        from experiments.a6_generalization.p3_optimization import directional_sharpness
        m=models.build(Config(width=64,depth=2,context=16)).train()
        saved={n:p.detach().clone() for n,p in m.named_parameters()}
        result=directional_sharpness(m,torch.randint(0,19,(2,16)))
        self.assertTrue(np.isfinite(result['max_loss_increase']))
        self.assertTrue(m.training)
        for n,p in m.named_parameters():torch.testing.assert_close(p,saved[n],rtol=0,atol=0)

    def test_equivalent_samples_stay_inside_the_measured_curve(self):
        from experiments.a6_generalization.analysis import equivalent_samples

        counts = [1000, 3000, 10000, 30000]
        losses = [1.40, 1.30, 1.20, 1.15]
        inside = equivalent_samples(counts, losses, 1.25)
        self.assertEqual(inside['status'], 'inside')
        self.assertEqual(inside['bracket'], [3000.0, 10000.0])
        self.assertAlmostEqual(inside['interpolated'], (3000 * 10000) ** 0.5, places=6)
        self.assertEqual(equivalent_samples(counts, losses, 1.10)['status'], 'better_than_all_scratch')
        self.assertEqual(equivalent_samples(counts, losses, 1.50)['status'], 'worse_than_all_scratch')
        self.assertIsNone(equivalent_samples(counts, [1.2, 1.3, 1.1, 1.0], 1.15)['interpolated'])

    def test_probability_ensembles_singleton_and_identical_members(self):
        z=np.random.default_rng(42).normal(size=(2,4,19));labels=np.array([[1,2,3,4],[4,3,2,1]])
        losses=ensemble_losses([z,z,z],labels)
        np.testing.assert_allclose(losses[0],losses[-1],rtol=1e-12)
        self.assertEqual(losses.shape,(3,2))

    def test_ensemble_averages_probabilities_not_logits(self):
        logits=[np.array([[[8.,0.]]]),np.array([[[0.,2.]]])]
        actual=ensemble_losses(logits,np.array([[0]]))[1,0]
        probability=(1/(1+np.exp(-8))+1/(1+np.exp(2)))/2
        self.assertAlmostEqual(actual,-np.log(probability),places=12)
        self.assertNotAlmostEqual(actual,np.log1p(np.exp(-3)),places=3)

    def test_sliced_ensemble_matches_the_full_probability_average(self):
        from experiments.a6_generalization.analysis import probability_ensemble_curve

        rng = np.random.default_rng(7)
        logits = [rng.normal(size=(5, 3, 4)) for _ in range(3)]
        labels = rng.integers(0, 4, size=(5, 3))
        full = ensemble_losses(logits, labels).mean(-1)
        sliced = probability_ensemble_curve(logits, labels, sequences_per_slice=2)
        np.testing.assert_allclose(sliced, full, rtol=1e-12)
        self.assertEqual(sliced.shape, (3,))

    def test_one_member_matches_token_mean_and_drops_the_leading_token(self):
        from experiments.a6_generalization.analysis import (
            probability_ensemble_curve,
            validation_token_labels,
        )

        rng = np.random.default_rng(3)
        rows = rng.integers(0, 5, size=(4, 7))
        labels = validation_token_labels(rows)
        self.assertEqual(labels.shape, (4, 6))
        np.testing.assert_array_equal(labels, rows[:, 1:])
        logits = rng.normal(size=(4, 6, 5))
        shifted = logits - logits.max(-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(-1, keepdims=True)
        correct = np.take_along_axis(probabilities, labels[..., None], -1)[..., 0]
        token_mean = float(-np.log(np.maximum(correct, 1e-30)).mean())
        curve = probability_ensemble_curve([logits], labels, sequences_per_slice=2)
        self.assertAlmostEqual(float(curve[0]), token_mean, places=12)

    def test_paired_bootstrap_difference_follows_the_replicate_axis(self):
        from experiments.a6_generalization.analysis import chunk_bootstrap

        chunks = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
        samples = chunk_bootstrap(chunks, repeats=20, seed=0)
        self.assertEqual(samples.shape, (2, 20))
        gain = samples[0] - samples[1]
        self.assertEqual(gain.shape, (20,))
        np.testing.assert_allclose(gain, -0.5)

if __name__=='__main__':unittest.main()
