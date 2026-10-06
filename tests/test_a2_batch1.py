"""CPU checks for the Batch 1 plan and the Batch 2 measurement interface."""

import math
import unittest
from dataclasses import asdict

import torch

from experiments.a2.batches.batch1 import (
    P41A_BASE_LRS,
    P41A_WIDTHS,
    P41C_BASE_LRS,
    P42A_EXPONENTS,
    P42A_WIDTHS,
    build_batch1,
    stress_key,
)
from experiments.a2.batches.batch2 import Batch2Error, plan_batch2
from experiments.a2.policies import (
    build_scaled_optimizer,
    depth_multipliers,
    initialize_stress,
    long_scaling,
    stress_groups,
)
from experiments.a2.policies import build_scaled_model
from experiments.a2.stress import StressConfig, StressTransformer
from model_config import LMConfig
from modeling import AutoregressiveLM, initialize_model


def _jobs(plan, problem, subpart):
    return [
        job
        for job in plan["jobs"]
        if any(
            membership["problem"] == problem and membership["subpart"] == subpart
            for membership in job["memberships"]
        )
    ]


def _synthetic_measurements():
    rows = []
    for batch_size in (8, 16, 32, 64):
        lr_star = 0.003 * (64 / batch_size) ** 0.5
        for learning_rate in (0.0015, 0.003, 0.006):
            rows.append(
                {
                    "problem": "3.2",
                    "subpart": "a",
                    "hypothesis": "fixed_wd_scale_lr",
                    "batch_size": batch_size,
                    "peak_lr": learning_rate,
                    "weight_decay": 0.1,
                    "width": 512,
                    "depth": 8,
                    "parameterization": "baseline",
                    "tokens": 614_400_000,
                    "optimizer": "adamw",
                    "lr_schedule": "linear",
                    "seed": 42,
                    "final_val_loss": (math.log(learning_rate) - math.log(lr_star)) ** 2 + 2.0,
                }
            )
        wd_star = 0.2 * (batch_size / 64) ** 0.5
        for weight_decay in (0.1, 0.2, 0.4, 0.8):
            if math.isclose(weight_decay, 0.1):
                loss = (math.log(0.0015) - math.log(lr_star)) ** 2 + 2.0
            else:
                loss = (math.log(weight_decay) - math.log(wd_star)) ** 2 + 2.0
            rows.append(
                {
                    "problem": "3.2",
                    "subpart": "b",
                    "hypothesis": "fixed_lr_scale_wd",
                    "batch_size": batch_size,
                    "peak_lr": 0.0015,
                    "weight_decay": weight_decay,
                    "width": 512,
                    "depth": 8,
                    "parameterization": "baseline",
                    "tokens": 614_400_000,
                    "optimizer": "adamw",
                    "lr_schedule": "linear",
                    "seed": 42,
                    "final_val_loss": loss,
                }
            )
    for parameterization in ("baseline", "mup"):
        for width in (128, 256):
            lr_star = 0.003 * (512 / width) ** 0.5
            for exponent in (-2, -1, 0, 1, 2):
                learning_rate = 0.003 * 2.0**exponent
                rows.append(
                    {
                        "problem": "4.2",
                        "subpart": "a",
                        "batch_size": 64,
                        "peak_lr": learning_rate,
                        "weight_decay": 0.1,
                        "width": width,
                        "depth": 8,
                        "parameterization": parameterization,
                        "tokens": 153_600_000,
                        "optimizer": "adamw",
                        "lr_schedule": "linear",
                        "seed": 42,
                        "final_val_loss": (math.log(learning_rate) - math.log(lr_star)) ** 2 + 3.0,
                    }
                )
    for learning_rate in (0.0015, 0.003, 0.006):
        rows.append(
            {
                "problem": "1",
                "subpart": "a",
                "batch_size": 64,
                "peak_lr": learning_rate,
                "weight_decay": 0.1,
                "width": 512,
                "depth": 8,
                "parameterization": "baseline",
                "tokens": 153_600_000,
                "optimizer": "adamw",
                "lr_schedule": "linear",
                "seed": 42,
                "final_val_loss": (math.log(learning_rate) - math.log(0.003)) ** 2 + 3.0,
            }
        )
    return rows


class Batch1PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_batch1()

    def test_labels_and_identities_are_unique(self):
        jobs = self.plan["jobs"]
        labels = [job["label"] for job in jobs]
        self.assertEqual(len(labels), len(set(labels)))
        names = [name for job in jobs for name in (job["label"], *job["aliases"])]
        self.assertEqual(len(names), len(set(names)))
        identities = [job["identity"] for job in jobs]
        self.assertEqual(len(identities), len(set(identities)))

    def test_later_batches_are_not_scheduled(self):
        for job in self.plan["jobs"]:
            self.assertNotIn(job["hyperparameters"].get("batch_size"), {128, 256})
            self.assertNotEqual(job["hyperparameters"].get("width"), 1024)
            for membership in job["memberships"]:
                self.assertNotEqual((membership["problem"], membership["subpart"]), ("3.2", "c"))
                self.assertNotEqual((membership["problem"], membership["subpart"]), ("4.2", "c"))
                self.assertNotEqual(membership["problem"], "3.1")
        self.assertTrue(self.plan["optional_p2c_transfer"]["scheduled"])

    def test_known_grid_counts(self):
        self.assertEqual(len(_jobs(self.plan, "4.1", "a")), len(P41A_BASE_LRS) * len(P41A_WIDTHS) * 2)
        depth = _jobs(self.plan, "4.1", "c")
        self.assertEqual(len(depth), len(P41C_BASE_LRS) * 7)
        shared_depth = [
            job
            for job in depth
            if job["hyperparameters"]["depth"] == 2
        ]
        self.assertEqual(len(shared_depth), len(P41C_BASE_LRS))
        for job in shared_depth:
            self.assertGreaterEqual(len(job["aliases"]), 2)
        self.assertEqual(len(_jobs(self.plan, "4.2", "a")), len(P42A_WIDTHS) * len(P42A_EXPONENTS) * 2)
        self.assertEqual(len(_jobs(self.plan, "4.2", "d")), 18)
        self.assertEqual(len(_jobs(self.plan, "3.2", "a")), 9)
        self.assertLessEqual(len(_jobs(self.plan, "1", "c")), 5)
        self.assertLessEqual(len(_jobs(self.plan, "1", "d")), 4)

    def test_supplied_runs_are_references(self):
        references = self.plan["references"]
        self.assertTrue(
            any(
                item["hyperparameters"]["token_budget"] == 614_400_000
                and item["hyperparameters"]["batch_size"] == 64
                and math.isclose(item["hyperparameters"]["peak_lr"], 0.003)
                and math.isclose(item["hyperparameters"]["weight_decay"], 0.1)
                for item in references
            )
        )
        self.assertTrue(
            any(
                item["hyperparameters"]["token_budget"] == 2_457_600_000
                and math.isclose(item["hyperparameters"]["peak_lr"], 0.003)
                and math.isclose(item["hyperparameters"]["weight_decay"], 0.1)
                for item in references
            )
        )
        scheduled_p2c = _jobs(self.plan, "2", "c")
        self.assertTrue(scheduled_p2c)
        self.assertTrue(
            all(
                not (
                    math.isclose(job["hyperparameters"]["peak_lr"], 0.003)
                    and math.isclose(job["hyperparameters"]["weight_decay"], 0.1)
                )
                for job in scheduled_p2c
            )
        )

    def test_fits_are_positive(self):
        fits = self.plan["fits"]
        for key in ("p1c_all6", "p1c_large3", "p1d_hyperball"):
            self.assertGreater(fits[key]["predicted_lr"], 0.0)
        self.assertGreater(fits["p2c_product"]["predicted_wd"], 0.0)
        self.assertAlmostEqual(fits["p1_1536m_best_sampled"]["learning_rate"], 0.003)

    def test_metadata_does_not_contradict_the_training_config(self):
        reserved_extra = {
            "model_name",
            "optim_lr",
            "train_tokens",
            "parameter_count",
            "run_name",
            "precision",
            "param_precision",
            "compute_precision",
            "loss_precision",
            "optimizer_state_precision",
            "uses_autocast",
        }
        for job in self.plan["jobs"]:
            if job["config"] is None:
                continue
            stored = asdict(job["config"])
            for key, value in job["metadata"].items():
                self.assertNotIn(key, reserved_extra)
                if key in stored and stored[key] != value:
                    self.fail(f"{job['label']} metadata {key}={value!r} contradicts {stored[key]!r}")


class PolicyTest(unittest.TestCase):
    def test_depth_two_prescriptions_collapse(self):
        keys = [
            stress_key(policy, 64, 2, 1e-3, "mp", 64, 2)
            for policy in ("mup", "depth_mup", "completep")
        ]
        self.assertEqual(len(set(keys)), 1)
        deeper = stress_key("mup", 64, 100, 1e-3, "mp", 64, 2)
        self.assertNotEqual(deeper, keys[0])
        self.assertNotEqual(
            stress_key("depth_mup", 64, 100, 1e-3, "mp", 64, 2),
            stress_key("completep", 64, 100, 1e-3, "mp", 64, 2),
        )

    def test_depth_multipliers(self):
        self.assertEqual(depth_multipliers("mup", 100, 2), (1.0, 1.0, 1.0))
        residual, block_lr, block_eps = depth_multipliers("depth_mup", 8, 2)
        self.assertAlmostEqual(residual, 0.5)
        self.assertAlmostEqual(block_lr, 0.5)
        self.assertAlmostEqual(block_eps, 0.5)
        residual, block_lr, block_eps = depth_multipliers("completep", 8, 2)
        self.assertAlmostEqual(residual, 0.25)
        self.assertEqual(block_lr, 1.0)
        self.assertAlmostEqual(block_eps, 0.25)

    def test_stress_groups_scale_hidden_lr_and_cover_parameters(self):
        config = StressConfig(width=64, depth=2, head_dim=64, precision="fp32")
        model = StressTransformer(config, device="cpu")
        initialize_stress(model, "mup", 32, 2)
        groups = stress_groups(model, 1e-3, "mup", 32, 2)
        covered = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual({id(parameter) for parameter in covered}, {id(parameter) for parameter in model.parameters()})
        hidden = next(group for group in groups if math.isclose(group["lr"], 5e-4))
        self.assertTrue(hidden["params"])
        self.assertTrue(all(group["lr"] > 0 and group["eps"] > 0 for group in groups))
        self.assertTrue(all(set(group) == {"params", "lr", "eps"} for group in groups))

    def test_reference_width_mup_matches_course_initialization(self):
        config = LMConfig(
            "tiny",
            vocab_size=128,
            context_length=16,
            hidden_size=64,
            intermediate_size=224,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
        )
        torch.manual_seed(0)
        baseline = AutoregressiveLM(config)
        torch.manual_seed(0)
        initialize_model(baseline)
        torch.manual_seed(0)
        scaled = build_scaled_model(
            config,
            **long_scaling("mup", 64, 2, reference_width=64, reference_depth=2),
        )
        torch.manual_seed(0)
        scaled.initialize_parameters()
        for left, right in zip(baseline.parameters(), scaled.parameters()):
            self.assertTrue(torch.allclose(left, right, rtol=1e-5, atol=1e-5))
        self.assertAlmostEqual(scaled.output_multiplier, 1.0)

    def test_hidden_learning_rate_scales_with_width(self):
        config = LMConfig(
            "wide",
            vocab_size=128,
            context_length=16,
            hidden_size=128,
            intermediate_size=448,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
        )
        model = build_scaled_model(config, **long_scaling("mup", 128, 8))
        optimizer = build_scaled_optimizer(
            model,
            "adamw",
            0.003,
            0.1,
            0.9,
            0.95,
        )
        hidden = model.model.layers[0].self_attn.q_proj.weight
        group = next(group for group in optimizer.param_groups if any(parameter is hidden for parameter in group["params"]))
        self.assertAlmostEqual(group["lr"], 0.012)
        self.assertIsNot(optimizer.defaults.get("fused"), True)


class Batch2InterfaceTest(unittest.TestCase):
    def test_missing_measurements_are_rejected(self):
        with self.assertRaises(Batch2Error):
            plan_batch2([])

    def test_plan_uses_measurements_for_later_targets(self):
        plan = plan_batch2(_synthetic_measurements(), supplied=[])
        batches = {
            job["hyperparameters"]["batch_size"]
            for hypothesis in plan["p32b"]["hypotheses"]
            for job in hypothesis["jobs"]
        }
        self.assertEqual(batches, {128, 256})
        widths = {
            job["hyperparameters"]["width"]
            for prescription in plan["p42c"]["prescriptions"]
            for job in prescription["jobs"]
        }
        self.assertEqual(widths, {1024})
        for prescription in plan["p42c"]["prescriptions"]:
            self.assertGreater(prescription["predicted_lr"], 0.0)
            roles = {job["membership"]["config_role"] for job in prescription["jobs"]}
            self.assertIn("powerlaw_prediction", roles)
            self.assertIn("direct_transfer", roles)


if __name__ == "__main__":
    unittest.main()
