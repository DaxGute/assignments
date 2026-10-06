"""Handoff checks for the three A2 batch commands. No jobs are submitted."""

import math
import unittest

from experiments.a2.batches.batch1 import build_batch1
from experiments.a2.batches.batch2 import Batch2Error, plan_batch2
from experiments.a2.batches.batch3 import build_momentum_jobs, select_measured_lr_wd
from experiments.a2.batches.batch3 import SourceRun
from experiments.a2.helpers.results import (
    canonicalize_parameterization,
    get_best_run,
    get_loss,
    read_metadata,
)


def _p32(**overrides):
    record = {
        "run_id": "run",
        "name": "run",
        "state": "finished",
        "assignment": "a2",
        "batch": "batch1",
        "problem": "3.2",
        "subpart": "a",
        "family": "batch_lr_source",
        "config_role": "source",
        "batch_size": 8,
        "peak_lr": 0.003,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "token_budget": 614_400_000,
        "tokens": 614_400_000,
        "width": 512,
        "depth": 8,
        "parameterization": "baseline",
        "optimizer": "adamw",
        "seed": 42,
        "lr_schedule": "linear",
        "warmup_percent": 0.01,
        "model_name": "a2-d8",
        "final_val_loss": 3.0,
        "memberships": [],
        "inconsistencies": [],
        "state_completed": True,
        "completed": True,
        "status": "finished",
    }
    record.update(overrides)
    return record


class MetadataContractTest(unittest.TestCase):
    def test_batch1_writes_canonical_fields(self):
        plan = build_batch1()
        job = next(
            item
            for item in plan["jobs"]
            if any(
                membership["problem"] == "3.2" and membership["subpart"] == "a"
                for membership in item["memberships"]
            )
        )
        metadata = job["metadata"]
        for key in (
            "assignment",
            "batch",
            "problem",
            "subpart",
            "family",
            "config_role",
            "token_budget",
            "batch_size",
            "peak_lr",
            "weight_decay",
            "beta1",
            "beta2",
            "width",
            "depth",
            "parameterization",
            "optimizer",
            "seed",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["assignment"], "a2")
        self.assertEqual(metadata["batch"], "batch1")
        self.assertEqual(metadata["parameterization"], "baseline")
        self.assertEqual(metadata["optimizer"], "adamw")
        self.assertEqual(metadata["peak_lr"], job["config"].learning_rate)
        self.assertNotIn(metadata["batch_size"], {128, 256})

    def test_reader_prefers_peak_lr_and_token_budget(self):
        metadata = _p32(batch_size=256, batch="batch2", subpart="b")
        config = {
            "assignment": "a2",
            "learning_rate": metadata["peak_lr"],
            "optim_lr": metadata["peak_lr"],
            "weight_decay": metadata["weight_decay"],
            "optimizer_name": "adamw",
            "beta1": 0.9,
            "beta2": 0.95,
            "batch_size": 256,
            "lr_schedule": "linear",
            "warmup_percent": 0.01,
            "model_name": "a2-d8",
            "model_seed": 42,
            "train_tokens": 614_400_000 - 192 * 1024,
            "experiment_metadata": metadata,
        }
        parsed = read_metadata(
            config,
            ("a2", "batch2"),
            {"val_loss": 2.5},
            state="finished",
            run_id="abc",
            name="tracked-name",
        )
        self.assertEqual(parsed["inconsistencies"], [])
        self.assertAlmostEqual(parsed["peak_lr"], metadata["peak_lr"])
        self.assertEqual(parsed["token_budget"], 614_400_000)
        self.assertAlmostEqual(get_loss(parsed), 2.5)
        self.assertEqual(parsed["batch"], "batch2")
        self.assertEqual(parsed["parameterization"], "baseline")
        self.assertTrue(parsed["completed"])

    def test_parameterization_aliases(self):
        self.assertEqual(canonicalize_parameterization("µP"), "mup")
        self.assertEqual(canonicalize_parameterization("standard_mup"), "mup")
        self.assertEqual(canonicalize_parameterization("baseline"), "baseline")
        parsed = read_metadata(
            {"parameterization": "standard_mup"},
            (),
            {},
            state="finished",
        )
        self.assertEqual(parsed["parameterization"], "mup")


class HandoffTest(unittest.TestCase):
    def test_best_measured_pairs_drive_batch3(self):
        runs = [
            _p32(run_id="b8-worse", peak_lr=0.006, weight_decay=0.1, final_val_loss=3.2),
            _p32(run_id="b8-best", peak_lr=0.003, weight_decay=0.2, final_val_loss=2.4),
            _p32(run_id="b8-dup", peak_lr=0.003, weight_decay=0.2, final_val_loss=2.4),
            _p32(run_id="b8-running", state="running", completed=False, final_val_loss=None),
            _p32(run_id="b8-failed", state="failed", completed=False, final_val_loss=1.0),
            _p32(
                run_id="b256-best",
                batch="batch2",
                subpart="b",
                batch_size=256,
                peak_lr=0.0015,
                weight_decay=0.4,
                final_val_loss=2.2,
            ),
            _p32(
                run_id="b256-worse",
                batch="batch2",
                subpart="b",
                batch_size=256,
                peak_lr=0.004,
                weight_decay=0.1,
                final_val_loss=2.8,
            ),
        ]
        small = get_best_run(runs, batch_size=8)
        large = get_best_run(runs, batch_size=256)
        self.assertEqual(small["winner"]["run_id"], "b8-best")
        self.assertEqual(large["winner"]["run_id"], "b256-best")
        self.assertTrue(small["deduped_groups"])
        sources = []
        for record in runs:
            sources.append(
                SourceRun(
                    run_id=record["run_id"],
                    name=record["name"],
                    state=record["state"],
                    assignment=record["assignment"],
                    problem=record["problem"],
                    subpart=record["subpart"],
                    batch=record["batch"],
                    family=record["family"],
                    config_role=record["config_role"],
                    batch_size=record["batch_size"],
                    peak_lr=record["peak_lr"],
                    weight_decay=record["weight_decay"],
                    beta1=record["beta1"],
                    beta2=record["beta2"],
                    token_budget=record["token_budget"],
                    lr_schedule=record["lr_schedule"],
                    warmup_percent=record["warmup_percent"],
                    model_name=record["model_name"],
                    optimizer_name=record["optimizer"],
                    momentum_role=None,
                    status=record["state"],
                    final_val_loss=record["final_val_loss"],
                    width=record["width"],
                    depth=record["depth"],
                    parameterization=record["parameterization"],
                )
            )
        selections = {
            8: select_measured_lr_wd(sources, 8),
            256: select_measured_lr_wd(sources, 256),
        }
        jobs = build_momentum_jobs(selections)
        self.assertEqual(len(jobs), 12)
        for job in jobs:
            if job.batch_size == 8:
                self.assertTrue(math.isclose(job.peak_lr, 0.003))
                self.assertTrue(math.isclose(job.weight_decay, 0.2))
            else:
                self.assertTrue(math.isclose(job.peak_lr, 0.0015))
                self.assertTrue(math.isclose(job.weight_decay, 0.4))
            self.assertEqual(job.metadata["batch"], "batch3")
            self.assertEqual(job.metadata["peak_lr"], job.config.learning_rate)
            self.assertEqual(job.metadata["weight_decay"], job.config.weight_decay)
            self.assertEqual(job.config.model_config.name, "a2-d8")

    def test_missing_batch1_is_explicit(self):
        with self.assertRaises(Batch2Error) as caught:
            plan_batch2([])
        message = str(caught.exception)
        self.assertIn("Batch 2 cannot run yet:", message)
        self.assertIn("required Batch 1 source experiments are missing or incomplete.", message)

    def test_no_job_depends_on_a_later_result_in_the_same_batch(self):
        plan = build_batch1()
        for job in plan["jobs"]:
            for membership in job["memberships"]:
                self.assertNotEqual((membership["problem"], membership["subpart"]), ("3.2", "c"))
                self.assertNotEqual((membership["problem"], membership["subpart"]), ("4.2", "c"))
            self.assertNotEqual(job["hyperparameters"].get("width"), 1024)
            if any(membership["problem"] == "3.2" for membership in job["memberships"]):
                self.assertNotIn(job["hyperparameters"]["batch_size"], {128, 256})
        later = plan_batch2(_closed_measurements(), supplied=[])
        for hypothesis in later["p32b"]["hypotheses"]:
            for job in hypothesis["jobs"]:
                source = job["membership"]["prediction_source"]
                self.assertEqual(source["source_batches"], [8, 16, 32, 64])
                self.assertNotIn(128, source["source_batches"])
                self.assertNotIn(256, source["source_batches"])
        for prescription in later["p42c"]["prescriptions"]:
            for job in prescription["jobs"]:
                source = job["membership"]["prediction_source"]
                self.assertEqual(list(source["source_widths"]), [128, 256, 512])
                self.assertNotIn(1024, source["source_widths"])


def _closed_measurements():
    rows = []
    for batch_size in (8, 16, 32, 64):
        for learning_rate in (0.0015, 0.003, 0.006):
            rows.append(
                _p32(
                    batch_size=batch_size,
                    peak_lr=learning_rate,
                    weight_decay=0.1,
                    final_val_loss=(math.log(learning_rate) - math.log(0.003)) ** 2 + 2.0,
                    tokens=614_400_000,
                )
            )
        for weight_decay in (0.1, 0.2, 0.4, 0.8):
            rows.append(
                _p32(
                    subpart="b",
                    batch_size=batch_size,
                    peak_lr=0.0015,
                    weight_decay=weight_decay,
                    final_val_loss=(math.log(weight_decay) - math.log(0.2)) ** 2 + 2.0,
                    tokens=614_400_000,
                )
            )
    for parameterization in ("baseline", "mup"):
        for width in (128, 256):
            for learning_rate in (0.0015, 0.003, 0.006):
                rows.append(
                    _p32(
                        problem="4.2",
                        subpart="a",
                        family="long_width_transfer",
                        batch_size=64,
                        peak_lr=learning_rate,
                        weight_decay=0.1,
                        width=width,
                        parameterization=parameterization,
                        tokens=153_600_000,
                        token_budget=153_600_000,
                        final_val_loss=(math.log(learning_rate) - math.log(0.003)) ** 2 + 3.0,
                    )
                )
    for learning_rate in (0.0015, 0.003, 0.006):
        rows.append(
            _p32(
                problem="1",
                subpart="a",
                batch_size=64,
                peak_lr=learning_rate,
                weight_decay=0.1,
                tokens=153_600_000,
                token_budget=153_600_000,
                final_val_loss=(math.log(learning_rate) - math.log(0.003)) ** 2 + 3.0,
            )
        )
    return rows


class SummaryLossTest(unittest.TestCase):
    def test_wandb_summary_is_used_without_scanning_history(self):
        from experiments.a2.helpers.results import describe_run

        class _Summary:
            def __init__(self, payload):
                self._json_dict = payload

        class _Run:
            config = {
                "assignment": "a2",
                "batch": "batch1",
                "problem": "3.2",
                "subpart": "a",
                "peak_lr": 0.003,
                "batch_size": 64,
                "token_budget": 614_400_000,
                "weight_decay": 0.1,
            }
            summary_metrics = {"val_loss": 2.5}
            summary = _Summary({"val_loss": 2.5})
            tags = ()
            state = "finished"
            id = "abc"
            name = "abc"

            def scan_history(self, **kwargs):
                raise AssertionError(kwargs)

        record = describe_run(_Run())
        self.assertEqual(record["final_val_loss"], 2.5)
        self.assertTrue(record["completed"])


if __name__ == "__main__":
    unittest.main()
