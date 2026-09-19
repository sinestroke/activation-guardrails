import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data import DataSettings
from src.eval import (
    ActivationPreprocessing,
    aggregate_metric_rows,
    apply_probe_overrides,
    apply_evaluation_dataset_overrides,
    apply_probe_scorer_config,
    checkpoint_activation_preprocessing,
    configure_metric_protocol,
    discover_seed_roots,
    evaluation_config,
    merge_existing_rows,
    metric_dict,
    plot_eval_summary,
    plot_views,
    score_probe_sets_split,
    score_probe_sites_split,
    scorer_enabled,
    selected_plot_splits,
    select_eval_splits,
    use_shared_evaluation_data,
    validate_seed_roots,
    validate_probe_checkpoint_provenance,
)
from src.train import resolve_artifact_layout


class EvalConfigTest(unittest.TestCase):
    def test_two_percent_metric_protocol_changes_eval_key(self):
        self.addCleanup(configure_metric_protocol, {"tpr_fpr": 0.01})
        configure_metric_protocol({"tpr_fpr": 0.02})

        metrics = metric_dict(
            [1] * 50 + [0] + [1] * 50 + [0] * 49,
            list(range(150, 0, -1)),
        )

        self.assertIn("tpr@2fpr", metrics)
        self.assertNotIn("tpr@1fpr", metrics)

    def test_cross_model_cli_overrides_preserve_source_and_target_provenance(self):
        args = SimpleNamespace(
            evaluation_name="source-to-target",
            target_backbone_name="target-model",
            probe_checkpoint_backbone_name="source-model",
            model_name=None,
            model_revision=None,
            model_loader=None,
            local_files_only=False,
            activation_types=None,
            probe_checkpoint_dir="checkpoints/source",
            batch_size=None,
            max_eval_examples=None,
            probe_seeds=[0, 1, 2, 3, 4],
        )

        config = apply_probe_overrides({}, args)

        self.assertEqual(config["evaluation_name"], "source-to-target")
        self.assertEqual(config["target_backbone_name"], "target-model")
        self.assertEqual(config["probe_checkpoint_backbone_name"], "source-model")
        self.assertEqual(config["checkpoint_dir"], "checkpoints/source")
        self.assertEqual(config["evaluation_seeds"], [0, 1, 2, 3, 4])

    def test_select_eval_splits_from_configured_patterns(self):
        settings = DataSettings(
            format="labelled_text",
            files={
                "train": "data/high-stakes/training/train.jsonl",
                "test_standard_balanced": [
                    "data/high-stakes/evals/test/a_balanced.jsonl",
                    "data/high-stakes/evals/test/b_balance.jsonl",
                ],
                "test_a_balanced": "data/high-stakes/evals/test/a_balanced.jsonl",
                "test_b_balance": "data/high-stakes/evals/test/b_balance.jsonl",
                "test_a_raw": "data/high-stakes/evals/test/a_raw.jsonl",
                "dev_a_balanced": "data/high-stakes/evals/dev/a_balanced.jsonl",
            },
        )
        eval_config = {
            "dataset": {
                "include_patterns": ["test_*_balanced", "test_*_balance"],
                "exclude_patterns": ["*_raw"],
                "require_file_patterns": ["*_balanced.jsonl", "*_balance.jsonl"],
                "disallow_file_patterns": ["*_raw.jsonl"],
            }
        }

        self.assertEqual(select_eval_splits(settings, eval_config), ["test_a_balanced", "test_b_balance", "test_standard_balanced"])

        explicit = {"dataset": {"splits": ["test_standard_balanced"], "require_file_patterns": ["*_balanced.jsonl", "*_balance.jsonl"]}}
        self.assertEqual(select_eval_splits(settings, explicit), ["test_standard_balanced"])

    def test_select_eval_splits_rejects_disallowed_files(self):
        settings = DataSettings(
            format="labelled_text",
            files={"test_raw": "data/high-stakes/evals/test/example_raw.jsonl"},
        )
        eval_config = {
            "dataset": {
                "splits": ["test_raw"],
                "disallow_file_patterns": ["*_raw.jsonl"],
            }
        }

        with self.assertRaisesRegex(ValueError, "Invalid evaluation splits"):
            select_eval_splits(settings, eval_config)

    def test_eval_dataset_and_probe_scorer_overrides_are_config_driven(self):
        config = {
            "data": {"format": "exchange_annotations"},
            "activation_types": ["residual"],
            "checkpoint_dir": "old",
        }
        eval_config = {
            "dataset": {
                "schema": "labelled_text",
                "files": {"test": "data/high-stakes/evals/test/test_balanced.jsonl"},
            },
            "scorers": {
                "probes": {
                    "enabled": True,
                    "checkpoint_dir": "checkpoints/new",
                    "activation_sites": ["mlp", "attention"],
                },
                "classifier": {"enabled": False},
            },
        }

        config = apply_evaluation_dataset_overrides(config, eval_config)
        config = apply_probe_scorer_config(config, eval_config)

        self.assertEqual(config["data"]["format"], "labelled_text")
        self.assertEqual(config["data"]["files"], {"test": "data/high-stakes/evals/test/test_balanced.jsonl"})
        self.assertEqual(config["activation_types"], ["mlp", "attention"])
        self.assertEqual(config["checkpoint_dir"], "checkpoints/new")
        self.assertTrue(scorer_enabled(eval_config, "probes"))
        self.assertFalse(scorer_enabled(eval_config, "classifier"))

    def test_evaluation_profiles_route_validation_outputs_and_splits(self):
        config = {
            "eval_splits": ["dev"],
            "evaluation": {
                "default_profile": "test",
                "output_dir": "results/eval/example",
                "dataset": {
                    "splits": ["test_standard_balanced"],
                    "require_file_patterns": ["*_balanced.jsonl", "*_balance.jsonl"],
                    "disallow_file_patterns": ["*_raw.jsonl"],
                },
                "profiles": {
                    "test": {},
                    "validation": {
                        "output_dir": "results/validation/example",
                        "dataset": {
                            "splits": ["dev"],
                            "require_file_patterns": [],
                            "disallow_file_patterns": [],
                        },
                        "plots": {"splits": ["dev"]},
                    },
                },
            },
        }

        test_config = evaluation_config(config)
        validation_config = evaluation_config(config, profile="validation")

        self.assertEqual(test_config["profile"], "test")
        self.assertEqual(test_config["output_dir"], "results/eval/example")
        self.assertEqual(test_config["dataset"]["splits"], ["test_standard_balanced"])
        self.assertEqual(validation_config["profile"], "validation")
        self.assertEqual(validation_config["output_dir"], "results/validation/example")
        self.assertEqual(validation_config["dataset"]["splits"], ["dev"])
        self.assertEqual(validation_config["dataset"]["require_file_patterns"], [])
        self.assertEqual(validation_config["plots"]["splits"], ["dev"])

    def test_unknown_evaluation_profile_is_rejected(self):
        config = {"evaluation": {"profiles": {"test": {}}}}

        with self.assertRaisesRegex(ValueError, "Unknown evaluation profile"):
            evaluation_config(config, profile="validation")

    def test_classifier_eval_uses_shared_eval_data_splits(self):
        classifier_config = {
            "model_name": "classifier-model",
            "checkpoint_dir": "classifier-checkpoints",
            "data": {
                "format": "labelled_text",
                "files": {"test_anthropic_hh_balanced": "old/anthropic_hh_balanced.jsonl"},
            },
        }
        probe_config = {
            "model_name": "probe-model",
            "checkpoint_dir": "probe-checkpoints",
            "data": {
                "format": "labelled_text",
                "files": {
                    "test_standard_balanced": [
                        "data/high-stakes/source_a/test.jsonl",
                        "data/high-stakes/source_b/test.jsonl",
                    ],
                    "test_secondary": "data/high-stakes/source_c/test.jsonl",
                },
                "input_field": "inputs",
                "label_field": "labels",
            },
        }

        merged = use_shared_evaluation_data(classifier_config, probe_config)

        self.assertEqual(merged["model_name"], "classifier-model")
        self.assertEqual(merged["checkpoint_dir"], "classifier-checkpoints")
        self.assertEqual(merged["data"], probe_config["data"])
        self.assertNotIn("test_anthropic_hh_balanced", merged["data"]["files"])

    def test_plot_splits_can_select_a_configured_subset(self):
        eval_config = {"plots": {"splits": ["test_standard_balanced"]}}
        splits = [
            "test_standard_balanced",
            "test_secondary",
            "test_tertiary",
        ]

        self.assertEqual(selected_plot_splits(eval_config, splits), {"test_standard_balanced"})

    def test_plot_views_gate_generated_svg_families(self):
        summary = [
            {
                "family": "probe",
                "method": "rmattn",
                "site": "mlp",
                "split": "test",
                "metric": "logspace_auroc",
                "n": 5,
                "mean": 0.81,
                "std": 0.02,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            views = plot_views({"plots": {"views": ["activation_comparison"]}})
            plot_eval_summary(summary, output_dir, views)

            plot_dir = output_dir / "plots"
            self.assertTrue((plot_dir / "activation_comparison_test_rmattn_logspace_auroc.svg").exists())
            self.assertFalse(any(plot_dir.glob("overview_*.svg")))

    def test_probe_eval_extracts_once_per_batch_and_scores_all_seeds(self):
        class FakeProbe:
            def __init__(self, offset: float) -> None:
                self.offset = offset

            def score(self, acts, probe_mask, hp):
                del probe_mask, hp
                return acts[:, 0, 0, 0].float() + self.offset

        batches = [
            {"labels": torch.tensor([0.0, 1.0])},
            {"labels": torch.tensor([1.0])},
        ]
        calls = []

        def fake_extract(adapter, model, batch, **kwargs):
            del adapter, model, kwargs
            calls.append(batch)
            batch_size = int(batch["labels"].numel())
            acts = torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1, 1, 1)
            mask = torch.ones((batch_size, 1), dtype=torch.bool)
            return acts, mask, batch["labels"]

        probe_sets = [
            (0, {"mean": FakeProbe(0.0)}, object()),
            (1, {"mean": FakeProbe(10.0), "swim": FakeProbe(20.0)}, object()),
        ]

        with patch("src.eval.extract_activations", side_effect=fake_extract):
            y_true, scores = score_probe_sets_split(
                probe_sets,
                adapter=None,
                model=None,
                loader=batches,
                site="residual",
                config={
                    "include_embedding_layer": False,
                    "use_nnsight": True,
                    "normalize_activations": True,
                    "activation_norm_eps": 1e-6,
                },
                preprocessing=ActivationPreprocessing(False, None),
            )

        self.assertEqual(len(calls), len(batches))
        self.assertEqual(y_true.tolist(), [0, 1, 1])
        self.assertEqual(scores[(0, "mean")].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(scores[(1, "mean")].tolist(), [10.0, 11.0, 10.0])
        self.assertEqual(scores[(1, "swim")].tolist(), [20.0, 21.0, 20.0])

    def test_fused_probe_eval_extracts_once_per_batch_for_all_sites(self):
        class FakeProbe:
            def score(self, acts, probe_mask, hp):
                del probe_mask, hp
                return acts[:, 0, 0, 0].float()

        batches = [
            {"labels": torch.tensor([0.0, 1.0])},
            {"labels": torch.tensor([1.0])},
        ]
        calls = []

        def fake_extract(adapter, model, batch, sites, **kwargs):
            del adapter, model, kwargs
            calls.append((batch, tuple(sites)))
            batch_size = int(batch["labels"].numel())
            base = torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1, 1, 1)
            activations = {
                site: base + float(index * 10)
                for index, site in enumerate(sites)
            }
            mask = torch.ones((batch_size, 1), dtype=torch.bool)
            return activations, mask, batch["labels"]

        probe_sets_by_site = {
            "residual": [(0, {"mean": FakeProbe()}, object())],
            "mlp": [(0, {"mean": FakeProbe()}, object()), (1, {"swim": FakeProbe()}, object())],
        }
        config = {
            "include_embedding_layer": False,
            "normalize_activations": False,
            "activation_norm_eps": 1e-6,
        }

        with patch("src.eval.extract_activation_sites", side_effect=fake_extract):
            y_true, scores = score_probe_sites_split(
                probe_sets_by_site,
                {
                    "residual": ActivationPreprocessing(False, None),
                    "mlp": ActivationPreprocessing(False, None),
                },
                adapter=None,
                model=None,
                loader=batches,
                config=config,
            )

        self.assertEqual(len(calls), len(batches))
        self.assertTrue(all(sites == ("residual", "mlp") for _batch, sites in calls))
        self.assertEqual(y_true.tolist(), [0, 1, 1])
        self.assertEqual(scores[("residual", 0, "mean")].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(scores[("mlp", 0, "mean")].tolist(), [10.0, 11.0, 10.0])
        self.assertEqual(scores[("mlp", 1, "swim")].tolist(), [10.0, 11.0, 10.0])

    def test_fused_probe_eval_exports_raw_logits_with_unique_row_keys(self):
        class FakeProbe:
            def score_logits(self, acts, probe_mask, hp):
                del probe_mask, hp
                return acts[:, 0, 0, 0].float()

        batch = {
            "labels": torch.tensor([0.0, 1.0]),
            "row_keys": ["test:row-0", "test:row-1"],
            "exchange_ids": ["duplicate-id", "duplicate-id"],
            "splits": ["test", "test"],
        }

        def fake_extract(adapter, model, _batch, sites, **kwargs):
            del adapter, model, kwargs
            activations = {
                site: torch.tensor([[[[-2.0]]], [[[3.0]]]])
                for site in sites
            }
            return activations, torch.ones((2, 1), dtype=torch.bool), _batch["labels"]

        exported = []
        with patch("src.eval.extract_activation_sites", side_effect=fake_extract):
            score_probe_sites_split(
                {"residual": [(0, {"mean": FakeProbe()}, object())]},
                {"residual": ActivationPreprocessing(False, None)},
                adapter=None,
                model=None,
                loader=[batch],
                config={"include_embedding_layer": False},
                prediction_rows=exported,
                target_backbone="target",
                training_backbone="training",
            )

        self.assertEqual([row["row_key"] for row in exported], ["test:row-0", "test:row-1"])
        self.assertEqual([row["exchange_id"] for row in exported], ["duplicate-id", "duplicate-id"])
        self.assertEqual([row["logit"] for row in exported], [-2.0, 3.0])

    def test_seed_discovery_can_filter_to_requested_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (0, 1, 2, 3, 4):
                (root / f"seed_{seed}").mkdir()

            discovered = discover_seed_roots(root, [0, 1, 2])

        self.assertEqual([seed for seed, _path in discovered], [0, 1, 2])

    def test_seed_validation_rejects_parent_of_artifact_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "direct-block-outputs-v1"
            for seed in (0, 1, 2):
                (run_root / f"seed_{seed}").mkdir(parents=True)

            discovered = discover_seed_roots(root, [0, 1, 2])
            with self.assertRaisesRegex(FileNotFoundError, "one level too high"):
                validate_seed_roots(root, discovered, [0, 1, 2], "probe")

    def test_merge_existing_results_replaces_only_evaluated_family(self):
        existing = [
            {"family": "classifier", "seed": 0, "auroc": 0.7},
            {"family": "probe", "seed": 0, "auroc": 0.6},
        ]
        current = [{"family": "probe", "seed": 0, "auroc": 0.9}]

        merged = merge_existing_rows(existing, current, {"probe"})

        self.assertEqual(
            merged,
            [
                {"family": "classifier", "seed": 0, "auroc": 0.7},
                {"family": "probe", "seed": 0, "auroc": 0.9},
            ],
        )

    def test_checkpoint_activation_preprocessing_uses_training_config(self):
        fallback = ActivationPreprocessing(True, 1e-6)

        raw = checkpoint_activation_preprocessing(
            {"config": {"normalize_activations": False, "activation_norm_eps": 1e-4}},
            Path("raw.pt"),
            fallback,
        )
        normalized = checkpoint_activation_preprocessing(
            {"config": {"normalize_activations": True, "activation_norm_eps": 1e-5}},
            Path("normalized.pt"),
            ActivationPreprocessing(False, None),
        )

        self.assertEqual(raw, ActivationPreprocessing(False, None))
        self.assertEqual(normalized, ActivationPreprocessing(True, 1e-5))

    def test_metric_aggregate_is_unweighted_across_datasets_per_seed(self):
        rows = []
        for seed, first, second in ((0, 0.2, 0.8), (1, 0.4, 1.0)):
            for split, value, count in (("small", first, 10), ("large", second, 1000)):
                rows.append(
                    {
                        "family": "probe",
                        "method": "mean",
                        "site": "residual",
                        "seed": seed,
                        "split": split,
                        "scope": "dataset",
                        "n_examples": count,
                        "auroc": value,
                        "logspace_auroc": value,
                        "tpr@1fpr": value,
                    }
                )
        eval_config = {
            "dataset": {
                "metric_aggregates": {
                    "paper_macro": {
                        "splits": ["small", "large"],
                        "reduction": "unweighted_mean",
                    }
                }
            }
        }

        aggregate = aggregate_metric_rows(rows, eval_config)

        self.assertEqual(len(aggregate), 2)
        self.assertAlmostEqual(aggregate[0]["auroc"], 0.5)
        self.assertAlmostEqual(aggregate[1]["auroc"], 0.7)
        self.assertEqual(aggregate[0]["n_datasets"], 2)

    def test_checkpoint_provenance_rejects_wrong_training_backbone(self):
        checkpoint = {
            "provenance": {
                "site": "mlp",
                "probe_name": "mean",
                "backbone_name": "gemma-3-12b-it-vanilla",
            }
        }

        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_probe_checkpoint_provenance(
                checkpoint,
                Path("mean.pt"),
                site="mlp",
                probe_name="mean",
                expected_backbone="other-backbone",
                require_provenance=True,
            )

    def test_artifact_layout_encodes_family_variant_and_run(self):
        config = {
            "experiment_name": "high-stakes-detection",
            "backbone_family": "example-family",
            "backbone_variant": "example-variant",
            "artifact_layout": {
                "enabled": True,
                "run_name": "rmsnorm-all-sites-v1",
                "output_root": "results/probe_training",
                "checkpoint_root": "checkpoints/probes",
            },
        }

        resolved = resolve_artifact_layout(config)

        suffix = "high-stakes-detection/example-family/example-variant/rmsnorm-all-sites-v1"
        self.assertTrue(resolved["output_dir"].endswith(suffix))
        self.assertTrue(resolved["checkpoint_dir"].endswith(suffix))


if __name__ == "__main__":
    unittest.main()
