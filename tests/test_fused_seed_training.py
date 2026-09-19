import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.train import DEFAULT_CONFIG, EarlyStopState, SiteTrainingBundle, train_bundles_fused


class ToyProbe(torch.nn.Module):
    def __init__(self, initial_weight: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(initial_weight))

    def forward(self, acts, probe_mask, labels, hp):
        del probe_mask, hp
        features = acts.float().mean(dim=(1, 2, 3))
        logits = features * self.weight
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return loss, {}

    def score(self, acts, probe_mask, hp):
        del probe_mask, hp
        features = acts.float().mean(dim=(1, 2, 3))
        return torch.sigmoid(features * self.weight)

    def feature_spec(self, site, include_embedding_layer):
        return {"site": site, "include_embedding_layer": include_embedding_layer}


class OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch
        self.batch_sampler = self

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1

    def set_epoch(self, epoch):
        self.epoch = epoch


class FusedSeedTrainingTest(unittest.TestCase):
    def test_seeds_share_extraction_but_keep_independent_optimizers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = copy.deepcopy(DEFAULT_CONFIG)
            config.update(
                {
                    "epochs": 1,
                    "micro_batch_size": 2,
                    "grad_accum_steps": 1,
                    "lr_scheduler": "none",
                    "log_every": 1,
                    "normalize_activations": False,
                    "include_embedding_layer": False,
                    "resolved_model": {},
                    "invocation_id": "test-invocation",
                    "run_id": "test-run",
                }
            )
            batch = {"input_ids": torch.ones(2, 1, dtype=torch.long)}
            loader = OneBatchLoader(batch)
            extraction_calls = []

            def extract_once(*args, **kwargs):
                extraction_calls.append(tuple(kwargs["sites"]))
                acts = torch.tensor([[[[1.0]]], [[[2.0]]]])
                mask = torch.ones(2, 1, dtype=torch.bool)
                labels = torch.tensor([0.0, 1.0])
                return {"residual": acts}, mask, labels

            bundles = []
            initial_weights = (0.1, -0.2)
            for seed, initial_weight in enumerate(initial_weights):
                probe = ToyProbe(initial_weight)
                seed_config = copy.deepcopy(config)
                seed_config["seed"] = seed
                seed_config["checkpoint_dir"] = str(root / "checkpoints" / f"seed_{seed}")
                seed_config["output_dir"] = str(root / "results" / f"seed_{seed}")
                bundles.append(
                    SiteTrainingBundle(
                        site="residual",
                        site_seed=seed,
                        run_seed=seed,
                        config=seed_config,
                        metrics_path=root / "results" / f"seed_{seed}" / "metrics.jsonl",
                        probes={"toy": probe},
                        optimizers={"toy": torch.optim.AdamW(probe.parameters(), lr=0.01)},
                        schedulers={},
                        states={"toy": EarlyStopState("toy")},
                    )
                )

            with (
                patch("src.train.make_loader", return_value=loader),
                patch("src.train.extract_activation_sites", side_effect=extract_once),
                patch("src.train.model_input_device", return_value=torch.device("cpu")),
            ):
                train_bundles_fused(
                    bundles,
                    adapter=None,
                    model=None,
                    tokenizer=object(),
                    train_dataset=object(),
                    dev_dataset=object(),
                    config=config,
                    loader_seed=0,
                )

            self.assertEqual(extraction_calls, [("residual",), ("residual",)])
            self.assertIsNot(bundles[0].optimizers["toy"], bundles[1].optimizers["toy"])
            self.assertNotEqual(float(bundles[0].probes["toy"].weight.detach()), initial_weights[0])
            self.assertNotEqual(float(bundles[1].probes["toy"].weight.detach()), initial_weights[1])
            for seed in (0, 1):
                self.assertTrue((root / "checkpoints" / f"seed_{seed}" / "residual" / "toy.pt").exists())


if __name__ == "__main__":
    unittest.main()
