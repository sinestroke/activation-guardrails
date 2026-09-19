import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.extraction import extract_activation_sites, rms_normalize_activations
from src.models import ModelAdapter


class ExtractionTest(unittest.TestCase):
    def test_rms_normalize_activations_normalizes_hidden_dim(self):
        torch.manual_seed(0)
        acts = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16)

        normalized = rms_normalize_activations(acts.clone(), eps=1e-6)

        self.assertEqual(normalized.dtype, torch.bfloat16)
        rms = normalized.float().square().mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=5e-3, rtol=5e-3))
        self.assertFalse(normalized.requires_grad)

    def test_fused_extraction_captures_all_sites_in_one_forward(self):
        class Offset(nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.value = value

            def forward(self, inputs):
                return inputs + self.value

        class Block(nn.Module):
            def __init__(self, offset: float) -> None:
                super().__init__()
                self.self_attn = Offset(offset)
                self.mlp = Offset(offset * 2.0)

            def forward(self, inputs):
                hidden = inputs + self.self_attn(inputs)
                return hidden + self.mlp(hidden)

        class Decoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Block(1.0), Block(2.0)])

        class ToyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Decoder()
                self.anchor = nn.Parameter(torch.ones((), dtype=torch.bfloat16), requires_grad=False)
                self.forward_calls = 0

            def forward(self, input_ids, attention_mask, output_hidden_states=False, use_cache=False):
                del attention_mask, use_cache
                self.forward_calls += 1
                hidden = input_ids.to(torch.bfloat16).unsqueeze(-1).repeat(1, 1, 3) * self.anchor
                hidden_states = [hidden]
                for layer in self.model.layers:
                    hidden = layer(hidden)
                    hidden_states.append(hidden)
                return SimpleNamespace(hidden_states=tuple(hidden_states) if output_hidden_states else None)

        model = ToyModel()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "probe_mask": torch.ones((2, 3), dtype=torch.bool),
            "labels": torch.tensor([0.0, 1.0]),
        }

        activations, mask, labels = extract_activation_sites(
            ModelAdapter(),
            model,
            batch,
            sites=["residual", "mlp", "attention"],
            normalize_activations=False,
        )

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(list(activations), ["residual", "mlp", "attention"])
        for acts in activations.values():
            self.assertEqual(tuple(acts.shape), (2, 3, 2, 3))
            self.assertEqual(acts.dtype, torch.bfloat16)
            self.assertFalse(acts.requires_grad)
        self.assertTrue(torch.equal(mask, batch["probe_mask"]))
        self.assertTrue(torch.equal(labels, batch["labels"]))


if __name__ == "__main__":
    unittest.main()
