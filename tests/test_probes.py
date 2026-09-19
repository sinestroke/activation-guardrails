import unittest

import torch

from src.probes import (
    PROBE_NAMES,
    ProbeHyperparams,
    build_probe,
    ema_streaming_logits,
    ema_streaming_scores,
    masked_mean,
    probe_hyperparams_from_mapping,
    sliding_window_mean,
)


class ProbeModuleTest(unittest.TestCase):
    def test_all_probes_forward_and_score(self):
        torch.manual_seed(0)
        acts = torch.randn(3, 20, 2, 4, dtype=torch.bfloat16)
        mask = torch.ones(3, 20, dtype=torch.bool)
        mask[1, :3] = False
        labels = torch.tensor([0.0, 1.0, 0.0])
        hp = ProbeHyperparams(M=4, K=2, rmattn_window=4, rmattn_hidden=5)

        for name in PROBE_NAMES:
            with self.subTest(name=name):
                probe = build_probe(name, 2, 4, hp)
                loss, out = probe(acts, mask, labels, hp)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                score = probe.score(acts, mask, hp)
                logits = probe.score_logits(acts, mask, hp)
                self.assertEqual(tuple(score.shape), (3,))
                self.assertEqual(tuple(logits.shape), (3,))
                self.assertTrue(torch.all((score >= 0) & (score <= 1)))
                self.assertTrue(torch.allclose(score, torch.sigmoid(logits)))
                self.assertIn("scores", out)

    def test_window_fallback_for_short_sequences(self):
        acts = torch.randn(2, 3, 2, 4, dtype=torch.bfloat16)
        mask = torch.ones(2, 3, dtype=torch.bool)
        labels = torch.tensor([0.0, 1.0])
        hp = ProbeHyperparams(M=8, K=4, rmattn_window=8, rmattn_hidden=5)

        for name in ("swim", "sctopk", "rmattn"):
            with self.subTest(name=name):
                probe = build_probe(name, 2, 4, hp)
                loss, out = probe(acts, mask, labels, hp)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(out["fallback_fraction"], 1.0)

    def test_rmattn_multimax_eval_aggregation(self):
        torch.manual_seed(0)
        acts = torch.randn(2, 12, 2, 4, dtype=torch.bfloat16)
        mask = torch.ones(2, 12, dtype=torch.bool)
        hp = ProbeHyperparams(
            rmattn_window=4,
            rmattn_hidden=5,
            rmattn_heads=3,
            rmattn_eval_aggregation="multimax",
        )
        probe = build_probe("rmattn", 2, 4, hp)

        scores = probe.score(acts, mask, hp)

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))

    def test_attention_logits_are_unscaled_by_default(self):
        probe = build_probe("attention", 2, 8, ProbeHyperparams())
        acts = torch.ones((1, 3, 2, 8), dtype=torch.float32)
        with torch.no_grad():
            probe.W_m.fill_(1.0)
            probe.b_m.zero_()

        logits = probe.attention_logits(acts)

        self.assertTrue(torch.allclose(logits, torch.full((1, 3), 16.0)))

    def test_legacy_attention_scale_checkpoint_field_is_preserved(self):
        hp = probe_hyperparams_from_mapping(
            {
                "M": 32,
                "attention_logit_scale": "sqrt_feature_dim",
            }
        )

        self.assertEqual(hp.M, 32)
        self.assertEqual(hp.attention_logit_scale, "sqrt_feature_dim")

    def test_sliding_windows_require_every_token_to_be_valid(self):
        logits = torch.tensor([[10.0, 2.0, 4.0, 6.0]])
        mask = torch.tensor([[False, True, True, True]])

        means, valid = sliding_window_mean(logits, mask, width=2)

        self.assertEqual(valid.tolist(), [[False, False, True, True]])
        self.assertEqual(means[valid].tolist(), [3.0, 5.0])

    def test_streaming_probes_apply_ema_to_raw_token_logits(self):
        acts = torch.tensor([[[[0.0]], [[0.0]], [[4.0]], [[4.0]]]])
        mask = torch.ones((1, 4), dtype=torch.bool)
        hp = ProbeHyperparams(M=2, gamma_ema=0.5, streaming_reduction="max")

        for name in ("swim", "sctopk"):
            with self.subTest(name=name):
                probe = build_probe(name, 1, 1, hp)
                with torch.no_grad():
                    probe.W.fill_(1.0)
                    probe.b.zero_()
                token_logits = probe.token_logits(acts)
                expected = ema_streaming_scores(
                    token_logits,
                    mask,
                    hp.gamma_ema,
                    hp.streaming_reduction,
                    fallback_logits=masked_mean(token_logits, mask),
                )
                window_logits, valid = sliding_window_mean(token_logits, mask, hp.M)
                window_mean_result = ema_streaming_scores(
                    window_logits,
                    valid,
                    hp.gamma_ema,
                    hp.streaming_reduction,
                    fallback_logits=masked_mean(token_logits, mask),
                )

                self.assertTrue(torch.allclose(probe.score(acts, mask, hp), expected))
                self.assertTrue(
                    torch.allclose(
                        probe.score_logits(acts, mask, hp),
                        ema_streaming_logits(
                            token_logits,
                            mask,
                            hp.gamma_ema,
                            hp.streaming_reduction,
                            fallback_logits=masked_mean(token_logits, mask),
                        ),
                    )
                )
                self.assertFalse(torch.allclose(expected, window_mean_result))


if __name__ == "__main__":
    unittest.main()
