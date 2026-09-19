import hashlib
import unittest
from pathlib import Path

from src.classifier_train import load_config as load_classifier_config
from src.data import DataSettings
from src.eval import evaluation_config, select_eval_splits
from src.train import load_config as load_probe_config


ROOT = Path(__file__).resolve().parents[1]


class ReleaseConfigTest(unittest.TestCase):
    def test_high_stakes_classifier_is_llama_1b(self):
        path = ROOT / "configs/classifiers/high-stakes-llama3.2-1b-vanilla.yaml"
        config = load_classifier_config(path)
        self.assertEqual(config["model_name"], "meta-llama/Llama-3.2-1B-Instruct")
        self.assertEqual(config["classifier_model_size"], "1b")
        self.assertEqual(config["classifier"]["system_prompt_path"], "constitutions/high-stakes-classifier.md")

    def test_chemical_classifier_uses_exact_release_constitution(self):
        path = ROOT / "configs/classifiers/chemical-harm-llama3.2-1b-instruct.yaml"
        config = load_classifier_config(path)
        self.assertEqual(config["classifier"]["system_prompt_path"], "constitutions/chemical-harm-classifier.md")
        constitution = ROOT / "constitutions/chemical-harm-classifier.md"
        self.assertTrue(constitution.is_file())
        self.assertEqual(
            hashlib.sha256(constitution.read_bytes()).hexdigest(),
            "31c9ef968e922d1aff969ce13a4d24a1fb7cfb443de1dd525d376ab64b66c23a",
        )

    def test_high_stakes_constitutions_are_both_present(self):
        for filename in ("high-stakes-classifier.md", "high-stakes-classifier-scale.md"):
            constitution = ROOT / "constitutions" / filename
            self.assertTrue(constitution.is_file())
            self.assertEqual(
                hashlib.sha256(constitution.read_bytes()).hexdigest(),
                "7c94275595195e6a25c8818475a4f56c3a7742031b86043b3b39c6c113d89b58",
            )

    def test_chemical_eval_includes_language_and_everyday_splits(self):
        path = ROOT / "configs/evaluation/chemical-harm.yaml"
        config = load_probe_config(path)
        splits = select_eval_splits(DataSettings.from_config(config), evaluation_config(config))
        self.assertIn("internal_eval_bn", splits)
        self.assertIn("internal_eval_hi", splits)
        self.assertIn("internal_eval_everyday", splits)

    def test_high_stakes_eval_uses_six_standard_splits(self):
        path = ROOT / "configs/evaluation/high-stakes.yaml"
        config = load_probe_config(path)
        splits = select_eval_splits(DataSettings.from_config(config), evaluation_config(config))
        self.assertEqual(len(splits), 6)
        self.assertFalse(any("keyword" in split for split in splits))


if __name__ == "__main__":
    unittest.main()
