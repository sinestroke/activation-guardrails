import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from src.extraction import should_use_nnsight_trace
from src.models import (
    GenericAdapter,
    Olmo3Adapter,
    adapter_for_config,
    tokenizer_metadata,
    validate_tokenizer_model_compatibility,
)


class ModelWrappingTest(unittest.TestCase):
    def test_olmo3_config_uses_explicit_adapter(self):
        config = SimpleNamespace(model_type="olmo3", architectures=["Olmo3ForCausalLM"])

        adapter = adapter_for_config(config)

        self.assertIsInstance(adapter, Olmo3Adapter)
        self.assertEqual(adapter.name, "olmo3")

    def test_tokenizer_metadata_records_template_hash(self):
        class FakeTokenizer:
            name_or_path = "allenai/Olmo-3-7B-Think-SFT"
            vocab_size = 100
            chat_template = "{{ messages }}"
            init_kwargs = {"_commit_hash": "tokenizer-commit"}
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 2
            unk_token_id = 0

            def __len__(self):
                return 100

        metadata = tokenizer_metadata(
            FakeTokenizer(),
            requested_tokenizer_name="allenai/Olmo-3-7B-Think-SFT",
        )

        self.assertEqual(metadata["resolved_commit_hash"], "tokenizer-commit")
        self.assertEqual(len(metadata["chat_template_sha256"]), 64)
        self.assertEqual(metadata["length"], 100)

    def test_tokenizer_larger_than_model_vocab_is_rejected(self):
        class FakeTokenizer:
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 2
            unk_token_id = 0

            def __len__(self):
                return 101

        with self.assertRaisesRegex(ValueError, "accepts only 100"):
            validate_tokenizer_model_compatibility(
                FakeTokenizer(),
                SimpleNamespace(vocab_size=100),
                tokenizer_name="trajectory-tokenizer",
                model_name="base-model",
            )

    def test_hf_reference_is_not_registered_as_child_module(self):
        wrapper = torch.nn.Module()
        hf_model = torch.nn.Linear(2, 2)

        wrapper.__dict__["_codex_hf_model"] = hf_model
        wrapper.eval()

        self.assertIs(wrapper._codex_hf_model, hf_model)
        self.assertNotIn("_codex_hf_model", wrapper._modules)

    def test_wrapped_hf_model_uses_hooks_instead_of_nnsight_trace(self):
        class Wrapper:
            _codex_wrapped_hf_text_only = True

            def trace(self, _inputs):
                raise AssertionError("trace should not be used")

        self.assertFalse(should_use_nnsight_trace(Wrapper()))

    def test_target_model_adapter_is_loaded_at_requested_revision_and_recorded(self):
        base_model = torch.nn.Linear(2, 2)
        calls = {}

        class FakeWrappedModel:
            def __init__(self, model):
                self.model = model
                self.peft_config = {
                    "default": SimpleNamespace(
                        base_model_name_or_path="google/gemma-3-12b-it",
                        _commit_hash="adapter-commit",
                    )
                }

            def merge_and_unload(self, safe_merge=True):
                calls["safe_merge"] = safe_merge
                return self.model

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, repo_id, **kwargs):
                calls.update({"model": model, "repo_id": repo_id, **kwargs})
                return FakeWrappedModel(model)

        fake_peft = ModuleType("peft")
        fake_peft.PeftModel = FakePeftModel
        runtime_config = {}
        spec = {
            "repo_id": "org/adapter",
            "revision": "epoch-3",
            "merge_and_unload": True,
            "safe_merge": True,
        }

        with patch.dict("sys.modules", {"peft": fake_peft}):
            loaded = GenericAdapter()._apply_target_model_adapter(
                base_model,
                spec,
                token="secret",
                local_files_only=False,
                runtime_config=runtime_config,
            )

        self.assertIs(loaded, base_model)
        self.assertEqual(calls["repo_id"], "org/adapter")
        self.assertEqual(calls["revision"], "epoch-3")
        self.assertEqual(calls["token"], "secret")
        self.assertTrue(calls["safe_merge"])
        self.assertEqual(
            runtime_config["resolved_target_model_adapter"],
            {
                "repo_id": "org/adapter",
                "revision": "epoch-3",
                "merge_and_unload": True,
                "safe_merge": True,
                "resolved_base_model_name_or_path": "google/gemma-3-12b-it",
                "resolved_commit_hash": "adapter-commit",
            },
        )


if __name__ == "__main__":
    unittest.main()
