import json
import tempfile
import unittest
from pathlib import Path

from src.data import DataSettings, LabelledTextProbeDataset, load_dataset_split, normalize_turns_for_chat_template


class FakeTokenizer:
    all_special_ids = []
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False, return_dict=True):
        self.calls.append((turns, add_generation_prompt))
        input_ids = list(range(10, 10 + len(turns)))
        if add_generation_prompt:
            input_ids.append(99)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class StrictGemmaLikeTokenizer(FakeTokenizer):
    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False, return_dict=True):
        for turn in turns:
            if turn["role"] not in {"user", "assistant"}:
                raise ValueError(f"unsupported role {turn['role']}")
        expected = "user"
        for turn in turns:
            if turn["role"] != expected:
                raise ValueError("conversation roles must alternate user/assistant")
            expected = "assistant" if expected == "user" else "user"
        return super().apply_chat_template(turns, tokenize, add_generation_prompt, return_dict)


class QwenLikeTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.template_kwargs = []

    def apply_chat_template(
        self,
        turns,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        **kwargs,
    ):
        self.template_kwargs.append(kwargs)
        return super().apply_chat_template(turns, tokenize, add_generation_prompt, return_dict)


class LabelledTextDatasetTest(unittest.TestCase):
    def test_plain_text_and_json_dialogues_share_encoded_schema(self):
        tokenizer = FakeTokenizer()
        rows = [
            {"ids": "plain", "inputs": "plain training prompt", "labels": "high-stakes"},
            {
                "ids": "dialogue",
                "inputs": json.dumps(
                    [
                        {"role": "system", "content": "system prompt"},
                        {"role": "user", "content": "hello"},
                    ]
                ),
                "labels": "low-stakes",
            },
            {"ids": "skip", "inputs": "ambiguous sample", "labels": "ambiguous"},
        ]
        settings = DataSettings(
            format="labelled_text",
            id_field="ids",
            input_field="inputs",
            label_field="labels",
            drop_label_values=("ambiguous",),
        )

        dataset = LabelledTextProbeDataset("train", rows, settings, tokenizer)

        self.assertEqual(len(dataset), 2)
        self.assertEqual([example.label for example in dataset.examples], [1, 0])
        self.assertEqual(dataset.dropped_labels, 1)
        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": "plain training prompt"}])
        self.assertEqual([turn["role"] for turn in tokenizer.calls[1][0]], ["user"])
        self.assertIn("SYSTEM: system prompt", tokenizer.calls[1][0][0]["content"])
        self.assertIn("USER: hello", tokenizer.calls[1][0][0]["content"])

    def test_json_dialogues_are_normalized_for_gemma_chat_templates(self):
        tokenizer = StrictGemmaLikeTokenizer()
        rows = [
            {
                "ids": "assistant-first",
                "inputs": json.dumps(
                    [
                        {"role": "system", "content": "medical context"},
                        {"role": "assistant", "content": "hello patient"},
                        {"role": "user", "content": "hello doctor"},
                        {"role": "tool", "content": "tool result"},
                    ]
                ),
                "labels": "high-stakes",
            }
        ]
        settings = DataSettings(format="labelled_text", id_field="ids", input_field="inputs", label_field="labels")

        dataset = LabelledTextProbeDataset("test", rows, settings, tokenizer)

        self.assertEqual(len(dataset), 1)
        rendered_turns = tokenizer.calls[0][0]
        self.assertEqual([turn["role"] for turn in rendered_turns], ["user", "assistant", "user"])
        self.assertIn("SYSTEM: medical context", rendered_turns[0]["content"])
        self.assertIn("hello patient", rendered_turns[1]["content"])
        self.assertIn("TOOL: tool result", rendered_turns[2]["content"])

    def test_consecutive_assistant_turns_are_merged(self):
        turns = normalize_turns_for_chat_template(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "assistant", "content": "second answer"},
            ]
        )

        self.assertEqual([turn["role"] for turn in turns], ["user", "assistant"])
        self.assertIn("SYSTEM: system prompt", turns[0]["content"])
        self.assertIn("ASSISTANT: second answer", turns[1]["content"])

    def test_qwen_native_roles_and_non_thinking_template_kwargs(self):
        tokenizer = QwenLikeTokenizer()
        rows = [
            {"ids": "plain", "inputs": "unstructured high-stakes text", "labels": "high-stakes"},
            {
                "ids": "dialogue",
                "inputs": json.dumps(
                    [
                        {"role": "system", "content": "system prompt"},
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ]
                ),
                "labels": "low-stakes",
            },
        ]
        settings = DataSettings.from_config(
            {
                "data": {
                    "format": "labelled_text",
                    "id_field": "ids",
                    "input_field": "inputs",
                    "label_field": "labels",
                    "chat_template": {
                        "role_policy": "native",
                        "add_generation_prompt": False,
                        "kwargs": {"enable_thinking": False},
                    },
                }
            }
        )

        LabelledTextProbeDataset("train", rows, settings, tokenizer)

        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": "unstructured high-stakes text"}])
        self.assertEqual(
            [turn["role"] for turn in tokenizer.calls[1][0]],
            ["system", "user", "assistant"],
        )
        self.assertEqual(tokenizer.template_kwargs, [{"enable_thinking": False}] * 2)
        self.assertTrue(all(add_generation_prompt is False for _turns, add_generation_prompt in tokenizer.calls))

    def test_load_dataset_split_dispatches_labelled_text(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "split.jsonl"
            path.write_text(
                json.dumps({"ids": "x", "inputs": "hello", "labels": "low-stakes"}) + "\n",
                encoding="utf-8",
            )
            settings = DataSettings(
                format="labelled_text",
                files={"dev": str(path)},
                id_field="ids",
                input_field="inputs",
                label_field="labels",
            )

            dataset = load_dataset_split("dev", settings, tokenizer)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.examples[0].exchange_id, "x")
        self.assertEqual(dataset.examples[0].label, 0)

    def test_load_dataset_split_concatenates_labelled_text_file_lists(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "first.jsonl"
            second = Path(tmp_dir) / "second.jsonl"
            first.write_text(
                json.dumps({"ids": "a", "inputs": "hello", "labels": "low-stakes"}) + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"ids": "b", "inputs": "urgent", "labels": "high-stakes"}) + "\n",
                encoding="utf-8",
            )
            settings = DataSettings(
                format="labelled_text",
                files={"combined": [str(first), str(second)]},
                id_field="ids",
                input_field="inputs",
                label_field="labels",
            )

            dataset = load_dataset_split("combined", settings, tokenizer)

        self.assertEqual([example.exchange_id for example in dataset.examples], ["a", "b"])
        self.assertEqual([example.label for example in dataset.examples], [0, 1])


if __name__ == "__main__":
    unittest.main()
