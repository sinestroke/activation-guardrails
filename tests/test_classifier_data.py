import json
import unittest

import torch

from src.classifier_data import (
    TokenizedClassifierDataset,
    completion_features,
    load_classifier_examples,
    load_labelled_text_examples,
)
from src.classifier_train import classification_scores_from_logprobs, sequence_log_likelihood
from src.data import DataSettings


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False, return_dict=True):
        input_ids = [2]
        for turn in turns:
            role = turn["role"]
            content = turn["content"]
            if role == "system":
                input_ids.extend([100, len(content) % 50])
            elif role == "user":
                input_ids.extend([101, len(content) % 50])
            elif role == "assistant":
                input_ids.append(102)
                if content == "YES":
                    input_ids.append(201)
                elif content == "NO":
                    input_ids.append(202)
                else:
                    input_ids.append(203)
                input_ids.append(3)
            else:
                input_ids.extend([104, len(content) % 50])
        if add_generation_prompt:
            input_ids.append(102)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class ThinkingAwareTokenizer(FakeChatTokenizer):
    def __init__(self):
        self.thinking_values = []

    def apply_chat_template(
        self,
        turns,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        enable_thinking=True,
    ):
        self.thinking_values.append(enable_thinking)
        return super().apply_chat_template(
            turns,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )


class ClassifierDataTest(unittest.TestCase):
    def test_labelled_text_examples_inject_judge_system_prompt(self):
        rows = [
            {"ids": "plain", "inputs": "plain training prompt", "labels": "high-stakes"},
            {
                "ids": "dialogue",
                "inputs": json.dumps(
                    [
                        {"role": "system", "content": "original system"},
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

        examples = load_labelled_text_examples(
            "train",
            rows,
            settings,
            system_prompt="Return YES or NO only.",
            positive_target="YES",
            negative_target="NO",
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].target, "YES")
        self.assertEqual(examples[1].target, "NO")
        self.assertEqual(examples[0].prompt_turns[0], {"role": "system", "content": "Return YES or NO only."})
        self.assertIn("Conversation to classify:", examples[0].prompt_turns[1]["content"])
        self.assertIn("USER: plain training prompt", examples[0].prompt_turns[1]["content"])
        self.assertIn("SYSTEM: original system", examples[1].prompt_turns[1]["content"])

    def test_supervised_labels_are_masked_until_answer(self):
        tokenizer = FakeChatTokenizer()
        rows = [{"ids": "x", "inputs": "hello", "labels": "high-stakes"}]
        settings = DataSettings(format="labelled_text", id_field="ids", input_field="inputs", label_field="labels")
        examples = load_labelled_text_examples(
            "train",
            rows,
            settings,
            system_prompt="Return YES or NO only.",
            positive_target="YES",
            negative_target="NO",
        )

        dataset = TokenizedClassifierDataset(examples, tokenizer, max_seq_len=1024)
        item = dataset[0]

        self.assertIn(201, item.input_ids)
        answer_start = item.input_ids.index(201)
        self.assertTrue(all(label == -100 for label in item.labels[:answer_start]))
        self.assertEqual(item.labels[answer_start:], item.input_ids[answer_start:])

    def test_completion_features_mark_only_target_tokens(self):
        tokenizer = FakeChatTokenizer()
        rows = [{"ids": "x", "inputs": "hello", "labels": "low-stakes"}]
        settings = DataSettings(format="labelled_text", id_field="ids", input_field="inputs", label_field="labels")
        example = load_labelled_text_examples(
            "dev",
            rows,
            settings,
            system_prompt="Return YES or NO only.",
            positive_target="YES",
            negative_target="NO",
        )[0]

        feature = completion_features(example, tokenizer, target="YES", target_name="positive", max_seq_len=1024)

        self.assertEqual(feature.target_name, "positive")
        self.assertEqual(sum(feature.target_mask), 2)
        self.assertEqual(feature.input_ids[-2:], [201, 3])

    def test_completion_features_forward_chat_template_kwargs(self):
        tokenizer = ThinkingAwareTokenizer()
        rows = [{"ids": "x", "inputs": "hello", "labels": "low-stakes"}]
        settings = DataSettings(format="labelled_text", id_field="ids", input_field="inputs", label_field="labels")
        example = load_labelled_text_examples(
            "dev",
            rows,
            settings,
            system_prompt="Return YES or NO only.",
            positive_target="YES",
            negative_target="NO",
        )[0]

        completion_features(
            example,
            tokenizer,
            target="YES",
            target_name="positive",
            max_seq_len=1024,
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(tokenizer.thinking_values, [False, False])

    def test_load_classifier_examples_concatenates_labelled_text_file_lists(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "first.jsonl"
            second = Path(tmp_dir) / "second.jsonl"
            first.write_text(json.dumps({"ids": "a", "inputs": "hello", "labels": "low-stakes"}) + "\n", encoding="utf-8")
            second.write_text(json.dumps({"ids": "b", "inputs": "urgent", "labels": "high-stakes"}) + "\n", encoding="utf-8")
            settings = DataSettings(
                format="labelled_text",
                files={"combined": [str(first), str(second)]},
                id_field="ids",
                input_field="inputs",
                label_field="labels",
            )

            examples = load_classifier_examples("combined", settings, "Return YES or NO only.")

        self.assertEqual([example.example_id for example in examples], ["a", "b"])
        self.assertEqual([example.label for example in examples], [0, 1])

    def test_duplicate_external_ids_receive_unique_internal_row_keys(self):
        rows = [
            {"ids": "reused", "inputs": "first", "labels": "low-stakes"},
            {"ids": "reused", "inputs": "second", "labels": "high-stakes"},
        ]
        settings = DataSettings(format="labelled_text", id_field="ids", input_field="inputs", label_field="labels")

        examples = load_labelled_text_examples(
            "dev",
            rows,
            settings,
            system_prompt="Return YES or NO only.",
            positive_target="YES",
            negative_target="NO",
        )

        self.assertEqual([example.example_id for example in examples], ["reused", "reused"])
        self.assertEqual([example.row_key for example in examples], ["dev:row-0", "dev:row-1"])
        self.assertEqual(len({example.row_key for example in examples}), 2)

        logprobs = {
            "dev:row-0": {"positive": -3.0, "negative": -1.0},
            "dev:row-1": {"positive": -0.5, "negative": -2.5},
        }
        labels, scores, missing = classification_scores_from_logprobs(examples, logprobs)
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(scores.tolist(), [-2.0, 2.0])
        self.assertEqual(missing, 0)

    def test_sequence_log_likelihood_uses_shifted_target_mask(self):
        logits = torch.zeros((1, 4, 5), dtype=torch.float32)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        target_mask = torch.tensor([[False, False, True, True]])
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0

        value = sequence_log_likelihood(logits, input_ids, target_mask)

        self.assertGreater(float(value.item()), -0.001)


if __name__ == "__main__":
    unittest.main()
