"""Data utilities for generative YES/NO classifier finetuning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from src.data import (
    DataSettings,
    build_turns,
    data_format,
    infer_id_field,
    infer_input_field,
    infer_label_field,
    infer_shared_id_field,
    load_jsonl,
    load_jsonl_many,
    map_label,
    should_drop_label,
    turns_from_labelled_input,
)


@dataclass(frozen=True)
class ClassifierExample:
    row_key: str
    example_id: str
    label: int
    target: str
    prompt_turns: list[dict[str, str]]
    full_turns: list[dict[str, str]]
    split: str
    truncated: bool
    row: dict[str, Any]


@dataclass(frozen=True)
class TokenizedClassifierExample:
    row_key: str
    example_id: str
    label: int
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    length: int
    truncated: bool


@dataclass(frozen=True)
class CompletionFeature:
    row_key: str
    example_id: str
    label: int
    target_name: str
    input_ids: list[int]
    attention_mask: list[int]
    target_mask: list[bool]


def read_system_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def label_to_target(label: int, positive_target: str = "YES", negative_target: str = "NO") -> str:
    return positive_target if int(label) == 1 else negative_target


def render_conversation(turns: Sequence[Mapping[str, str]]) -> str:
    lines = ["Conversation to classify:"]
    for turn in turns:
        role = str(turn.get("role", "user")).upper()
        content = str(turn.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def classifier_prompt_turns(system_prompt: str, source_turns: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": render_conversation(source_turns)},
    ]


def chat_template_ids(
    tokenizer: Any,
    turns: list[dict[str, str]],
    add_generation_prompt: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    template_kwargs = dict(chat_template_kwargs or {})
    try:
        encoded = tokenizer.apply_chat_template(
            turns,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            **template_kwargs,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            turns,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(item) for item in input_ids]


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def full_and_target_ids(
    tokenizer: Any,
    prompt_turns: list[dict[str, str]],
    target: str,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], int, list[int]]:
    prompt_ids = chat_template_ids(
        tokenizer,
        prompt_turns,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_turns = prompt_turns + [{"role": "assistant", "content": target}]
    full_ids = chat_template_ids(
        tokenizer,
        full_turns,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    label_start = len(prompt_ids) if full_ids[: len(prompt_ids)] == prompt_ids else common_prefix_len(prompt_ids, full_ids)
    target_ids = full_ids[label_start:]
    if not target_ids:
        raise ValueError(f"Could not resolve target tokens for target={target!r}")
    return full_ids, label_start, target_ids


def preserve_target_truncate(
    input_ids: list[int],
    label_start: int,
    max_seq_len: int,
) -> tuple[list[int], int, bool]:
    if len(input_ids) <= max_seq_len:
        return input_ids, label_start, False
    target_ids = input_ids[label_start:]
    prompt_ids = input_ids[:label_start]
    keep_prompt = max(0, int(max_seq_len) - len(target_ids))
    if keep_prompt <= 0:
        kept = target_ids[-int(max_seq_len) :]
        return kept, 0, True
    kept_prompt = prompt_ids[-keep_prompt:]
    kept = kept_prompt + target_ids
    return kept, len(kept_prompt), True


def supervised_features(
    example: ClassifierExample,
    tokenizer: Any,
    max_seq_len: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> TokenizedClassifierExample:
    full_ids, label_start, _target_ids = full_and_target_ids(
        tokenizer,
        example.prompt_turns,
        example.target,
        chat_template_kwargs=chat_template_kwargs,
    )
    input_ids, label_start, truncated = preserve_target_truncate(full_ids, label_start, int(max_seq_len))
    labels = [-100] * label_start + input_ids[label_start:]
    attention_mask = [1] * len(input_ids)
    return TokenizedClassifierExample(
        row_key=example.row_key,
        example_id=example.example_id,
        label=example.label,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        length=len(input_ids),
        truncated=example.truncated or truncated,
    )


def completion_features(
    example: ClassifierExample,
    tokenizer: Any,
    target: str,
    target_name: str,
    max_seq_len: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> CompletionFeature:
    full_ids, label_start, _target_ids = full_and_target_ids(
        tokenizer,
        example.prompt_turns,
        target,
        chat_template_kwargs=chat_template_kwargs,
    )
    input_ids, label_start, _truncated = preserve_target_truncate(full_ids, label_start, int(max_seq_len))
    attention_mask = [1] * len(input_ids)
    target_mask = [False] * label_start + [True] * (len(input_ids) - label_start)
    return CompletionFeature(
        row_key=example.row_key,
        example_id=example.example_id,
        label=example.label,
        target_name=target_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_mask=target_mask,
    )


def _make_example(
    split: str,
    row_key: str,
    example_id: str,
    row: dict[str, Any],
    label: int,
    source_turns: list[dict[str, str]],
    system_prompt: str,
    positive_target: str,
    negative_target: str,
) -> ClassifierExample:
    prompt_turns = classifier_prompt_turns(system_prompt, source_turns)
    target = label_to_target(label, positive_target=positive_target, negative_target=negative_target)
    return ClassifierExample(
        row_key=row_key,
        example_id=example_id,
        label=label,
        target=target,
        prompt_turns=prompt_turns,
        full_turns=prompt_turns + [{"role": "assistant", "content": target}],
        split=split,
        truncated=False,
        row=row,
    )


def load_labelled_text_examples(
    split: str,
    rows: Sequence[dict[str, Any]],
    settings: DataSettings,
    system_prompt: str,
    positive_target: str,
    negative_target: str,
    limit: int | None = None,
) -> list[ClassifierExample]:
    input_field = settings.input_field or infer_input_field(rows)
    label_field = settings.label_field or infer_label_field(rows)
    id_field = settings.id_field or infer_id_field(rows)
    examples: list[ClassifierExample] = []
    dropped_labels = 0
    missing_inputs = 0
    for index, row in enumerate(rows):
        if input_field not in row or row[input_field] is None:
            missing_inputs += 1
            continue
        raw_label = row[label_field]
        if should_drop_label(raw_label, settings):
            dropped_labels += 1
            continue
        label = map_label(raw_label, settings.label_mapping)
        source_turns = turns_from_labelled_input(row[input_field], settings.chat_template_mode)
        example_id = str(row[id_field]) if id_field and row.get(id_field) is not None else f"{split}:{index}"
        examples.append(
            _make_example(
                split,
                f"{split}:row-{index}",
                example_id,
                row,
                label,
                source_turns,
                system_prompt,
                positive_target,
                negative_target,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    print(
        "classifier_data",
        json.dumps(
            {
                "split": split,
                "format": "labelled_text",
                "total": len(examples),
                "positive": sum(example.label for example in examples),
                "negative": len(examples) - sum(example.label for example in examples),
                "positive_rate": (sum(example.label for example in examples) / len(examples)) if examples else math.nan,
                "dropped_labels": dropped_labels,
                "missing_inputs": missing_inputs,
                "id_field": id_field,
                "input_field": input_field,
                "label_field": label_field,
            },
            sort_keys=True,
        ),
    )
    return examples


def load_exchange_annotation_examples(
    split: str,
    exchange_rows: Sequence[dict[str, Any]],
    annotation_rows: Sequence[dict[str, Any]],
    settings: DataSettings,
    system_prompt: str,
    positive_target: str,
    negative_target: str,
    limit: int | None = None,
) -> list[ClassifierExample]:
    exchange_id_field, annotation_id_field = (
        (settings.id_field, settings.annotation_id_field or settings.id_field)
        if settings.id_field
        else infer_shared_id_field(exchange_rows, annotation_rows)
    )
    if exchange_id_field is None or annotation_id_field is None:
        raise ValueError("id fields could not be resolved")
    label_field = settings.label_field or infer_label_field(annotation_rows)
    annotation_by_id = {str(row[annotation_id_field]): row for row in annotation_rows}
    examples: list[ClassifierExample] = []
    missing_annotations = 0
    dropped_labels = 0
    for index, row in enumerate(exchange_rows):
        example_id = str(row[exchange_id_field])
        annotation = annotation_by_id.get(example_id)
        if annotation is None:
            missing_annotations += 1
            continue
        raw_label = annotation[label_field]
        if should_drop_label(raw_label, settings):
            dropped_labels += 1
            continue
        label = map_label(raw_label, settings.label_mapping)
        examples.append(
            _make_example(
                split,
                f"{split}:row-{index}",
                example_id,
                row,
                label,
                build_turns(row),
                system_prompt,
                positive_target,
                negative_target,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    print(
        "classifier_data",
        json.dumps(
            {
                "split": split,
                "format": "exchange_annotations",
                "total": len(examples),
                "positive": sum(example.label for example in examples),
                "negative": len(examples) - sum(example.label for example in examples),
                "positive_rate": (sum(example.label for example in examples) / len(examples)) if examples else math.nan,
                "dropped_labels": dropped_labels,
                "missing_annotations": missing_annotations,
                "id_field": exchange_id_field,
                "annotation_id_field": annotation_id_field,
                "label_field": label_field,
            },
            sort_keys=True,
        ),
    )
    return examples


def load_classifier_examples(
    split: str,
    settings: DataSettings,
    system_prompt: str,
    positive_target: str = "YES",
    negative_target: str = "NO",
    limit: int | None = None,
) -> list[ClassifierExample]:
    if data_format(settings) == "labelled_text":
        rows = load_jsonl_many(settings.files[split])
        return load_labelled_text_examples(
            split,
            rows,
            settings,
            system_prompt,
            positive_target,
            negative_target,
            limit=limit,
        )
    exchange_rows = load_jsonl(settings.exchange_files[split])
    annotation_rows = load_jsonl(settings.annotation_files[split])
    return load_exchange_annotation_examples(
        split,
        exchange_rows,
        annotation_rows,
        settings,
        system_prompt,
        positive_target,
        negative_target,
        limit=limit,
    )


class TokenizedClassifierDataset(Dataset[TokenizedClassifierExample]):
    def __init__(
        self,
        examples: Sequence[ClassifierExample],
        tokenizer: Any,
        max_seq_len: int,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.examples = [
            supervised_features(
                example,
                tokenizer,
                max_seq_len=max_seq_len,
                chat_template_kwargs=chat_template_kwargs,
            )
            for example in examples
        ]
        self.truncations = sum(example.truncated for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedClassifierExample:
        return self.examples[index]


def collate_classifier_features(
    examples: Sequence[TokenizedClassifierExample],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    max_len = max(example.length for example in examples)
    batch = len(examples)
    input_ids = torch.full((batch, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    labels = torch.full((batch, max_len), -100, dtype=torch.long)
    class_labels = torch.tensor([example.label for example in examples], dtype=torch.long)
    for row, example in enumerate(examples):
        length = example.length
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row, :length] = torch.tensor(example.attention_mask, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "class_labels": class_labels,
        "row_keys": [example.row_key for example in examples],
        "example_ids": [example.example_id for example in examples],
    }


def collate_completion_features(
    examples: Sequence[CompletionFeature],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    max_len = max(len(example.input_ids) for example in examples)
    batch = len(examples)
    input_ids = torch.full((batch, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    target_mask = torch.zeros((batch, max_len), dtype=torch.bool)
    class_labels = torch.tensor([example.label for example in examples], dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row, :length] = torch.tensor(example.attention_mask, dtype=torch.long)
        target_mask[row, :length] = torch.tensor(example.target_mask, dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "class_labels": class_labels,
        "row_keys": [example.row_key for example in examples],
        "example_ids": [example.example_id for example in examples],
        "target_names": [example.target_name for example in examples],
    }
