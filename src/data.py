"""Schema-aware exchange/annotation loading and collation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler


DEFAULT_EXCHANGE_FILES: dict[str, str] = {}
DEFAULT_ANNOTATION_FILES: dict[str, str] = {}


@dataclass(frozen=True)
class DataSettings:
    format: str = "exchange_annotations"
    exchange_files: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_EXCHANGE_FILES))
    annotation_files: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ANNOTATION_FILES))
    files: Mapping[str, str | Sequence[str]] = field(default_factory=dict)
    id_field: str | None = None
    annotation_id_field: str | None = None
    input_field: str | None = None
    label_field: str | None = None
    label_mapping: Mapping[str, int] | None = None
    drop_label_values: tuple[str, ...] = ()
    probe_region: str = "all"
    max_seq_len: int = 8192
    prefer_stored_token_ids: bool = False
    chat_template_mode: str = "auto"
    chat_template_role_policy: str = "compatible"
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    add_generation_prompt: bool = False
    print_rows: int = 2

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DataSettings":
        data = config.get("data", {})
        chat_template = data.get("chat_template", {}) or {}
        return cls(
            format=str(data.get("format", "exchange_annotations")).replace("-", "_"),
            exchange_files=data.get("exchange_files", DEFAULT_EXCHANGE_FILES),
            annotation_files=data.get("annotation_files", DEFAULT_ANNOTATION_FILES),
            files=data.get("files", {}),
            id_field=data.get("id_field"),
            annotation_id_field=data.get("annotation_id_field"),
            input_field=data.get("input_field"),
            label_field=data.get("label_field", config.get("label_field")),
            label_mapping=data.get("label_mapping", config.get("label_mapping")),
            drop_label_values=tuple(normalize_label_value(value) for value in data.get("drop_label_values", ())),
            probe_region=config.get("probe_region", data.get("probe_region", "all")),
            max_seq_len=int(config.get("max_seq_len", data.get("max_seq_len", 8192))),
            prefer_stored_token_ids=bool(data.get("prefer_stored_token_ids", False)),
            chat_template_mode=str(chat_template.get("mode", data.get("chat_template_mode", "auto"))).replace("-", "_"),
            chat_template_role_policy=str(chat_template.get("role_policy", "compatible")).replace("-", "_").lower(),
            chat_template_kwargs=dict(chat_template.get("kwargs", {}) or {}),
            add_generation_prompt=bool(chat_template.get("add_generation_prompt", data.get("add_generation_prompt", False))),
            print_rows=int(data.get("print_rows", 2)),
        )


@dataclass
class EncodedExample:
    row_key: str
    exchange_id: str
    label: int
    input_ids: list[int]
    attention_mask: list[int]
    probe_mask: list[bool]
    length: int
    split: str
    prompt_id: str | None
    truncated: bool
    row: dict[str, Any]


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def data_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def load_jsonl_many(paths: str | Path | Sequence[str | Path], limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in data_paths(paths):
        remaining = None if limit is None else max(0, int(limit) - len(rows))
        if remaining == 0:
            break
        rows.extend(load_jsonl(path, limit=remaining))
    return rows


def compact_value(value: Any, max_items: int = 12, max_chars: int = 220) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
    if isinstance(value, list):
        if len(value) <= max_items:
            return [compact_value(item, max_items=max_items, max_chars=max_chars) for item in value]
        return [compact_value(item, max_items=max_items, max_chars=max_chars) for item in value[:max_items]] + [
            f"... ({len(value) - max_items} more)"
        ]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                compact["..."] = f"{len(value) - max_items} more keys"
                break
            compact[key] = compact_value(item, max_items=max_items, max_chars=max_chars)
        return compact
    return value


def data_format(settings: DataSettings) -> str:
    value = settings.format.replace("-", "_").lower()
    aliases = {
        "exchange_annotation": "exchange_annotations",
        "exchange_annotations": "exchange_annotations",
        "exchanges": "exchange_annotations",
        "labelled_text": "labelled_text",
        "labeled_text": "labelled_text",
        "single_file": "labelled_text",
    }
    if value not in aliases:
        raise ValueError(f"Unknown data.format={settings.format!r}")
    return aliases[value]


def available_splits(settings: DataSettings) -> list[str]:
    if data_format(settings) == "labelled_text":
        return list(settings.files.keys())
    return list(settings.exchange_files.keys())


def inspect_data_files(settings: DataSettings, splits: Sequence[str] | None = None) -> None:
    selected = list(splits or available_splits(settings))
    print("=== Data schema preview ===")
    for split in selected:
        if data_format(settings) == "labelled_text":
            for path in data_paths(settings.files[split]):
                rows = load_jsonl(path, limit=settings.print_rows)
                print(f"[{split}] labelled_text: {path}")
                for index, row in enumerate(rows, start=1):
                    print(json.dumps({"row": index, "keys": list(row.keys()), "preview": compact_value(row)}, sort_keys=True))
            continue

        exchange_path = settings.exchange_files[split]
        annotation_path = settings.annotation_files[split]
        for kind, path in (("exchange", exchange_path), ("annotation", annotation_path)):
            rows = load_jsonl(path, limit=settings.print_rows)
            print(f"[{split}] {kind}: {path}")
            for index, row in enumerate(rows, start=1):
                print(json.dumps({"row": index, "keys": list(row.keys()), "preview": compact_value(row)}, sort_keys=True))


def infer_shared_id_field(exchange_rows: Sequence[dict[str, Any]], annotation_rows: Sequence[dict[str, Any]]) -> tuple[str, str]:
    if not exchange_rows or not annotation_rows:
        raise ValueError("Cannot infer id fields from empty data")
    exchange_keys = set(exchange_rows[0])
    annotation_keys = set(annotation_rows[0])
    for candidate in ("exchange_id", "id", "conversation_id", "sample_id", "prompt_id"):
        if candidate in exchange_keys and candidate in annotation_keys:
            return candidate, candidate
    for exchange_key in sorted(exchange_keys):
        if not exchange_key.endswith("_id"):
            continue
        exchange_values = {row.get(exchange_key) for row in exchange_rows[:50]}
        for annotation_key in sorted(annotation_keys):
            if not annotation_key.endswith("_id"):
                continue
            annotation_values = {row.get(annotation_key) for row in annotation_rows[:50]}
            if exchange_values & annotation_values:
                return exchange_key, annotation_key
    raise ValueError("Could not infer a shared id field between exchanges and annotations")


def infer_label_field(annotation_rows: Sequence[dict[str, Any]]) -> str:
    if not annotation_rows:
        raise ValueError("Cannot infer label field from empty annotations")
    keys = annotation_rows[0].keys()
    for candidate in ("harm_label", "labels", "label", "binary_label", "is_harmful", "harmful", "judge_label"):
        if candidate in keys:
            return candidate
    for key in keys:
        lowered = key.lower()
        if "label" in lowered or "harm" in lowered:
            return key
    raise ValueError("Could not infer annotation label field")


def normalize_label_value(value: Any) -> str:
    return str(value).strip().lower()


def should_drop_label(value: Any, settings: DataSettings) -> bool:
    return normalize_label_value(value) in settings.drop_label_values


def map_label(value: Any, mapping: Mapping[str, int] | None) -> int:
    if mapping:
        for key in (value, str(value), normalize_label_value(value)):
            if key in mapping:
                return int(mapping[key])
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    lowered = str(value).strip().lower()
    defaults = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "harmful": 1,
        "unsafe": 1,
        "high-stakes": 1,
        "high_stakes": 1,
        "high stakes": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "not_harmful": 0,
        "benign": 0,
        "safe": 0,
        "unharmful": 0,
        "low-stakes": 0,
        "low_stakes": 0,
        "low stakes": 0,
    }
    if lowered in defaults:
        return defaults[lowered]
    raise ValueError(f"Cannot map label value {value!r} to binary 0/1")


def get_nested(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def extract_token_ids(row: Mapping[str, Any]) -> list[int] | None:
    for keys in (
        ("full_sequence_token_ids",),
        ("tokenization", "full_sequence_token_ids"),
    ):
        value = get_nested(row, keys)
        if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
            return list(value)
    return None


def turns_from_sequence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    turns = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", item.get("speaker", "user")))
        content = str(item.get("content", item.get("text", "")))
        turns.append({"role": role, "content": content})
    return turns


def turns_from_labelled_input(value: Any, mode: str = "auto") -> list[dict[str, str]]:
    mode = mode.replace("-", "_").lower()
    if mode not in {"auto", "single_user", "json_messages"}:
        raise ValueError(f"Unknown chat_template.mode={mode!r}")
    if isinstance(value, list):
        turns = turns_from_sequence(value)
        if turns:
            return turns

    if isinstance(value, str) and mode in {"auto", "json_messages"}:
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            turns = turns_from_sequence(parsed)
            if turns:
                return turns
            if mode == "json_messages":
                raise ValueError("chat_template.mode=json_messages expected a JSON list of chat messages")

    return [{"role": "user", "content": str(value)}]


def render_context_turns(turns: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"{role.upper()}: {content}" for role, content in turns if content)


def merge_turn_content(existing: str, role: str, content: str) -> str:
    labelled = f"{role.upper()}: {content}" if content else role.upper()
    return f"{existing}\n\n{labelled}" if existing else labelled


def normalize_native_turns(turns: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    role_aliases = {
        "developer": "system",
        "human": "user",
        "model": "assistant",
        "bot": "assistant",
        "function": "tool",
    }
    supported = {"system", "user", "assistant", "tool"}
    for turn in turns:
        raw_role = str(turn.get("role", turn.get("speaker", "user"))).strip() or "user"
        role_key = raw_role.lower().replace("-", "_")
        role = role_aliases.get(role_key, role_key)
        content = str(turn.get("content", turn.get("text", "")))
        if role not in supported:
            content = f"{raw_role.upper()}: {content}"
            role = "user"
        normalized.append({"role": role, "content": content})
    return normalized or [{"role": "user", "content": ""}]


def normalize_turns_for_chat_template(
    turns: Sequence[Mapping[str, str]],
    role_policy: str = "compatible",
) -> list[dict[str, str]]:
    """Convert arbitrary role-labelled exchanges to user/assistant chat turns.

    Some target tokenizers, including Gemma-family chat templates, reject
    explicit system/tool roles or conversations whose first non-system message
    is an assistant message. Evaluation datasets can still be full exchanges,
    so system/tool role information is preserved as text inside user turns.
    """

    policy = str(role_policy).lower().replace("-", "_")
    if policy in {"native", "preserve_native"}:
        return normalize_native_turns(turns)
    if policy not in {"compatible", "gemma_compatible", "canonical"}:
        raise ValueError(f"Unknown chat_template.role_policy={role_policy!r}")

    normalized: list[dict[str, str]] = []
    pending_context: list[tuple[str, str]] = []

    def flush_context_as_user() -> None:
        nonlocal pending_context
        context = render_context_turns(pending_context)
        pending_context = []
        if not context:
            return
        if normalized and normalized[-1]["role"] == "user":
            normalized[-1]["content"] = merge_turn_content(normalized[-1]["content"], "context", context)
        else:
            normalized.append({"role": "user", "content": context})

    for turn in turns:
        raw_role = str(turn.get("role", turn.get("speaker", "user"))).strip() or "user"
        role_key = raw_role.lower().replace("-", "_")
        content = str(turn.get("content", turn.get("text", "")))

        if role_key in {"system", "developer"}:
            pending_context.append((raw_role, content))
            continue

        if role_key in {"assistant", "model", "bot"}:
            role = "assistant"
            rendered_content = content
            merge_content = content
        elif role_key in {"user", "human"}:
            role = "user"
            rendered_content = content
            merge_content = content
        else:
            # Tokenizers generally do not know about tool/function/domain roles.
            # Keep the role label in-band and let it occupy a user turn.
            role = "user"
            rendered_content = f"{raw_role.upper()}: {content}"
            merge_content = content

        if pending_context and role == "user":
            context = render_context_turns(pending_context)
            pending_context = []
            if context:
                rendered_content = merge_turn_content(context, raw_role, content)
                merge_content = rendered_content
        elif pending_context:
            flush_context_as_user()

        if role == "assistant" and not normalized:
            normalized.append({"role": "user", "content": "Conversation context:"})

        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = merge_turn_content(normalized[-1]["content"], raw_role, merge_content)
        else:
            normalized.append({"role": role, "content": rendered_content})

    flush_context_as_user()
    return normalized or [{"role": "user", "content": ""}]


def build_turns(row: Mapping[str, Any]) -> list[dict[str, str]]:
    for key in ("turns", "messages", "conversation"):
        value = row.get(key)
        turns = turns_from_sequence(value)
        if turns:
            return turns

    prompt = row.get("prompt", row.get("user", row.get("input", "")))
    response = row.get("response", row.get("assistant", row.get("output", "")))
    turns: list[dict[str, str]] = []
    system_prompt = row.get("system_prompt") or get_nested(row, ("generation_config", "system_prompt"))
    if system_prompt:
        turns.append({"role": "system", "content": str(system_prompt)})
    turns.append({"role": "user", "content": str(prompt)})
    if response:
        turns.append({"role": "assistant", "content": str(response)})
    return turns


def tokenizer_special_ids(tokenizer: Any, row: Mapping[str, Any]) -> set[int]:
    ids: set[int] = set()
    for attr in ("all_special_ids",):
        value = getattr(tokenizer, attr, None)
        if value:
            ids.update(int(item) for item in value if item is not None)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        for source in (row, row.get("tokenization", {})):
            if isinstance(source, Mapping) and source.get(key) is not None:
                ids.add(int(source[key]))
    return ids


def apply_chat_template(
    turns: list[dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
    role_policy: str = "compatible",
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    if tokenizer is None:
        raise ValueError("A tokenizer is required when prefer_stored_token_ids=false")
    turns = normalize_turns_for_chat_template(turns, role_policy=role_policy)
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
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
    else:
        input_ids = encoded
        attention_mask = [1] * len(input_ids)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(attention_mask, "tolist"):
        attention_mask = attention_mask.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    return [int(item) for item in input_ids], [int(item) for item in attention_mask]


def response_start(row: Mapping[str, Any]) -> int | None:
    for keys in (
        ("response_start_index",),
        ("tokenization", "response_start_index"),
        ("prompt_input_len",),
        ("tokenization", "prompt_input_len"),
    ):
        value = get_nested(row, keys)
        if value is not None:
            return int(value)
    return None


def build_probe_mask(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    row: Mapping[str, Any],
    tokenizer: Any,
    probe_region: str,
) -> list[bool]:
    special_ids = tokenizer_special_ids(tokenizer, row)
    mask = [bool(attn) and token_id not in special_ids for token_id, attn in zip(input_ids, attention_mask)]
    if probe_region == "all":
        return mask
    if probe_region != "assistant_only":
        raise ValueError(f"Unknown probe_region={probe_region!r}")
    start = response_start(row)
    if start is None:
        raise ValueError("assistant_only requires response_start_index or prompt_input_len in the exchange row")
    return [valid and index >= start for index, valid in enumerate(mask)]


class ExchangeProbeDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        split: str,
        exchange_rows: Sequence[dict[str, Any]],
        annotation_rows: Sequence[dict[str, Any]],
        settings: DataSettings,
        tokenizer: Any,
        limit: int | None = None,
    ) -> None:
        self.split = split
        exchange_id_field, annotation_id_field = (
            (settings.id_field, settings.annotation_id_field or settings.id_field)
            if settings.id_field
            else infer_shared_id_field(exchange_rows, annotation_rows)
        )
        if exchange_id_field is None or annotation_id_field is None:
            raise ValueError("id fields could not be resolved")
        label_field = settings.label_field or infer_label_field(annotation_rows)
        annotation_by_id = {str(row[annotation_id_field]): row for row in annotation_rows}

        examples: list[EncodedExample] = []
        truncations = 0
        missing_annotations = 0
        for index, row in enumerate(exchange_rows):
            exchange_id = str(row[exchange_id_field])
            annotation = annotation_by_id.get(exchange_id)
            if annotation is None:
                missing_annotations += 1
                continue
            label = map_label(annotation[label_field], settings.label_mapping)
            input_ids: list[int]
            attention_mask: list[int]
            stored_ids = extract_token_ids(row)
            if settings.prefer_stored_token_ids and stored_ids is not None:
                input_ids = stored_ids
                attention_mask = [1] * len(input_ids)
            else:
                input_ids, attention_mask = apply_chat_template(
                    build_turns(row),
                    tokenizer,
                    add_generation_prompt=settings.add_generation_prompt,
                    role_policy=settings.chat_template_role_policy,
                    chat_template_kwargs=settings.chat_template_kwargs,
                )

            truncated = len(input_ids) > settings.max_seq_len
            if truncated:
                truncations += 1
                input_ids = input_ids[: settings.max_seq_len]
                attention_mask = attention_mask[: settings.max_seq_len]
            probe_mask = build_probe_mask(input_ids, attention_mask, row, tokenizer, settings.probe_region)
            examples.append(
                EncodedExample(
                    row_key=f"{split}:row-{index}",
                    exchange_id=exchange_id,
                    label=label,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    probe_mask=probe_mask,
                    length=len(input_ids),
                    split=split,
                    prompt_id=str(row.get("prompt_id")) if row.get("prompt_id") is not None else None,
                    truncated=truncated,
                    row=row,
                )
            )
            if limit is not None and len(examples) >= limit:
                break

        self.examples = examples
        self.id_field = exchange_id_field
        self.annotation_id_field = annotation_id_field
        self.label_field = label_field
        self.truncations = truncations
        self.missing_annotations = missing_annotations

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]

    def class_balance(self) -> dict[str, Any]:
        positives = sum(example.label for example in self.examples)
        total = len(self.examples)
        negatives = total - positives
        return {
            "split": self.split,
            "total": total,
            "positive": positives,
            "negative": negatives,
            "positive_rate": positives / total if total else math.nan,
            "truncations": self.truncations,
            "missing_annotations": self.missing_annotations,
            "id_field": self.id_field,
            "annotation_id_field": self.annotation_id_field,
            "label_field": self.label_field,
        }


def infer_input_field(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot infer input field from empty data")
    keys = rows[0].keys()
    for candidate in ("inputs", "input", "prompt", "text", "messages", "turns", "conversation"):
        if candidate in keys:
            return candidate
    raise ValueError("Could not infer labelled-text input field")


def infer_id_field(rows: Sequence[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    keys = rows[0].keys()
    for candidate in ("ids", "id", "sample_id", "prompt_id", "exchange_id", "conversation_id"):
        if candidate in keys:
            return candidate
    return None


class LabelledTextProbeDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        split: str,
        rows: Sequence[dict[str, Any]],
        settings: DataSettings,
        tokenizer: Any,
        limit: int | None = None,
    ) -> None:
        self.split = split
        input_field = settings.input_field or infer_input_field(rows)
        label_field = settings.label_field or infer_label_field(rows)
        id_field = settings.id_field or infer_id_field(rows)

        examples: list[EncodedExample] = []
        truncations = 0
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
            input_ids, attention_mask = apply_chat_template(
                turns_from_labelled_input(row[input_field], settings.chat_template_mode),
                tokenizer,
                add_generation_prompt=settings.add_generation_prompt,
                role_policy=settings.chat_template_role_policy,
                chat_template_kwargs=settings.chat_template_kwargs,
            )

            truncated = len(input_ids) > settings.max_seq_len
            if truncated:
                truncations += 1
                input_ids = input_ids[: settings.max_seq_len]
                attention_mask = attention_mask[: settings.max_seq_len]
            probe_mask = build_probe_mask(input_ids, attention_mask, row, tokenizer, settings.probe_region)
            example_id = str(row[id_field]) if id_field and row.get(id_field) is not None else f"{split}:{index}"
            examples.append(
                EncodedExample(
                    row_key=f"{split}:row-{index}",
                    exchange_id=example_id,
                    label=label,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    probe_mask=probe_mask,
                    length=len(input_ids),
                    split=split,
                    prompt_id=str(row.get("prompt_id")) if row.get("prompt_id") is not None else None,
                    truncated=truncated,
                    row=row,
                )
            )
            if limit is not None and len(examples) >= limit:
                break

        self.examples = examples
        self.id_field = id_field
        self.input_field = input_field
        self.label_field = label_field
        self.truncations = truncations
        self.dropped_labels = dropped_labels
        self.missing_inputs = missing_inputs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]

    def class_balance(self) -> dict[str, Any]:
        positives = sum(example.label for example in self.examples)
        total = len(self.examples)
        negatives = total - positives
        return {
            "split": self.split,
            "total": total,
            "positive": positives,
            "negative": negatives,
            "positive_rate": positives / total if total else math.nan,
            "truncations": self.truncations,
            "dropped_labels": self.dropped_labels,
            "missing_inputs": self.missing_inputs,
            "id_field": self.id_field,
            "input_field": self.input_field,
            "label_field": self.label_field,
        }


def load_dataset_split(
    split: str,
    settings: DataSettings,
    tokenizer: Any,
    limit: int | None = None,
) -> ExchangeProbeDataset | LabelledTextProbeDataset:
    if data_format(settings) == "labelled_text":
        rows = load_jsonl_many(settings.files[split])
        dataset = LabelledTextProbeDataset(split, rows, settings, tokenizer, limit=limit)
        print("class_balance", json.dumps(dataset.class_balance(), sort_keys=True))
        return dataset

    exchanges = load_jsonl(settings.exchange_files[split])
    annotations = load_jsonl(settings.annotation_files[split])
    dataset = ExchangeProbeDataset(split, exchanges, annotations, settings, tokenizer, limit=limit)
    print("class_balance", json.dumps(dataset.class_balance(), sort_keys=True))
    return dataset


def collate_examples(examples: Sequence[EncodedExample], pad_token_id: int = 0) -> dict[str, Any]:
    max_len = max(example.length for example in examples)
    batch = len(examples)
    input_ids = torch.full((batch, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    probe_mask = torch.zeros((batch, max_len), dtype=torch.bool)
    labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
    lengths = torch.tensor([example.length for example in examples], dtype=torch.long)

    for row, example in enumerate(examples):
        length = example.length
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row, :length] = torch.tensor(example.attention_mask, dtype=torch.long)
        probe_mask[row, :length] = torch.tensor(example.probe_mask, dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "probe_mask": probe_mask,
        "labels": labels,
        "lengths": lengths,
        "row_keys": [example.row_key for example in examples],
        "exchange_ids": [example.exchange_id for example in examples],
        "prompt_ids": [example.prompt_id for example in examples],
        "splits": [example.split for example in examples],
    }


class LengthBucketedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_multiplier: int = 50,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_size = max(self.batch_size, int(bucket_multiplier) * self.batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            if self.shuffle:
                bucket_start = rng.randrange(self.batch_size) if len(bucket) > self.batch_size else 0
                bucket = bucket[bucket_start:] + bucket[:bucket_start]
            buckets.append(bucket)
        if self.shuffle:
            rng.shuffle(buckets)

        for bucket in buckets:
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                if batch:
                    yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)
