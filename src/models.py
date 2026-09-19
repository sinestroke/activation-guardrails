"""Model adapters for activation extraction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ModelDims:
    n_layers: int
    hidden_size: int


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_dtype_from_name(name: str | None) -> torch.dtype:
    if name in (None, "auto"):
        return torch.bfloat16
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def require_hf_token(env_name: str) -> str:
    token = os.environ.get(env_name)
    if not token:
        raise RuntimeError(
            f"Hugging Face authentication is required for the private checkpoint. "
            f"Set {env_name} or run `huggingface-cli login` before loading the model."
        )
    return token


def local_files_only_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("local_files_only", False)) or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) == "1"


def unwrap_model(model: Any) -> Any:
    for attr in ("_model", "model", "module"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            if hasattr(inner, "config") or isinstance(inner, nn.Module):
                return inner
    return model


def get_nested_attr(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def first_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return value


def tokenizer_metadata(
    tokenizer: Any,
    requested_tokenizer_name: str,
    requested_revision: str | None = None,
) -> dict[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    chat_template = getattr(tokenizer, "chat_template", None)
    template_text = chat_template if isinstance(chat_template, str) else None
    special_token_ids = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        )
    }
    return {
        "requested_tokenizer_name": str(requested_tokenizer_name),
        "requested_revision": requested_revision,
        "resolved_name_or_path": getattr(tokenizer, "name_or_path", None),
        "resolved_commit_hash": init_kwargs.get("_commit_hash"),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "length": len(tokenizer),
        "special_token_ids": special_token_ids,
        "chat_template_sha256": (
            hashlib.sha256(template_text.encode("utf-8")).hexdigest() if template_text is not None else None
        ),
    }


def validate_tokenizer_model_compatibility(
    tokenizer: Any,
    model_config: Any,
    *,
    tokenizer_name: str,
    model_name: str,
) -> None:
    text_config = getattr(model_config, "text_config", None) or model_config
    model_vocab_size = getattr(text_config, "vocab_size", None)
    if model_vocab_size is None:
        return

    tokenizer_length = len(tokenizer)
    if tokenizer_length > int(model_vocab_size):
        raise ValueError(
            f"Tokenizer {tokenizer_name!r} has {tokenizer_length} token IDs, but model {model_name!r} "
            f"accepts only {model_vocab_size}. Choose a tokenizer from the same model family/checkpoint."
        )

    special_token_ids = [
        getattr(tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    ]
    invalid_ids = [
        token_id
        for token_id in special_token_ids
        if token_id is not None and token_id >= model_vocab_size
    ]
    if invalid_ids:
        raise ValueError(
            f"Tokenizer {tokenizer_name!r} uses special token IDs outside model {model_name!r}'s "
            f"vocabulary: {invalid_ids} >= {model_vocab_size}."
        )


class ModelAdapter:
    name = "generic"
    decoder_path_candidates: tuple[str, ...] = (
        "model.layers",
        "model.decoder.layers",
        "layers",
    )

    def load(self, model_name: str, config: dict[str, Any]) -> tuple[Any, Any]:
        hf_token_env = str(config.get("hf_token_env", "HF_TOKEN"))
        local_files_only = local_files_only_enabled(config)
        token = os.environ.get(hf_token_env)
        if not token and not local_files_only:
            token = require_hf_token(hf_token_env)
        dtype = torch_dtype_from_name(config.get("dtype", "bfloat16"))
        loader = str(config.get("model_loader", "nnsight")).lower()
        device_map = config.get("device_map", "auto")
        trust_remote_code = bool(config.get("trust_remote_code", False))
        attn_implementation = config.get("attn_implementation")
        revision = config.get("model_revision")
        tokenizer_name = str(config.get("tokenizer_name") or model_name)
        tokenizer_revision = config.get("tokenizer_revision")
        if tokenizer_revision is None and tokenizer_name == model_name:
            tokenizer_revision = revision
        target_model_adapter = self._target_model_adapter_config(config)

        from transformers import AutoConfig, AutoTokenizer

        model_config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            revision=revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            revision=tokenizer_revision,
        )
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if bool(config.get("require_tokenizer_model_vocab_match", True)):
            validate_tokenizer_model_compatibility(
                tokenizer,
                model_config,
                tokenizer_name=tokenizer_name,
                model_name=model_name,
            )
        config["resolved_tokenizer"] = tokenizer_metadata(
            tokenizer,
            requested_tokenizer_name=tokenizer_name,
            requested_revision=tokenizer_revision,
        )

        print(
            "versions",
            {
                "torch": str(torch.__version__),
                "transformers": package_version("transformers"),
                "nnsight": package_version("nnsight"),
            },
        )
        print("resolved_tokenizer", config["resolved_tokenizer"])

        model = None
        if loader == "nnsight":
            nnsight_version = package_version("nnsight")
            if nnsight_version is None:
                raise RuntimeError("model_loader=nnsight requires `nnsight` to be installed")
            use_custom_hf_wrapper = (
                tokenizer_name != model_name
                or tokenizer_revision != revision
                or target_model_adapter is not None
                or (
                    self._is_vision_language_config(model_config)
                    and bool(config.get("nnsight_wrap_hf_for_vlm", True))
                )
            )
            if use_custom_hf_wrapper:
                model = self._load_hf_then_wrap_nnsight(
                    model_name=model_name,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    token=token,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    attn_implementation=attn_implementation,
                    revision=revision,
                    target_model_adapter=target_model_adapter,
                    runtime_config=config,
                )
            else:
                nnsight_cls = self._select_nnsight_model_class(model_config)
                nnsight_kwargs = {
                    "device_map": device_map,
                    "dispatch": True,
                    "torch_dtype": dtype,
                    "token": token,
                    "local_files_only": local_files_only,
                }
                if revision is not None:
                    nnsight_kwargs["revision"] = revision
                model = nnsight_cls(model_name, **nnsight_kwargs)

        if loader == "hf":
            model_cls = self._select_hf_model_class(model_config)
            kwargs: dict[str, Any] = {
                "token": token,
                "torch_dtype": dtype,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "local_files_only": local_files_only,
            }
            if revision is not None:
                kwargs["revision"] = revision
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation
            model = model_cls.from_pretrained(model_name, **kwargs)
            model = self._apply_target_model_adapter(
                model,
                target_model_adapter,
                token=token,
                local_files_only=local_files_only,
                runtime_config=config,
            )
        elif loader != "nnsight":
            raise ValueError("model_loader must be either 'nnsight' or 'hf'")

        if model is None:
            raise RuntimeError("Model loader did not return a model")
        self.freeze(model)
        return model, tokenizer

    def _target_model_adapter_config(self, config: Mapping[str, Any]) -> dict[str, Any] | None:
        value = config.get("target_model_adapter")
        if value in (None, False):
            return None
        if isinstance(value, str):
            value = {"repo_id": value}
        if not isinstance(value, Mapping):
            raise ValueError("target_model_adapter must be a mapping, repo-id string, or null")
        if not bool(value.get("enabled", True)):
            return None
        repo_id = value.get("repo_id")
        if not repo_id:
            raise ValueError("target_model_adapter.repo_id is required")
        return {
            "repo_id": str(repo_id),
            "revision": None if value.get("revision") is None else str(value["revision"]),
            "merge_and_unload": bool(value.get("merge_and_unload", True)),
            "safe_merge": bool(value.get("safe_merge", True)),
        }

    def _apply_target_model_adapter(
        self,
        model: Any,
        adapter_config: Mapping[str, Any] | None,
        *,
        token: str | None,
        local_files_only: bool,
        runtime_config: dict[str, Any],
    ) -> Any:
        if adapter_config is None:
            return model

        from peft import PeftModel

        repo_id = str(adapter_config["repo_id"])
        revision = adapter_config.get("revision")
        peft_model = PeftModel.from_pretrained(
            model,
            repo_id,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            is_trainable=False,
        )
        peft_configs = getattr(peft_model, "peft_config", {})
        resolved_config = next(iter(peft_configs.values()), None) if isinstance(peft_configs, Mapping) else None
        resolved = {
            "repo_id": repo_id,
            "revision": revision,
            "merge_and_unload": bool(adapter_config.get("merge_and_unload", True)),
            "safe_merge": bool(adapter_config.get("safe_merge", True)),
            "resolved_base_model_name_or_path": getattr(resolved_config, "base_model_name_or_path", None),
            "resolved_commit_hash": getattr(resolved_config, "_commit_hash", None),
        }
        runtime_config["resolved_target_model_adapter"] = resolved

        if resolved["merge_and_unload"]:
            model = peft_model.merge_and_unload(safe_merge=resolved["safe_merge"])
            print(
                f"loaded_target_model_adapter repo_id={repo_id} revision={revision} "
                f"merge_and_unload=true safe_merge={resolved['safe_merge']}"
            )
            return model

        print(f"loaded_target_model_adapter repo_id={repo_id} revision={revision} merge_and_unload=false")
        return peft_model

    def _hf_from_pretrained_kwargs(
        self,
        token: str | None,
        dtype: torch.dtype,
        device_map: Any,
        trust_remote_code: bool,
        local_files_only: bool,
        attn_implementation: Any,
        revision: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "token": token,
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if revision is not None:
            kwargs["revision"] = revision
        return kwargs

    def _load_hf_then_wrap_nnsight(
        self,
        model_name: str,
        model_config: Any,
        tokenizer: Any,
        token: str | None,
        dtype: torch.dtype,
        device_map: Any,
        trust_remote_code: bool,
        local_files_only: bool,
        attn_implementation: Any,
        revision: str | None,
        target_model_adapter: Mapping[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> Any:
        """Load with Transformers, optionally attach PEFT, then wrap with nnSight.

        nnSight's repo-id loader classifies Gemma 3 configs as VLMs and asks
        for an image processor. Text-only Gemma 3 workflows use this
        workflow, so we bypass the repo-id classification by giving nnSight the
        already-loaded Transformers model as a custom language model.
        """

        model_cls = self._select_hf_model_class(model_config)
        hf_model = model_cls.from_pretrained(
            model_name,
            **self._hf_from_pretrained_kwargs(
                token=token,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                attn_implementation=attn_implementation,
                revision=revision,
            ),
        )
        hf_model = self._apply_target_model_adapter(
            hf_model,
            target_model_adapter,
            token=token,
            local_files_only=local_files_only,
            runtime_config=runtime_config if runtime_config is not None else {},
        )
        hf_model.requires_grad_(False)
        hf_model.eval()

        from nnsight import LanguageModel

        wrapped = LanguageModel(hf_model, tokenizer=tokenizer)
        # Keep the real HF module available for residual fast-path forwards and
        # device discovery without registering it as a PyTorch child module.
        # nn.Module.__setattr__ registers Module values in _modules; doing that
        # on an nnSight wrapper can create recursive train()/eval() traversal.
        wrapped.__dict__["_codex_hf_model"] = hf_model
        wrapped.__dict__["_codex_wrapped_hf_text_only"] = True
        print("loaded_nnsight_custom_hf_wrapper=true")
        return wrapped

    def _is_vision_language_config(self, config: Any) -> bool:
        architectures = [str(item).lower() for item in getattr(config, "architectures", []) or []]
        model_type = str(getattr(config, "model_type", "")).lower()
        joined = " ".join(architectures + [model_type])
        return (
            "gemma3forconditionalgeneration" in joined
            or "automodelforimagetexttotext" in joined
            or "vision" in joined
            or "image" in joined
            or hasattr(config, "vision_config")
        )

    def _select_nnsight_model_class(self, config: Any) -> Any:
        if self._is_vision_language_config(config):
            try:
                from nnsight import VisionLanguageModel
            except ImportError as exc:
                raise RuntimeError(
                    "This checkpoint is multimodal. Install an nnSight version with "
                    "VisionLanguageModel support, or run with `--model-loader hf`."
                ) from exc
            return VisionLanguageModel

        from nnsight import LanguageModel

        return LanguageModel

    def _select_hf_model_class(self, config: Any) -> Any:
        from transformers import AutoModelForCausalLM

        architectures = [str(item) for item in getattr(config, "architectures", []) or []]
        if any("Gemma3ForConditionalGeneration" in arch for arch in architectures):
            try:
                from transformers import AutoModelForImageTextToText

                return AutoModelForImageTextToText
            except ImportError:
                return AutoModelForCausalLM
        return AutoModelForCausalLM

    def freeze(self, model: Any) -> None:
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        else:
            target = unwrap_model(model)
            if hasattr(target, "requires_grad_"):
                target.requires_grad_(False)
        if hasattr(model, "eval"):
            model.eval()
        else:
            target = unwrap_model(model)
            if hasattr(target, "eval"):
                target.eval()

    def config(self, model: Any) -> Any:
        target = unwrap_model(model)
        cfg = getattr(target, "config", None)
        if cfg is None and hasattr(model, "config"):
            cfg = model.config
        if cfg is None:
            raise ValueError("Could not find model config")
        return cfg

    def dims(self, config: Any) -> ModelDims:
        text_config = getattr(config, "text_config", None) or config
        n_layers = getattr(text_config, "num_hidden_layers", None) or getattr(text_config, "n_layer", None)
        hidden = getattr(text_config, "hidden_size", None) or getattr(text_config, "n_embd", None)
        if n_layers is None or hidden is None:
            raise ValueError("Could not resolve num_hidden_layers/hidden_size from config")
        return ModelDims(n_layers=int(n_layers), hidden_size=int(hidden))

    def resolve_decoder_layers(self, model: Any) -> tuple[nn.ModuleList | list[Any], str]:
        roots = [model, unwrap_model(model)]
        seen: set[int] = set()
        for root in roots:
            if id(root) in seen:
                continue
            seen.add(id(root))
            for path in self.decoder_path_candidates:
                value = get_nested_attr(root, path)
                if self._looks_like_layers(value):
                    return value, path
        for root in roots:
            found = self._find_layers_by_inspection(root)
            if found is not None:
                return found
        raise ValueError("Could not locate decoder layers; inspect the printed module tree")

    def _looks_like_layers(self, value: Any) -> bool:
        if not isinstance(value, (nn.ModuleList, list, tuple)) or len(value) == 0:
            return False
        first = value[0]
        return hasattr(first, "self_attn") or hasattr(first, "mlp") or isinstance(first, nn.Module)

    def _find_layers_by_inspection(self, model: Any) -> tuple[nn.ModuleList, str] | None:
        if not isinstance(model, nn.Module):
            return None
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                first = module[0]
                if hasattr(first, "self_attn") and hasattr(first, "mlp"):
                    return module, name
        return None

    def layer_activation(self, layer: Any, site: str) -> Any:
        if site == "residual":
            return first_tensor(getattr(layer, "output", None))
        if site == "attention":
            attn = getattr(layer, "self_attn")
            return first_tensor(getattr(attn, "output", getattr(attn, "outputs", None)))
        if site == "mlp":
            mlp = getattr(layer, "mlp")
            return first_tensor(getattr(mlp, "output", getattr(mlp, "outputs", None)))
        raise ValueError(f"Unknown activation site {site!r}")

    def activation_module(self, layer: Any, site: str) -> nn.Module:
        if site == "residual":
            return layer
        if site == "attention":
            return getattr(layer, "self_attn")
        if site == "mlp":
            return getattr(layer, "mlp")
        raise ValueError(f"Unknown activation site {site!r}")

    def format_chat(self, turns: list[dict[str, str]], tokenizer: Any) -> tuple[list[int], list[int]]:
        from src.data import apply_chat_template

        return apply_chat_template(turns, tokenizer)

    def assistant_span(self, turns: list[dict[str, str]], tokenizer: Any) -> list[bool]:
        full_ids, _ = self.format_chat(turns, tokenizer)
        non_assistant_turns = [turn for turn in turns if turn.get("role") != "assistant"]
        prefix_ids, _ = self.format_chat(non_assistant_turns, tokenizer)
        start = min(len(prefix_ids), len(full_ids))
        return [index >= start for index in range(len(full_ids))]


class Gemma3Adapter(ModelAdapter):
    name = "gemma3"
    decoder_path_candidates = (
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.model.language_model.layers",
        "model.layers",
    )

    def dims(self, config: Any) -> ModelDims:
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            return super().dims(config)
        return super().dims(text_config)


class LlamaStyleAdapter(ModelAdapter):
    name = "llama_style"
    decoder_path_candidates = (
        "model.layers",
        "base_model.model.model.layers",
        "layers",
    )


class Qwen3Adapter(LlamaStyleAdapter):
    name = "qwen3"


class Olmo3Adapter(LlamaStyleAdapter):
    name = "olmo3"


class GenericAdapter(ModelAdapter):
    name = "generic"


def adapter_for_config(config: Any) -> ModelAdapter:
    architectures = [str(item).lower() for item in getattr(config, "architectures", []) or []]
    model_type = str(getattr(config, "model_type", "")).lower()
    joined = " ".join(architectures + [model_type])
    if "gemma3" in joined or "gemma-3" in joined or "gemma3forconditionalgeneration" in joined:
        return Gemma3Adapter()
    if "qwen3" in joined or "qwen" in joined:
        return Qwen3Adapter()
    if "olmo3" in joined or "olmo-3" in joined:
        return Olmo3Adapter()
    if "llama" in joined or "mistral" in joined:
        return LlamaStyleAdapter()
    return GenericAdapter()


def load_adapter_and_model(config: dict[str, Any]) -> tuple[ModelAdapter, Any, Any, ModelDims, str]:
    model_name = str(config["model_name"])
    pre_adapter = GenericAdapter()
    model, tokenizer = pre_adapter.load(model_name, config)
    cfg = pre_adapter.config(model)
    adapter = adapter_for_config(cfg)
    adapter.freeze(model)
    layers, path = adapter.resolve_decoder_layers(model)
    dims = adapter.dims(cfg)
    print(f"resolved_adapter={adapter.name} decoder_path={path} dims={dims}")
    print_module_tree(unwrap_model(model), max_depth=int(config.get("module_tree_depth", 3)))
    if len(layers) != dims.n_layers:
        print(f"warning: resolved {len(layers)} layers, config reports {dims.n_layers}")
    return adapter, model, tokenizer, dims, path


def print_module_tree(model: Any, max_depth: int = 3, max_children: int = 160) -> None:
    print("=== Model module tree ===")
    if not isinstance(model, nn.Module):
        print(f"{type(model).__name__} (not a torch.nn.Module; nnsight proxy may hide the tree)")
        return
    printed = 0
    for name, module in model.named_modules():
        depth = 0 if not name else name.count(".") + 1
        if depth > max_depth:
            continue
        indent = "  " * depth
        label = name if name else "<root>"
        print(f"{indent}{label}: {module.__class__.__name__}")
        printed += 1
        if printed >= max_children:
            print(f"... truncated after {max_children} modules")
            break


def layer_subset(layers: Iterable[Any], limit: int | None = None) -> list[Any]:
    selected = list(layers)
    if limit is not None:
        selected = selected[: int(limit)]
    return selected
