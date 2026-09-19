"""Online activation extraction for probe training."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.models import ModelAdapter, first_tensor, layer_subset, unwrap_model


def model_input_device(model: Any) -> torch.device:
    hf_model = getattr(model, "_codex_hf_model", None)
    if isinstance(hf_model, nn.Module):
        for param in hf_model.parameters():
            if param.device.type != "meta":
                return param.device
    if hasattr(model, "device") and getattr(model, "device") is not None:
        return torch.device(getattr(model, "device"))
    for root in (model, unwrap_model(model)):
        if isinstance(root, nn.Module):
            for param in root.parameters():
                if param.device.type != "meta":
                    return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("input_ids", "attention_mask", "probe_mask", "labels", "lengths"):
        value = moved.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device)
    return moved


def callable_model(model: Any) -> Any:
    hf_model = getattr(model, "_codex_hf_model", None)
    if isinstance(hf_model, nn.Module):
        return hf_model
    if isinstance(model, nn.Module):
        return model
    target = unwrap_model(model)
    if isinstance(target, nn.Module):
        return target
    if callable(model):
        return model
    raise TypeError("Could not find a callable model object")


def should_use_nnsight_trace(model: Any, requested: bool = True) -> bool:
    if not requested or not hasattr(model, "trace"):
        return False
    return not bool(getattr(model, "_codex_wrapped_hf_text_only", False))


def output_hidden_states(outputs: Any) -> tuple[Tensor, ...]:
    for attr in ("hidden_states",):
        value = getattr(outputs, attr, None)
        if value is not None:
            return tuple(value)
    language_outputs = getattr(outputs, "language_model_outputs", None)
    if language_outputs is not None and getattr(language_outputs, "hidden_states", None) is not None:
        return tuple(language_outputs.hidden_states)
    if isinstance(outputs, dict):
        if outputs.get("hidden_states") is not None:
            return tuple(outputs["hidden_states"])
        if outputs.get("language_model_outputs") is not None:
            nested = outputs["language_model_outputs"]
            if isinstance(nested, dict) and nested.get("hidden_states") is not None:
                return tuple(nested["hidden_states"])
            if getattr(nested, "hidden_states", None) is not None:
                return tuple(nested.hidden_states)
    raise ValueError("Model output did not include hidden_states")


def stack_layer_activations(raw_layers: list[Tensor], expected_batch: int, expected_seq: int) -> Tensor:
    if not raw_layers:
        raise ValueError("No activations were captured")
    normalized: list[Tensor] = []
    for index, value in enumerate(raw_layers):
        tensor = first_tensor(value)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Activation {index} is not a tensor: {type(tensor)}")
        if tensor.ndim != 3:
            raise ValueError(f"Activation {index} should be [B,S,H], got {tuple(tensor.shape)}")
        if tensor.shape[0] != expected_batch or tensor.shape[1] != expected_seq:
            raise ValueError(
                f"Activation {index} shape {tuple(tensor.shape)} does not match batch/seq "
                f"({expected_batch}, {expected_seq})"
            )
        normalized.append(tensor.detach().to(dtype=torch.bfloat16))
    acts = torch.stack(normalized, dim=2).detach()
    acts.requires_grad_(False)
    return acts


def rms_normalize_activations(acts: Tensor, eps: float = 1e-6, layer_chunk_size: int = 1) -> Tensor:
    """RMS-normalize activations over hidden dim without a full fp32 copy."""

    if acts.ndim != 4:
        raise ValueError(f"Expected activations [B,S,L,H], got {tuple(acts.shape)}")
    acts = acts.detach()
    chunk_layers = max(1, int(layer_chunk_size))
    for start in range(0, acts.shape[2], chunk_layers):
        stop = min(start + chunk_layers, acts.shape[2])
        chunk = acts[:, :, start:stop, :].float()
        rms = chunk.square().mean(dim=-1, keepdim=True).clamp_min(float(eps)).sqrt()
        chunk.div_(rms)
        chunk.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        acts[:, :, start:stop, :].copy_(chunk.to(dtype=acts.dtype))
        del chunk, rms
    acts = acts.to(dtype=torch.bfloat16).detach()
    acts.requires_grad_(False)
    return acts


def extract_residual_hf(
    model: Any,
    batch: dict[str, Any],
    include_embedding_layer: bool = False,
    layer_limit: int | None = None,
) -> Tensor:
    target = callable_model(model)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.inference_mode():
        outputs = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = output_hidden_states(outputs)
    selected = list(hidden if include_embedding_layer else hidden[1:])
    if layer_limit is not None:
        selected = selected[: int(layer_limit)]
    return stack_layer_activations(selected, input_ids.shape[0], input_ids.shape[1])


def extract_with_nnsight(
    adapter: ModelAdapter,
    model: Any,
    batch: dict[str, Any],
    site: str,
    layer_limit: int | None = None,
) -> Tensor:
    layers, _ = adapter.resolve_decoder_layers(model)
    selected_layers = layer_subset(layers, layer_limit)
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    with model.trace(inputs):
        saved = list().save()
        for layer in selected_layers:
            saved.append(adapter.layer_activation(layer, site))
    raw_layers = [item.detach() if torch.is_tensor(item) else item for item in saved]
    return stack_layer_activations(raw_layers, batch["input_ids"].shape[0], batch["input_ids"].shape[1])


def extract_with_hooks(
    adapter: ModelAdapter,
    model: Any,
    batch: dict[str, Any],
    site: str,
    layer_limit: int | None = None,
) -> Tensor:
    target = callable_model(model)
    layers, _ = adapter.resolve_decoder_layers(model)
    selected_layers = layer_subset(layers, layer_limit)
    captured: list[Tensor] = []
    handles: list[Any] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = first_tensor(output)
        if not torch.is_tensor(tensor):
            raise TypeError(f"Hook for {site} did not receive a tensor output")
        captured.append(tensor.detach())

    for layer in selected_layers:
        module = adapter.activation_module(layer, site)
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            target(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return stack_layer_activations(captured, batch["input_ids"].shape[0], batch["input_ids"].shape[1])


def extract_sites_with_hooks(
    adapter: ModelAdapter,
    model: Any,
    batch: dict[str, Any],
    sites: list[str] | tuple[str, ...],
    include_embedding_layer: bool = False,
    layer_limit: int | None = None,
) -> dict[str, Tensor]:
    """Capture several activation sites during one frozen-model forward."""

    requested = list(dict.fromkeys(str(site) for site in sites))
    unknown = [site for site in requested if site not in {"residual", "mlp", "attention"}]
    if unknown:
        raise ValueError(f"Unknown activation sites: {unknown}")
    if not requested:
        raise ValueError("At least one activation site is required")

    target = callable_model(model)
    layers, _ = adapter.resolve_decoder_layers(model)
    selected_layers = layer_subset(layers, layer_limit)
    captured: dict[str, list[Tensor]] = {site: [] for site in requested if site != "residual"}
    handles: list[Any] = []

    def hook_for(site: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if not torch.is_tensor(tensor):
                raise TypeError(f"Hook for {site} did not receive a tensor output")
            captured[site].append(tensor.detach())

        return hook

    for site in captured:
        for layer in selected_layers:
            handles.append(adapter.activation_module(layer, site).register_forward_hook(hook_for(site)))

    needs_residual = "residual" in requested
    try:
        with torch.inference_mode():
            outputs = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=needs_residual,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    expected_batch = batch["input_ids"].shape[0]
    expected_seq = batch["input_ids"].shape[1]
    activations: dict[str, Tensor] = {}
    if needs_residual:
        hidden = output_hidden_states(outputs)
        residual_layers = list(hidden if include_embedding_layer else hidden[1:])
        if layer_limit is not None:
            residual_layers = residual_layers[: int(layer_limit)]
        activations["residual"] = stack_layer_activations(residual_layers, expected_batch, expected_seq)
        del residual_layers, hidden
    del outputs

    for site, raw_layers in captured.items():
        if len(raw_layers) != len(selected_layers):
            raise RuntimeError(f"Expected {len(selected_layers)} {site} layers, captured {len(raw_layers)}")
        activations[site] = stack_layer_activations(raw_layers, expected_batch, expected_seq)
        raw_layers.clear()
    return {site: activations[site] for site in requested}


def validate_extracted_activations(acts: Tensor, expected_batch: int, expected_seq: int) -> None:
    expected = (expected_batch, expected_seq)
    if acts.shape[:2] != expected:
        raise AssertionError(f"Expected activations to start with {expected}, got {tuple(acts.shape)}")
    if acts.dtype != torch.bfloat16:
        raise AssertionError(f"Expected bf16 activations, got {acts.dtype}")
    if acts.requires_grad:
        raise AssertionError("Extracted activations must be detached")


def extract_activation_sites(
    adapter: ModelAdapter,
    model: Any,
    batch: dict[str, Any],
    sites: list[str] | tuple[str, ...],
    include_embedding_layer: bool = False,
    layer_limit: int | None = None,
    normalize_activations: bool = True,
    activation_norm_eps: float = 1e-6,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Extract all requested sites in one backbone forward."""

    device = model_input_device(model)
    batch = move_batch_to_device(batch, device)
    activations = extract_sites_with_hooks(
        adapter,
        model,
        batch,
        sites,
        include_embedding_layer=include_embedding_layer,
        layer_limit=layer_limit,
    )
    for acts in activations.values():
        if normalize_activations:
            rms_normalize_activations(acts, eps=activation_norm_eps)
        validate_extracted_activations(acts, batch["input_ids"].shape[0], batch["input_ids"].shape[1])
    return activations, batch["probe_mask"].bool(), batch["labels"].float()


def extract_activations(
    adapter: ModelAdapter,
    model: Any,
    batch: dict[str, Any],
    site: str,
    include_embedding_layer: bool = False,
    layer_limit: int | None = None,
    use_nnsight: bool = True,
    normalize_activations: bool = True,
    activation_norm_eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    device = model_input_device(model)
    batch = move_batch_to_device(batch, device)
    if site == "residual":
        acts = extract_residual_hf(model, batch, include_embedding_layer=include_embedding_layer, layer_limit=layer_limit)
    elif should_use_nnsight_trace(model, requested=use_nnsight):
        acts = extract_with_nnsight(adapter, model, batch, site=site, layer_limit=layer_limit)
    else:
        acts = extract_with_hooks(adapter, model, batch, site=site, layer_limit=layer_limit)

    if normalize_activations:
        acts = rms_normalize_activations(acts, eps=activation_norm_eps)

    validate_extracted_activations(acts, batch["input_ids"].shape[0], batch["input_ids"].shape[1])
    return acts, batch["probe_mask"].bool(), batch["labels"].float()


def residual_sanity_check(
    adapter: ModelAdapter,
    model: Any,
    batch: dict[str, Any],
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    if not should_use_nnsight_trace(model):
        reason = "wrapped HF text-only model uses hooks" if getattr(model, "_codex_wrapped_hf_text_only", False) else "model has no nnSight trace method"
        print(f"residual_sanity_check skipped: {reason}")
        return False
    device = model_input_device(model)
    batch = move_batch_to_device(batch, device)
    hf = extract_residual_hf(model, batch, include_embedding_layer=False, layer_limit=1)
    ns = extract_with_nnsight(adapter, model, batch, site="residual", layer_limit=1)
    ok = torch.allclose(hf[:, : min(4, hf.shape[1]), 0].float(), ns[:, : min(4, ns.shape[1]), 0].float(), atol=atol, rtol=rtol)
    print(f"residual_sanity_check={ok}")
    return bool(ok)
