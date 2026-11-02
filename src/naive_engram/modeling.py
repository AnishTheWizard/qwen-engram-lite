from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
import json
import math
import random

import torch
from torch import Tensor, nn

from .configuration import EngramConfig


class CausalNGramHasher(nn.Module):
    """Deterministically maps causal token n-grams to independent hash heads."""

    def __init__(self, config: EngramConfig) -> None:
        super().__init__()
        self.orders = config.ngram_orders
        self.num_hash_heads = config.num_hash_heads
        self.num_buckets = config.num_buckets

        rng = random.Random(config.hash_seed)
        multipliers: list[list[list[int]]] = []
        offsets: list[list[int]] = []
        # Keep products comfortably inside signed int64 for normal vocab IDs.
        for order in self.orders:
            multipliers.append(
                [[rng.randrange(1, 2**31, 2) for _ in range(order)] for _ in range(self.num_hash_heads)]
            )
            offsets.append([rng.randrange(0, 2**31) for _ in range(self.num_hash_heads)])

        # Orders have different widths, so buffers are kept one per order.
        for i, (order_multipliers, order_offsets) in enumerate(zip(multipliers, offsets, strict=True)):
            self.register_buffer(
                f"multipliers_{i}", torch.tensor(order_multipliers, dtype=torch.long), persistent=False
            )
            self.register_buffer(f"offsets_{i}", torch.tensor(order_offsets, dtype=torch.long), persistent=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(dtype=torch.long)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.to(dtype=torch.bool)

        all_hashes: list[Tensor] = []
        all_valid: list[Tensor] = []
        batch, sequence = input_ids.shape
        for order_index, order in enumerate(self.orders):
            if sequence == 0:
                windows = input_ids.new_empty((batch, 0, order))
                mask_windows = attention_mask.new_empty((batch, 0, order))
            else:
                token_pad = input_ids.new_zeros((batch, order - 1))
                mask_pad = attention_mask.new_zeros((batch, order - 1))
                windows = torch.cat((token_pad, input_ids), dim=1).unfold(1, order, 1)
                mask_windows = torch.cat((mask_pad, attention_mask), dim=1).unfold(1, order, 1)

            multipliers = getattr(self, f"multipliers_{order_index}")
            offsets = getattr(self, f"offsets_{order_index}")
            # [batch, sequence, heads, order] -> [batch, sequence, heads]
            hashes = (
                windows.unsqueeze(2) * multipliers.view(1, 1, self.num_hash_heads, order)
            ).sum(dim=-1)
            hashes = torch.remainder(hashes + offsets.view(1, 1, -1), self.num_buckets)
            valid = mask_windows.all(dim=-1, keepdim=True).expand(-1, -1, self.num_hash_heads)
            all_hashes.append(hashes)
            all_valid.append(valid)

        return torch.cat(all_hashes, dim=-1), torch.cat(all_valid, dim=-1)


class EngramMemory(nn.Module):
    """Hashed memory lookup followed by key/value gated residual fusion."""

    def __init__(self, hidden_size: int, config: EngramConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_width = config.memory_width
        self.tables = nn.ModuleList(
            [nn.Embedding(config.num_buckets, config.embedding_dim) for _ in range(config.num_tables)]
        )
        self.memory_norm = nn.LayerNorm(self.memory_width, elementwise_affine=True)
        self.query_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(self.memory_width, hidden_size, bias=False)
        self.value_proj = nn.Linear(self.memory_width, hidden_size, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(float(config.gate_bias)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.memory_norm.weight)
        nn.init.zeros_(self.memory_norm.bias)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        # A small value projection makes insertion stable without blocking
        # gradients to the table and gate on the first optimization step.
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=1e-3)

    @property
    def table_parameter_count(self) -> int:
        return sum(table.weight.numel() for table in self.tables)

    def forward(self, hidden_states: Tensor, hash_indices: Tensor, valid: Tensor) -> Tensor:
        if hidden_states.shape[:2] != hash_indices.shape[:2]:
            raise RuntimeError(
                "Engram context shape does not match hidden states. Cached decoding is unsupported; "
                "use EngramQwenForCausalLM.generate(), which disables the KV cache."
            )
        memories = [table(hash_indices[..., i]) for i, table in enumerate(self.tables)]
        memory = torch.cat(memories, dim=-1)
        # Invalid prefix/padding tables are exactly zero and cannot contribute.
        expanded_valid = valid.unsqueeze(-1).expand(*valid.shape, memories[0].shape[-1])
        memory_mask = expanded_valid.reshape_as(memory).to(dtype=memory.dtype)
        memory = memory * memory_mask
        memory = self.memory_norm(memory)
        # Mask again after affine normalization so its learned bias can never
        # turn an invalid prefix or padding lookup into a residual.
        memory = memory * memory_mask

        compute_dtype = self.query_proj.weight.dtype
        hidden_for_fusion = hidden_states.to(dtype=compute_dtype)
        query = self.query_proj(self.query_norm(hidden_for_fusion))
        key = self.key_proj(memory.to(dtype=compute_dtype))
        value = self.value_proj(memory.to(dtype=compute_dtype))
        score = (query * key).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(score + self.gate_bias.to(dtype=score.dtype))
        residual = gate * value
        return hidden_states + residual.to(dtype=hidden_states.dtype)


class EngramInjectedDecoderLayer(nn.Module):
    """Adds the Engram residual immediately before an existing decoder layer."""

    def __init__(self, decoder_layer: nn.Module, engram: EngramMemory) -> None:
        super().__init__()
        self.decoder_layer = decoder_layer
        self.engram = engram
        self._hash_indices: Tensor | None = None
        self._valid: Tensor | None = None

    def set_context(self, hash_indices: Tensor, valid: Tensor) -> None:
        # Plain attributes deliberately keep this context out of checkpoints.
        # Retaining it through backward supports activation checkpoint replay.
        self._hash_indices = hash_indices
        self._valid = valid

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        if self._hash_indices is None or self._valid is None:
            raise RuntimeError("Engram context was not prepared; call the top-level Engram model")
        hidden_states = self.engram(hidden_states, self._hash_indices, self._valid)
        return self.decoder_layer(hidden_states, *args, **kwargs)


def _decoder_layers(backbone: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h"):
        current: Any = backbone
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, nn.ModuleList):
            return current
    raise TypeError("Could not locate decoder layers (expected backbone.model.layers)")


class EngramQwenForCausalLM(nn.Module):
    """Frozen Qwen causal LM with trainable Engram injections."""

    adapter_config_name = "engram_config.json"
    adapter_weights_name = "engram.safetensors"

    def __init__(self, backbone: nn.Module, engram_config: EngramConfig | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.engram_config = engram_config or EngramConfig()
        self.hasher = CausalNGramHasher(self.engram_config)

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        layers = _decoder_layers(self.backbone)
        if max(self.engram_config.layer_indices) >= len(layers):
            raise ValueError(
                f"Engram layer index exceeds backbone depth {len(layers)}: "
                f"{self.engram_config.layer_indices}"
            )
        hidden_size = int(self.backbone.config.hidden_size)
        for index in self.engram_config.layer_indices:
            if isinstance(layers[index], EngramInjectedDecoderLayer):
                raise ValueError(f"decoder layer {index} is already Engram-wrapped")
            layers[index] = EngramInjectedDecoderLayer(
                layers[index], EngramMemory(hidden_size, self.engram_config)
            )

        # Only newly created Engram parameters are trainable.
        for _, injection in self.injections():
            injection.engram.requires_grad_(True)
        self.backbone.eval()

    @property
    def config(self) -> Any:
        return self.backbone.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def injections(self) -> Iterator[tuple[int, EngramInjectedDecoderLayer]]:
        layers = _decoder_layers(self.backbone)
        for index in self.engram_config.layer_indices:
            injection = layers[index]
            if not isinstance(injection, EngramInjectedDecoderLayer):
                raise RuntimeError(f"expected Engram wrapper at decoder layer {index}")
            yield index, injection

    def train(self, mode: bool = True) -> "EngramQwenForCausalLM":
        super().train(mode)
        # Stage A keeps the pretrained backbone in inference mode as well as
        # freezing it; only the inserted modules switch to training mode.
        self.backbone.eval()
        for _, injection in self.injections():
            injection.engram.train(mode)
        return self

    def _set_context(self, input_ids: Tensor, attention_mask: Tensor | None) -> None:
        hash_indices, valid = self.hasher(input_ids, attention_mask)
        for _, injection in self.injections():
            injection.set_context(hash_indices, valid)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None:
            raise ValueError("Engram requires input_ids; inputs_embeds alone cannot define token n-grams")
        self._set_context(input_ids, attention_mask)
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> Tensor:
        """Generate correctly by recomputing full prefixes without a KV cache."""

        kwargs["use_cache"] = False

        def prepare_context(_module: nn.Module, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]):
            input_ids = hook_kwargs.get("input_ids")
            if input_ids is None and hook_args:
                input_ids = hook_args[0]
            if input_ids is None:
                raise ValueError("generation requires input_ids")
            self._set_context(input_ids, hook_kwargs.get("attention_mask"))
            return hook_args, hook_kwargs

        handle = self.backbone.register_forward_pre_hook(prepare_context, with_kwargs=True)
        try:
            return self.backbone.generate(*args, **kwargs)
        finally:
            handle.remove()

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        for index, injection in self.injections():
            for name, value in injection.engram.state_dict().items():
                state[f"layers.{index}.{name}"] = value.detach().cpu().contiguous()
        return state

    def save_adapter(self, output_dir: str | Path, base_model_name: str | None = None) -> None:
        from safetensors.torch import save_file

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        config_data = self.engram_config.to_dict()
        config_data["base_model_name"] = base_model_name or getattr(
            self.backbone.config, "_name_or_path", "Qwen/Qwen2.5-0.5B"
        )
        (output / self.adapter_config_name).write_text(json.dumps(config_data, indent=2) + "\n")
        save_file(self.adapter_state_dict(), str(output / self.adapter_weights_name))

    def load_adapter(self, adapter_dir: str | Path, strict: bool = True) -> None:
        from safetensors.torch import load_file

        weights = load_file(str(Path(adapter_dir) / self.adapter_weights_name), device="cpu")
        expected: set[str] = set()
        unexpected = set(weights)
        for index, injection in self.injections():
            prefix = f"layers.{index}."
            local = {name[len(prefix) :]: value for name, value in weights.items() if name.startswith(prefix)}
            expected.update(prefix + name for name in injection.engram.state_dict())
            unexpected.difference_update(local_name for local_name in weights if local_name.startswith(prefix))
            injection.engram.load_state_dict(local, strict=strict)
        if strict:
            missing = expected.difference(weights)
            if missing or unexpected:
                raise RuntimeError(f"adapter key mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        engram_config: EngramConfig | None = None,
        **kwargs: Any,
    ) -> "EngramQwenForCausalLM":
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        return cls(backbone, engram_config)

    @classmethod
    def from_adapter(cls, adapter_dir: str | Path, **model_kwargs: Any) -> "EngramQwenForCausalLM":
        config_data = json.loads((Path(adapter_dir) / cls.adapter_config_name).read_text())
        base_model_name = str(config_data.pop("base_model_name"))
        model = cls.from_pretrained(base_model_name, EngramConfig.from_dict(config_data), **model_kwargs)
        model.load_adapter(adapter_dir)
        return model
