from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from naive_engram import EngramConfig, EngramQwenForCausalLM
from naive_engram.modeling import CausalNGramHasher, EngramInjectedDecoderLayer, EngramMemory


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states + self.projection(hidden_states),)


class FakeCore(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, depth: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([FakeDecoderLayer(hidden_size) for _ in range(depth)])


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 31, hidden_size: int = 12, depth: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_cache=True, _name_or_path="fake")
        self.model = FakeCore(vocab_size, hidden_size, depth)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def tiny_config(**overrides) -> EngramConfig:
    values = dict(
        layer_indices=(1, 3),
        ngram_orders=(2, 3),
        num_hash_heads=2,
        num_buckets=23,
        embedding_dim=4,
    )
    values.update(overrides)
    return EngramConfig(**values)


def test_hashes_are_causal_and_prefixes_are_invalid() -> None:
    hasher = CausalNGramHasher(tiny_config())
    full_ids = torch.tensor([[4, 8, 15, 16, 23, 1]])
    prefix_ids = full_ids[:, :4]
    full_hashes, valid = hasher(full_ids)
    prefix_hashes, prefix_valid = hasher(prefix_ids)

    torch.testing.assert_close(full_hashes[:, :4], prefix_hashes)
    torch.testing.assert_close(valid[:, :4], prefix_valid)
    assert not valid[0, 0].any()  # no complete bigram or trigram
    assert valid[0, 1, :2].all() and not valid[0, 1, 2:].any()
    assert valid[0, 2].all()


def test_padding_invalidates_cross_padding_ngrams() -> None:
    hasher = CausalNGramHasher(tiny_config())
    _, valid = hasher(torch.tensor([[0, 0, 5, 6, 7]]), torch.tensor([[0, 0, 1, 1, 1]]))
    assert not valid[0, :3].any()
    assert valid[0, 3, :2].all() and not valid[0, 3, 2:].any()
    assert valid[0, 4].all()


def test_backbone_is_frozen_and_engram_trains() -> None:
    backbone = FakeCausalLM()
    model = EngramQwenForCausalLM(backbone, tiny_config())
    assert isinstance(backbone.model.layers[1], EngramInjectedDecoderLayer)
    assert isinstance(backbone.model.layers[3], EngramInjectedDecoderLayer)

    original_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".engram." not in name
    ]
    assert original_parameters and all(not parameter.requires_grad for parameter in original_parameters)
    assert all(parameter.requires_grad for _, layer in model.injections() for parameter in layer.engram.parameters())

    ids = torch.tensor([[2, 3, 5, 7, 11]])
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    output.loss.backward()
    assert all(parameter.grad is None for parameter in original_parameters)
    assert any(
        parameter.grad is not None
        for _, layer in model.injections()
        for parameter in layer.engram.parameters()
    )


def test_requested_table_parameter_count() -> None:
    config = EngramConfig()
    assert config.num_tables == 4
    assert config.memory_parameters_per_layer == 16_777_216


def test_invalid_prefix_has_exactly_zero_residual_after_norm_bias_trains() -> None:
    config = tiny_config(layer_indices=(1,))
    memory = EngramMemory(hidden_size=12, config=config)
    memory.memory_norm.bias.data.fill_(3.0)
    hidden = torch.randn(1, 2, 12)
    hashes = torch.zeros(1, 2, config.num_tables, dtype=torch.long)
    valid = torch.zeros(1, 2, config.num_tables, dtype=torch.bool)
    torch.testing.assert_close(memory(hidden, hashes, valid), hidden)


def test_adapter_round_trip(tmp_path) -> None:
    model = EngramQwenForCausalLM(FakeCausalLM(), tiny_config())
    model.save_adapter(tmp_path, "fake")
    clone = EngramQwenForCausalLM(FakeCausalLM(), tiny_config())
    clone.load_adapter(tmp_path)
    original = model.adapter_state_dict()
    restored = clone.adapter_state_dict()
    assert original.keys() == restored.keys()
    for key in original:
        torch.testing.assert_close(original[key], restored[key])


def test_real_qwen2_layer_api_and_checkpointed_backward() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    qwen_config = Qwen2Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = EngramQwenForCausalLM(Qwen2ForCausalLM(qwen_config), tiny_config(embedding_dim=2))
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    ids = torch.randint(0, qwen_config.vocab_size, (1, 8))
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False)
    output.loss.backward()
    assert output.logits.shape == (1, 8, qwen_config.vocab_size)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if ".engram." not in name
    )
