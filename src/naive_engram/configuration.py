from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class EngramConfig:
    """Architecture settings for every Engram injection.

    Layer indices are zero-based: ``2`` means immediately before Qwen decoder
    block 2 (the third decoder block).
    """

    layer_indices: tuple[int, ...] = (2, 12)
    ngram_orders: tuple[int, ...] = (2, 3)
    num_hash_heads: int = 2
    num_buckets: int = 65_536
    embedding_dim: int = 64
    hash_seed: int = 17
    gate_bias: float = -2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_indices", tuple(self.layer_indices))
        object.__setattr__(self, "ngram_orders", tuple(self.ngram_orders))
        if not self.layer_indices or min(self.layer_indices) < 0:
            raise ValueError("layer_indices must contain non-negative indices")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must not contain duplicates")
        if not self.ngram_orders or min(self.ngram_orders) < 1:
            raise ValueError("ngram_orders must contain positive orders")
        if self.num_hash_heads < 1:
            raise ValueError("num_hash_heads must be positive")
        if self.num_buckets < 2:
            raise ValueError("num_buckets must be at least 2")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

    @property
    def num_tables(self) -> int:
        return len(self.ngram_orders) * self.num_hash_heads

    @property
    def memory_width(self) -> int:
        return self.num_tables * self.embedding_dim

    @property
    def memory_parameters_per_layer(self) -> int:
        return self.num_tables * self.num_buckets * self.embedding_dim

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["layer_indices"] = list(self.layer_indices)
        data["ngram_orders"] = list(self.ngram_orders)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EngramConfig":
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "EngramConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
