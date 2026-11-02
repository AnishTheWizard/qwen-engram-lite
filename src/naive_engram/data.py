from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PackedTokenDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Tokenize documents and concatenate them into fixed-length LM blocks.

    An EOS token separates documents. Incomplete final blocks are dropped so
    every batch has a static shape and no padding contaminates an n-gram.
    """

    def __init__(
        self,
        source: Any,
        tokenizer: Callable[..., dict[str, Any]],
        text_column: str,
        block_size: int,
        eos_token_id: int,
    ) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.source = source
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.eos_token_id = eos_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        source = self.source
        worker = get_worker_info()
        if worker is not None:
            source = source.shard(num_shards=worker.num_workers, index=worker.id)
        for row in source:
            text = row.get(self.text_column)
            if not isinstance(text, str) or not text:
                continue
            encoded = self.tokenizer(text, add_special_tokens=False, truncation=False)
            buffer.extend(encoded["input_ids"])
            buffer.append(self.eos_token_id)
            while len(buffer) >= self.block_size:
                input_ids = torch.tensor(buffer[: self.block_size], dtype=torch.long)
                del buffer[: self.block_size]
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(self.block_size, dtype=torch.long),
                    "labels": input_ids.clone(),
                }


def stack_batch(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([example[key] for example in examples]) for key in examples[0]}


def load_text_source(
    dataset_name: str | None,
    dataset_config: str | None,
    train_file: str | None,
    split: str,
    streaming: bool,
    shuffle_buffer: int,
    seed: int,
) -> Any:
    from datasets import load_dataset

    if train_file:
        suffix = train_file.rsplit(".", 1)[-1].lower()
        builders = {"txt": "text", "json": "json", "jsonl": "json", "parquet": "parquet"}
        if suffix not in builders:
            raise ValueError("train_file must be .txt, .json, .jsonl, or .parquet")
        source = load_dataset(builders[suffix], data_files={"train": train_file}, split="train", streaming=streaming)
    elif dataset_name:
        source = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    else:
        raise ValueError("provide --dataset-name or --train-file")

    if shuffle_buffer > 0:
        if streaming:
            source = source.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            source = source.shuffle(seed=seed)
    return source
