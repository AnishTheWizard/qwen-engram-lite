from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from math import ceil
from pathlib import Path
import json
import time

import torch
from torch.utils.data import DataLoader

from .configuration import EngramConfig
from .data import PackedTokenDataset, load_text_source, stack_batch
from .modeling import EngramQwenForCausalLM


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Stage-A training for naive Engram + frozen Qwen")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-name", default="HuggingFaceFW/fineweb")
    source.add_argument("--train-file")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--streaming", action=BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)

    parser.add_argument("--output-dir", default="outputs/engram-qwen-0.5b")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-process micro-batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-tokens", type=int, default=5_000_000)
    parser.add_argument("--max-steps", type=int, default=None, help="Overrides --max-train-tokens")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--engram-layers", default="2,12", help="Zero-based decoder layer indices")
    parser.add_argument("--ngram-orders", default="2,3")
    parser.add_argument("--hash-heads", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=65_536)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--backbone-dtype", choices=("float32", "float16", "bfloat16", "auto"), default="auto")
    return parser


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"expected comma-separated integers, got {value!r}") from error


def _dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }[name]


def _optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or "norm" in name else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def train(args: Namespace) -> None:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    engram_config = EngramConfig(
        layer_indices=_csv_ints(args.engram_layers),
        ngram_orders=_csv_ints(args.ngram_orders),
        num_hash_heads=args.hash_heads,
        num_buckets=args.buckets,
        embedding_dim=args.embedding_dim,
        hash_seed=args.seed,
        gate_bias=args.gate_bias,
    )
    load_kwargs = {"torch_dtype": _dtype(args.backbone_dtype), "low_cpu_mem_usage": True}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = EngramQwenForCausalLM.from_pretrained(args.model, engram_config, **load_kwargs)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.backbone.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")
    source = load_text_source(
        args.dataset_name,
        args.dataset_config,
        args.train_file,
        args.split,
        args.streaming,
        args.shuffle_buffer,
        args.seed,
    )
    packed = PackedTokenDataset(source, tokenizer, args.text_column, args.block_size, tokenizer.eos_token_id)
    dataloader = DataLoader(
        packed,
        batch_size=args.batch_size,
        collate_fn=stack_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    tokens_per_step = (
        args.block_size
        * args.batch_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes
    )
    max_steps = args.max_steps or ceil(args.max_train_tokens / tokens_per_step)
    warmup_steps = int(max_steps * args.warmup_ratio)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    accelerator.print(
        json.dumps(
            {
                "backbone": args.model,
                "engram_layers": list(engram_config.layer_indices),
                "memory_parameters_per_layer": engram_config.memory_parameters_per_layer,
                "total_trainable_parameters": model.trainable_parameter_count(),
                "tokens_per_optimizer_step": tokens_per_step,
                "max_steps": max_steps,
            },
            indent=2,
        )
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output_dir = Path(args.output_dir)
    update_step = 0
    running_loss = 0.0
    running_micro_steps = 0
    started = time.monotonic()
    exhausted = True
    for batch in dataloader:
        with accelerator.accumulate(model):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += accelerator.gather(loss.detach()).mean().item()
        running_micro_steps += 1
        if not accelerator.sync_gradients:
            continue

        update_step += 1
        if update_step % args.log_every == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            accelerator.print(
                f"step={update_step}/{max_steps} loss={running_loss / running_micro_steps:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} tokens/s={update_step * tokens_per_step / elapsed:.0f}"
            )
            running_loss = 0.0
            running_micro_steps = 0

        if args.save_every > 0 and update_step % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_adapter(output_dir / f"checkpoint-{update_step}", args.model)

        if update_step >= max_steps:
            exhausted = False
            break

    if exhausted and update_step < max_steps:
        accelerator.print(f"dataset exhausted after {update_step} optimizer steps (requested {max_steps})")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_adapter(output_dir / "final", args.model)
        (output_dir / "train_args.json").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    accelerator.print(f"saved final Engram adapter to {output_dir / 'final'}")


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
