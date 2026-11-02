from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import islice
from pathlib import Path
from typing import Any
import gc
import json
import re
import string
import time

import torch

from .modeling import EngramQwenForCausalLM


TRIVIA_PUNCTUATION = set(string.punctuation + "‘’´`_")


def normalize_trivia_answer(text: str) -> str:
    """Normalization used by the official TriviaQA evaluation script."""

    text = text.lower()
    text = "".join(" " if character in TRIVIA_PUNCTUATION else character for character in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_trivia_answer(prediction)
    return float(any(normalized_prediction == normalize_trivia_answer(reference) for reference in references))


def token_f1(prediction: str, references: Sequence[str]) -> float:
    prediction_tokens = normalize_trivia_answer(prediction).split()

    def score(reference: str) -> float:
        reference_tokens = normalize_trivia_answer(reference).split()
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        overlap = sum(common.values())
        if not prediction_tokens or not reference_tokens:
            return float(prediction_tokens == reference_tokens)
        if overlap == 0:
            return 0.0
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((score(reference) for reference in references), default=0.0)


def contains_alias(prediction: str, references: Sequence[str]) -> float:
    """Diagnostic lenient score; not an official TriviaQA headline metric."""

    prediction_tokens = normalize_trivia_answer(prediction).split()
    for reference in references:
        reference_tokens = normalize_trivia_answer(reference).split()
        width = len(reference_tokens)
        if width and any(prediction_tokens[i : i + width] == reference_tokens for i in range(len(prediction_tokens) - width + 1)):
            return 1.0
    return 0.0


def answer_aliases(row: dict[str, Any]) -> list[str]:
    """Read both the current nested schema and the older flattened schema."""

    answer = row.get("answer", row)
    aliases = list(answer.get("aliases") or [])
    value = answer.get("value")
    if value:
        aliases.append(value)
    # Preserve order while removing exact duplicates.
    return list(dict.fromkeys(str(alias) for alias in aliases if alias))


def truncate_generation(text: str, stop_strings: Sequence[str]) -> str:
    end = len(text)
    for stop in stop_strings:
        position = text.find(stop)
        if position >= 0:
            end = min(end, position)
    return text[:end].strip()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Open-domain TriviaQA evaluation for Qwen + Engram")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Baseline model or fallback base model")
    parser.add_argument("--adapter-dir", help="Trained Engram adapter directory, usually outputs/.../final")
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Evaluate the frozen base model and Engram adapter on exactly the same examples",
    )
    parser.add_argument("--dataset", default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset-config", default="rc.nocontext")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--streaming", action=BooleanOptionalAction, default=True)
    parser.add_argument("--max-examples", type=int, default=500, help="Use 0 for the entire split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--output-dir", default="outputs/triviaqa")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_rows(args: Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )
    rows: Iterable[dict[str, Any]] = dataset
    if args.max_examples > 0:
        rows = islice(rows, args.max_examples)
    selected = list(rows)
    if not selected:
        raise RuntimeError("TriviaQA selection is empty")
    return selected


def adapter_base_model(adapter_dir: str | Path) -> str:
    config_path = Path(adapter_dir) / EngramQwenForCausalLM.adapter_config_name
    return str(json.loads(config_path.read_text())["base_model_name"])


def load_model(
    model_name: str,
    adapter_dir: str | None,
    device: torch.device,
    dtype: torch.dtype,
):
    if adapter_dir:
        model = EngramQwenForCausalLM.from_adapter(adapter_dir, torch_dtype=dtype)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    return model


def evaluate_model(
    label: str,
    model_name: str,
    adapter_dir: str | None,
    rows: Sequence[dict[str, Any]],
    args: Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_name, adapter_dir, device, dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{label}_predictions.jsonl"
    predictions_file = predictions_path.open("w", encoding="utf-8")
    totals = {"exact_match": 0.0, "f1": 0.0, "contains_alias": 0.0}
    started = time.monotonic()

    try:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            prompts = [f"Question: {row['question'].strip()}\nAnswer:" for row in batch_rows]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to(device)
            prompt_width = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)

            for row, raw_prediction in zip(batch_rows, completions, strict=True):
                prediction = truncate_generation(raw_prediction, ("\n", ".", ","))
                aliases = answer_aliases(row)
                scores = {
                    "exact_match": exact_match(prediction, aliases),
                    "f1": token_f1(prediction, aliases),
                    "contains_alias": contains_alias(prediction, aliases),
                }
                for metric, value in scores.items():
                    totals[metric] += value
                predictions_file.write(
                    json.dumps(
                        {
                            "question_id": row.get("question_id"),
                            "question": row["question"],
                            "prediction": prediction,
                            "raw_prediction": raw_prediction,
                            "references": aliases,
                            **scores,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            completed = min(start + len(batch_rows), len(rows))
            if args.log_every > 0 and (completed % args.log_every < len(batch_rows) or completed == len(rows)):
                elapsed = max(time.monotonic() - started, 1e-6)
                running_em = 100 * totals["exact_match"] / completed
                print(f"[{label}] {completed}/{len(rows)} EM={running_em:.2f} examples/s={completed / elapsed:.2f}")
    finally:
        predictions_file.close()

    elapsed = time.monotonic() - started
    metrics = {
        "label": label,
        "model": model_name,
        "adapter_dir": adapter_dir,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_examples": len(rows),
        "exact_match": 100 * totals["exact_match"] / len(rows),
        "f1": 100 * totals["f1"] / len(rows),
        "contains_alias": 100 * totals["contains_alias"] / len(rows),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / max(elapsed, 1e-6),
        "predictions_file": str(predictions_path),
    }
    (output_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return metrics


def run(args: Namespace) -> list[dict[str, Any]]:
    if args.compare_baseline and not args.adapter_dir:
        raise ValueError("--compare-baseline requires --adapter-dir")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size and max-new-tokens must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    rows = load_rows(args)
    base_model = adapter_base_model(args.adapter_dir) if args.adapter_dir else args.model
    print(
        f"Loaded {len(rows)} TriviaQA examples; device={device}, dtype={dtype}, "
        f"base_model={base_model}"
    )

    metrics: list[dict[str, Any]] = []
    if not args.adapter_dir or args.compare_baseline:
        metrics.append(evaluate_model("baseline", base_model, None, rows, args, device, dtype))
    if args.adapter_dir:
        metrics.append(evaluate_model("engram", base_model, args.adapter_dir, rows, args, device, dtype))

    if len(metrics) == 2:
        comparison = {
            "baseline": metrics[0],
            "engram": metrics[1],
            "delta": {
                metric: metrics[1][metric] - metrics[0][metric]
                for metric in ("exact_match", "f1", "contains_alias")
            },
        }
        output = Path(args.output_dir) / "comparison.json"
        output.write_text(json.dumps(comparison, indent=2) + "\n")
        print("Engram minus baseline:", json.dumps(comparison["delta"], indent=2))
    return metrics


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
