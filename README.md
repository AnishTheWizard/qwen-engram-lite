# Naive Engram for Qwen2.5-0.5B

This project implements the proposed Stage-A experiment end to end:

- load the base (not Instruct) `Qwen/Qwen2.5-0.5B` checkpoint;
- freeze every original Qwen parameter and keep the backbone in evaluation mode;
- derive causal bigrams and trigrams directly from input token IDs;
- use two deterministic hash heads per n-gram and 65,536 buckets per head;
- retrieve four 64-dimensional embeddings at each token;
- fuse them through projected key/value vectors and a learned sigmoid gate;
- add the residual immediately before zero-based Qwen decoder layers 2 and 12;
- train only the Engram tables and fusion layers with ordinary next-token loss.

With the defaults, each injection has exactly `4 × 65,536 × 64 = 16,777,216`
hash-table parameters. Tables are independent at layers 2 and 12. This is a
small experimental interpretation of the diagram, not an exact reproduction
of DeepSeek's production implementation.

## Setup

Use Python 3.10–3.13. The current machine's Python 3.14 is newer than the
supported PyTorch/Transformers environment, so create a 3.12 environment first:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The first run downloads the Qwen checkpoint and the selected dataset.

## Smoke test on a local text file

Start small before using millions of tokens:

```bash
accelerate launch -m naive_engram.train \
  --train-file data/smoke.txt \
  --max-steps 20 \
  --block-size 256 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --mixed-precision bf16 \
  --backbone-dtype bfloat16 \
  --gradient-checkpointing \
  --output-dir outputs/smoke
```

For a CPU or a GPU without bfloat16 support, use `--mixed-precision no`.

## Five-million-token Stage-A run

The default data source is the streaming FineWeb 10B sample. The token limit is
converted to optimizer steps using the global batch size; it can overshoot by
at most one optimizer step.

```bash
accelerate launch -m naive_engram.train \
  --dataset-name HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --text-column text \
  --max-train-tokens 5000000 \
  --block-size 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 3e-4 \
  --mixed-precision bf16 \
  --backbone-dtype bfloat16 \
  --gradient-checkpointing \
  --output-dir outputs/engram-qwen-0.5b
```

Only compact Engram adapters are saved. A final directory contains:

```text
final/
├── engram_config.json
└── engram.safetensors
```

Increase `--max-train-tokens` to 50–100 million after the loss and gate/table
gradients look healthy. Multi-GPU training works through `accelerate launch`.

The training log prints next-token loss, learning rate, and effective token
throughput. The effective global tokens per optimizer step are:

```text
block_size × per_device_batch × gradient_accumulation × number_of_processes
```

For a clean experiment, record a baseline evaluation before training, keep the
hash seed and prompt fixed, and evaluate every saved checkpoint on the same
question IDs. Generic language-model training is not guaranteed to improve
closed-book factual QA, so the baseline comparison is essential.

## Sample from the adapter

```bash
sample-engram outputs/engram-qwen-0.5b/final \
  --prompt "The surprising thing about memory is" \
  --max-new-tokens 64
```

Generation deliberately disables Qwen's KV cache and recomputes the full token
prefix at every step. That is slower but necessary for this simple wrapper,
because each lookup needs the preceding one or two token IDs.

## TriviaQA evaluation bench

The evaluator uses TriviaQA `rc.nocontext` validation examples for closed-book
generation. Its prompt is:

```text
Question: Who was the man behind The Chipmunks?
Answer:
```

It performs greedy decoding and scores the completion against every supplied
answer alias. `exact_match` and token `f1` use the official TriviaQA
normalization (lowercase; replace punctuation; remove articles; normalize
whitespace). `contains_alias` is also reported as a lenient diagnostic, but it
is not the headline metric.

First run a fast 100-question baseline:

```bash
eval-triviaqa \
  --model Qwen/Qwen2.5-0.5B \
  --max-examples 100 \
  --batch-size 4 \
  --output-dir outputs/triviaqa-baseline
```

After training, compare the frozen baseline and adapter on exactly the same 500
examples:

```bash
eval-triviaqa \
  --adapter-dir outputs/engram-qwen-0.5b/final \
  --compare-baseline \
  --max-examples 500 \
  --batch-size 4 \
  --output-dir outputs/triviaqa-comparison
```

Set `--max-examples 0` for the full validation split. Engram decoding is slower
than the baseline because it currently recomputes the full prefix rather than
using Qwen's KV cache. Reduce `--batch-size` if memory is tight.

Each run writes auditable artifacts:

```text
outputs/triviaqa-comparison/
├── baseline_metrics.json
├── baseline_predictions.jsonl
├── engram_metrics.json
├── engram_predictions.jsonl
└── comparison.json
```

The JSONL files include the question ID, generated answer, aliases, and
per-example scores, making regressions easy to inspect. Because Qwen2.5-0.5B is
a base language model rather than an instruction-tuned QA model, absolute
TriviaQA accuracy may be modest; the controlled Engram-minus-baseline delta is
the useful measurement for this experiment.

## Important semantics

- Layer numbers are zero-based. `2,12` means before the third and thirteenth
  decoder blocks.
- Hash windows are causal and end at the current token. Positions without a
  complete n-gram, and n-grams crossing padding, contribute zero.
- Document boundaries are separated by EOS. Fixed blocks concatenate documents,
  as in ordinary continued pretraining.
- `inputs_embeds`-only calls are unsupported because token IDs are required for
  hashing.
- Checkpoints contain only the Engram weights; the unchanged Qwen base model is
  loaded from the name recorded in `engram_config.json`.

Run the unit tests with `pytest`.
