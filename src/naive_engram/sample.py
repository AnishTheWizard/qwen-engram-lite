from __future__ import annotations

from argparse import ArgumentParser

import torch

from .modeling import EngramQwenForCausalLM


def main() -> None:
    parser = ArgumentParser(description="Generate from a trained Engram adapter")
    parser.add_argument("adapter_dir")
    parser.add_argument("--prompt", default="The surprising thing about memory is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = EngramQwenForCausalLM.from_adapter(args.adapter_dir, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(model.backbone.config._name_or_path)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
