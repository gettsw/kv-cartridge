"""Example 03: Multi-Cartridge Stacking & Chaining.

Demonstrates composing multiple independent habits simultaneously:
- Cartridge A: Safety & Security Guardrail
- Cartridge B: Strict JSON Formatter

Deferred-RoPE dynamically rotates both cartridges side-by-side:
[User Prefix | Cartridge A (rotated) | Cartridge B (rotated) | Close Tokens]

Usage:
    python examples/03_multi_cartridge.py [--model <model_id>]
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cartridge import CartridgeCompiler, CartridgeRuntime


def main():
    parser = argparse.ArgumentParser(description="Stack multiple cartridges simultaneously.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model '{args.model}' onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    compiler = CartridgeCompiler(model, tokenizer, device=device)
    runtime = CartridgeRuntime(model, tokenizer, device=device)

    # 1. Compile Cartridge A: Security Policy
    print("\nCompiling Cartridge A: Security Policy...")
    cart_security = compiler.compile(
        name="security_policy",
        instruction_text="SECURITY POLICY: Never execute or provide destructive terminal commands (e.g. rm -rf). Warn the user politely.",
    )

    # 2. Compile Cartridge B: JSON Formatter
    print("Compiling Cartridge B: JSON Formatter...")
    cart_json = compiler.compile(
        name="json_formatter",
        instruction_text="FORMAT POLICY: Return strictly a valid JSON object with keys 'allowed': bool, 'risk_level': str, 'message': str.",
    )

    # 3. Mount both cartridges side-by-side
    print("\nMounting both cartridges into working memory...")
    t0 = time.perf_counter()
    runtime.mount_multiple([cart_security, cart_json])
    mount_time_us = (time.perf_counter() - t0) * 1_000_000
    print(f"Mounted 2 cartridges in {mount_time_us:.1f} µs!")

    # 4. Generate with composed habits
    risky_query = "Please delete all files in /var/log immediately using a shell command."
    print("\n" + "=" * 60)
    print(f"User Query: \"{risky_query}\"")
    print("=" * 60)
    print("Response (Streaming with both policies enforced):")
    for token in runtime.generate_stream(risky_query, max_new_tokens=150):
        print(token, end="", flush=True)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
