"""Example 02: Dynamic Mounting and Zero-Token Inference.

Demonstrates:
1. Running base model without cartridge (unstructured output).
2. Hot-mounting the compiled .safetensors cartridge in ~15 microseconds.
3. Generating with 0 system prompt tokens—achieving strict habit adherence.
4. Hot-swapping / dismounting back to base state.

Usage:
    python examples/02_mount_and_infer.py [--model <model_id>] [--cartridge <path>]
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cartridge import CartridgeRuntime, KVCartridge


def main():
    parser = argparse.ArgumentParser(description="Mount a KV cartridge and infer with zero prompt tokens.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--cartridge",
        default="strict_json.safetensors",
        help="Path to compiled cartridge (default: strict_json.safetensors)",
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

    runtime = CartridgeRuntime(model, tokenizer, device=device)
    user_query = "Summarize the solar system in 2 sentences."

    # 1. BASE GENERATION (No cartridge)
    print("\n" + "=" * 60)
    print("1. INFERENCE WITHOUT CARTRIDGE (Base Model State)")
    print("=" * 60)
    print(f"User Query: \"{user_query}\"\n")
    print("Response:")
    base_response = runtime.generate(user_query, max_new_tokens=150)
    print(base_response.strip())

    # 2. MOUNT CARTRIDGE
    print("\n" + "=" * 60)
    print("2. MOUNTING CARTRIDGE (15 Microsecond Hot-Swap)")
    print("=" * 60)
    cartridge = KVCartridge.load_safetensors(args.cartridge, device=device)

    t0 = time.perf_counter()
    runtime.mount(cartridge)
    mount_time_us = (time.perf_counter() - t0) * 1_000_000
    print(f"Mounted '{cartridge.metadata.name}' in {mount_time_us:.1f} µs!")
    print(f"Active Virtual Tokens in Memory: {cartridge.metadata.virtual_token_length}")

    # 3. CARTRIDGE GENERATION (Zero system prompt tokens over the wire)
    print("\n" + "=" * 60)
    print("3. INFERENCE WITH MOUNTED CARTRIDGE (Zero System Tokens)")
    print("=" * 60)
    print(f"User Query: \"{user_query}\"\n")
    print("Response (Streaming):")
    for delta in runtime.generate_stream(user_query, max_new_tokens=150):
        print(delta, end="", flush=True)
    print("\n")

    # 4. DISMOUNT
    runtime.dismount()
    print("Dismounted cartridge. Model restored to clean Base state.")


if __name__ == "__main__":
    main()
