"""Example 01: Ahead-of-Time (AOT) Cartridge Compilation.

Compiles a behavioral habit (e.g., Strict JSON Schema Enforcement)
into a portable, zero-token .safetensors Key-Value cartridge.

Usage:
    python examples/01_compile_cartridge.py [--model <model_id>]
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cartridge import CartridgeCompiler


def main():
    parser = argparse.ArgumentParser(description="Compile an instruction into a KV cartridge.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID or path (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output",
        default="strict_json.safetensors",
        help="Target output file (default: strict_json.safetensors)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading base model '{args.model}' onto {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # The instruction text defining the habit
    instruction = (
        "CRITICAL POLICY: Respond strictly with a valid JSON object. "
        "Do not include any conversational filler, markdown fences, or text outside the JSON. "
        "The JSON must contain keys: 'status', 'summary', and 'data'."
    )

    print("\nCompiling habit into AOT Key-Value Cartridge via carrier pass...")
    t0 = time.perf_counter()
    compiler = CartridgeCompiler(model, tokenizer, device=device)
    cartridge = compiler.compile(
        name="strict_json",
        instruction_text=instruction,
        description="Enforces strict JSON schema output with status, summary, and data keys.",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Compilation finished in {elapsed_ms:.1f} ms!")
    print(f"Virtual token length: {cartridge.metadata.virtual_token_length} tokens")

    # Serialize to .safetensors format with cryptographic SHA-256 header
    cartridge.save_safetensors(args.output)
    print(f"Cartridge saved to '{args.output}' ({cartridge.metadata.file_size_kb:.1f} KB)")
    print("\nThis cartridge can now be mounted dynamically with 0 system prompt tokens!")


if __name__ == "__main__":
    main()
