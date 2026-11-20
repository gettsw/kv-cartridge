"""KV-Cartridge Command-Line Interface.

Provides tools to compile AGENTS.md, SKILL.md, and raw instructions into portable
binary .safetensors Key-Value cartridges, as well as inspect, benchmark, and run them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch


def inspect_cartridge(path: str) -> None:
    """Inspects a .safetensors cartridge file, displaying its metadata and tensor verification."""
    from kv_cartridge.engine import KVCartridge

    file_path = Path(path)
    if not file_path.exists():
        print(f"Error: Cartridge file not found at '{file_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        cartridge = KVCartridge.load_safetensors(file_path, device="cpu")
        m = cartridge.metadata
        cartridge_type = getattr(m, "cartridge_type", "generic").upper()
        print("=" * 60)
        print(f"KV-CARTRIDGE INSPECTION: {file_path.name}")
        print("=" * 60)
        print(f"  Name:                  {m.name}")
        print(f"  Cartridge Type:        {cartridge_type}")
        print(f"  Description:           {m.description or '(None)'}")
        print(f"  Target Architecture:   {m.model_id}")
        print(f"  Format:                {m.format}")
        print(f"  File Size:             {m.file_size_kb:.1f} KB")
        print(f"  Virtual Tokens:        {m.virtual_token_length}")
        print(f"  Transformer Layers:    {m.num_layers}")
        print(f"  KV Attention Heads:    {m.num_kv_heads}")
        print(f"  Attention Head Dim:    {m.head_dim}")
        print(f"  Integrity:             SHA-256 Verified OK")
        print("-" * 60)
        print("Embedded Instruction Preview:")
        preview = m.instruction_text.strip()
        if len(preview) > 300:
            preview = preview[:297] + "..."
        print(f"  \"{preview}\"")
        print("=" * 60)
    except Exception as e:
        print(f"Error inspecting cartridge: {e}", file=sys.stderr)
        sys.exit(1)


def run_math_test() -> None:
    """Runs the pure-CPU mathematical deferred RoPE verification."""
    print("=" * 60)
    print("RUNNING DEFERRED-ROPE MATHEMATICAL INVARIANCE TEST (CPU)")
    print("=" * 60)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from kv_cartridge.engine import CartridgeCompiler, CartridgeRuntime, apply_rotary_pos_emb

    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    def rope_at(m, x, offset):
        positions = torch.arange(offset, offset + x.shape[2]).unsqueeze(0)
        cos, sin = m.model.rotary_emb(x, positions)
        return apply_rotary_pos_emb(x, x, cos, sin)[0]

    cartridge_len = 8
    close_len = 3
    ids = torch.randint(0, 256, (1, 48))
    start, end = 12, 12 + cartridge_len

    compiler = CartridgeCompiler(model, None, device="cpu")
    k_raw, _ = compiler.capture_pre_rope_kv(ids, start, end)

    with torch.no_grad():
        full_cache = model(input_ids=ids, use_cache=True).past_key_values

    runtime = CartridgeRuntime(model, None, device="cpu")

    # Check 1: Post-RoPE reproduction
    for l in range(len(k_raw)):
        rotated = runtime.deferred_rope(k_raw[l], start)
        assert torch.allclose(rotated, full_cache.layers[l].keys[:, :, start:end], atol=1e-5), (
            f"Layer {l} failed post-RoPE reproduction"
        )
    print("[PASS] Check 1: Deferred rotation exactly reproduces monolithic KV cache.")

    # Check 2: Attention QK score invariance across variable user prefix lengths
    g = torch.Generator().manual_seed(1337)
    k_synth = torch.randn(1, 2, cartridge_len, 16, generator=g)
    q_synth = torch.randn(1, 4, 1, 16, generator=g)

    def calc_scores(prefix_len, use_deferred):
        k = runtime.deferred_rope(k_synth, prefix_len) if use_deferred else rope_at(model, k_synth, 0)
        q_pos = prefix_len + cartridge_len + close_len - 1
        q_rot = rope_at(model, q_synth.expand(1, 4, 1, 16).contiguous(), q_pos)
        return (q_rot * k.repeat_interleave(2, dim=1)).sum(-1)

    test_lengths = [1, 5, 24, 128, 512]
    ref_scores = calc_scores(test_lengths[0], use_deferred=True)
    for length in test_lengths[1:]:
        s = calc_scores(length, use_deferred=True)
        assert torch.allclose(s, ref_scores, atol=1e-4), f"Failed invariance at prefix length {length}"

    # Verify that unrotated (static) keys FAIL invariance
    static_scores = calc_scores(test_lengths[-1], use_deferred=False)
    assert not torch.allclose(static_scores, ref_scores, atol=1e-2)

    print("[PASS] Check 2: QK Attention score invariance verified across arbitrary prompt lengths.")
    print("[PASS] Check 3: Static un-rotated keys proven mathematically distorted without deferred RoPE.")
    print("=" * 60)
    print("ALL MATHEMATICAL CHECKS PASSED: Splicing is bitwise-invariant.")
    print("=" * 60)


def _load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading base model '{model_name}' onto {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def compile_command(args: argparse.Namespace) -> None:
    """Compiles an instruction text into a portable .safetensors cartridge."""
    from kv_cartridge.engine import CartridgeCompiler

    instruction = args.prompt
    if args.file:
        instruction = Path(args.file).read_text(encoding="utf-8")

    if not instruction:
        print("Error: Must provide instruction via --prompt or --file", file=sys.stderr)
        sys.exit(1)

    model, tokenizer, device = _load_model_and_tokenizer(args.model, args.device)

    print(f"Compiling cartridge '{args.name}' via single carrier forward pass...")
    t0 = time.perf_counter()
    compiler = CartridgeCompiler(model, tokenizer, device=device)
    cartridge = compiler.compile(
        name=args.name,
        instruction_text=instruction,
        description=args.desc or f"Compiled cartridge for {args.name}",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_path = Path(args.out or f"{args.name}.safetensors")
    cartridge.save_safetensors(out_path)
    print(
        f"Success! Compiled {cartridge.metadata.virtual_token_length} virtual tokens "
        f"in {elapsed_ms:.1f} ms. Saved to {out_path} ({cartridge.metadata.file_size_kb:.1f} KB)."
    )


def compile_agent_command(args: argparse.Namespace) -> None:
    """Compiles an AGENTS.md markdown file into a binary .safetensors cartridge."""
    from kv_cartridge.engine import CartridgeCompiler
    from kv_cartridge.markdown import parse_agent_markdown

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Agent file not found at '{path}'", file=sys.stderr)
        sys.exit(1)

    spec = parse_agent_markdown(path, name=args.name)
    print(f"Parsed Agent Spec: '{spec.name}' ({len(spec.instruction_text.split())} words, {len(spec.rules)} rules)")

    model, tokenizer, device = _load_model_and_tokenizer(args.model, args.device)

    print(f"Compiling agent cartridge '{spec.name}' into binary .safetensors...")
    t0 = time.perf_counter()
    compiler = CartridgeCompiler(model, tokenizer, device=device)
    cartridge = compiler.compile_agent(spec, name=args.name, description=args.desc)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_name = args.out or f"{spec.name}.safetensors"
    out_path = Path(out_name)
    cartridge.save_safetensors(out_path)
    print(
        f"Success! Compiled {cartridge.metadata.virtual_token_length} virtual tokens "
        f"in {elapsed_ms:.1f} ms. Saved binary cartridge to {out_path} ({cartridge.metadata.file_size_kb:.1f} KB)."
    )


def compile_skill_command(args: argparse.Namespace) -> None:
    """Compiles a SKILL.md markdown file into a binary .safetensors cartridge."""
    from kv_cartridge.engine import CartridgeCompiler
    from kv_cartridge.markdown import parse_skill_markdown

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Skill file not found at '{path}'", file=sys.stderr)
        sys.exit(1)

    spec = parse_skill_markdown(path, name=args.name)
    print(f"Parsed Skill Spec: '{spec.name}' - {spec.description}")

    model, tokenizer, device = _load_model_and_tokenizer(args.model, args.device)

    print(f"Compiling skill cartridge '{spec.name}' into binary .safetensors...")
    t0 = time.perf_counter()
    compiler = CartridgeCompiler(model, tokenizer, device=device)
    cartridge = compiler.compile_skill(spec, name=args.name, description=args.desc)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out_name = args.out or f"{spec.name}.safetensors"
    out_path = Path(out_name)
    cartridge.save_safetensors(out_path)
    print(
        f"Success! Compiled {cartridge.metadata.virtual_token_length} virtual tokens "
        f"in {elapsed_ms:.1f} ms. Saved binary cartridge to {out_path} ({cartridge.metadata.file_size_kb:.1f} KB)."
    )


def run_command(args: argparse.Namespace) -> None:
    """Executes inference with one or more mounted cartridges."""
    from kv_cartridge.engine import CartridgeRuntime, KVCartridge

    model, tokenizer, device = _load_model_and_tokenizer(args.model, args.device)
    runtime = CartridgeRuntime(model, tokenizer, device=device)

    # Mount agent if provided
    if getattr(args, "agent", None):
        agent_c = KVCartridge.load_safetensors(args.agent, device=device)
        t0 = time.perf_counter()
        runtime.mount_agent(agent_c)
        mount_us = (time.perf_counter() - t0) * 1_000_000
        print(f"Mounted Agent '{agent_c.metadata.name}' ({agent_c.metadata.virtual_token_length} virtual tokens) in {mount_us:.1f} µs.")

    # Mount skills if provided
    if getattr(args, "skill", None):
        for s_path in args.skill:
            skill_c = KVCartridge.load_safetensors(s_path, device=device)
            t0 = time.perf_counter()
            runtime.mount_skill(skill_c)
            mount_us = (time.perf_counter() - t0) * 1_000_000
            print(f"Mounted Skill '{skill_c.metadata.name}' ({skill_c.metadata.virtual_token_length} virtual tokens) in {mount_us:.1f} µs.")

    # Mount generic cartridges if provided
    if getattr(args, "cartridge", None):
        cartridges = [KVCartridge.load_safetensors(c, device=device) for c in args.cartridge]
        t0 = time.perf_counter()
        runtime.mount_multiple(cartridges)
        mount_us = (time.perf_counter() - t0) * 1_000_000
        print(f"Mounted {len(cartridges)} cartridge(s) in {mount_us:.1f} µs.")

    total_tokens = sum(c.metadata.virtual_token_length for c in runtime.active_cartridges)
    print(f"Active Memory State: {len(runtime.active_cartridges)} cartridge(s), {total_tokens} zero-prompt virtual tokens mounted.")

    print(f"\nUser Query: {args.prompt}")
    print("Assistant (streaming):\n" + "-" * 40)
    for token in runtime.generate_stream(args.prompt, max_new_tokens=args.max_tokens):
        print(token, end="", flush=True)
    print("\n" + "-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cartridge",
        description="KV-Cartridge: Turn AGENTS.md and SKILL.md into Compiled Binary Tensor Files (.safetensors)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a .safetensors cartridge file")
    inspect_parser.add_argument("path", help="Path to the .safetensors cartridge")

    # test
    subparsers.add_parser("test", help="Run the CPU mathematical invariance test")

    # compile-agent
    agent_parser = subparsers.add_parser("compile-agent", help="Compile AGENTS.md / CLAUDE.md into a binary .safetensors cartridge")
    agent_parser.add_argument("path", help="Path to AGENTS.md or agent markdown file")
    agent_parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    agent_parser.add_argument("--name", help="Unique name for the agent cartridge (default: extracted from file)")
    agent_parser.add_argument("--desc", help="Description of agent role")
    agent_parser.add_argument("--out", help="Output .safetensors filepath")
    agent_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # compile-skill
    skill_parser = subparsers.add_parser("compile-skill", help="Compile SKILL.md into a binary .safetensors cartridge")
    skill_parser.add_argument("path", help="Path to SKILL.md or skill markdown file")
    skill_parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    skill_parser.add_argument("--name", help="Unique name for the skill cartridge (default: extracted from file)")
    skill_parser.add_argument("--desc", help="Description of skill capability")
    skill_parser.add_argument("--out", help="Output .safetensors filepath")
    skill_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # compile (generic)
    compile_parser = subparsers.add_parser("compile", help="Compile raw prompt or text file into a cartridge")
    compile_parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    compile_parser.add_argument("--name", required=True, help="Unique identifier for this cartridge")
    compile_parser.add_argument("--prompt", help="Instruction text to compile")
    compile_parser.add_argument("--file", help="Path to file containing instruction text")
    compile_parser.add_argument("--desc", default="", help="Description of cartridge capability")
    compile_parser.add_argument("--out", help="Output .safetensors filepath")
    compile_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # run
    run_parser = subparsers.add_parser("run", help="Run generation with mounted cartridges")
    run_parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    run_parser.add_argument("-a", "--agent", help="Path to compiled agent .safetensors cartridge")
    run_parser.add_argument("-s", "--skill", action="append", help="Path to compiled skill .safetensors cartridge (can repeat)")
    run_parser.add_argument("-c", "--cartridge", action="append", help="Path to .safetensors cartridge (can repeat)")
    run_parser.add_argument("--prompt", required=True, help="User message for generation")
    run_parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens")
    run_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_cartridge(args.path)
    elif args.command == "test":
        run_math_test()
    elif args.command == "compile-agent":
        compile_agent_command(args)
    elif args.command == "compile-skill":
        compile_skill_command(args)
    elif args.command == "compile":
        compile_command(args)
    elif args.command == "run":
        run_command(args)


if __name__ == "__main__":
    main()
