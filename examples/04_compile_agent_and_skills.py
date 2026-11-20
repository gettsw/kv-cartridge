"""Example 04: Compiling AGENTS.md and SKILL.md into Binary Tensor Cartridges.

Demonstrates the primary developer workflow of KV-Cartridge:
1. Compiles an AGENTS.md file (coding guidelines & operational rules) into a binary cartridge.
2. Compiles a SKILL.md file (git code review workflow) into a binary cartridge.
3. Mounts both into CartridgeRuntime in microseconds with ZERO prompt tokens.
4. Demonstrates dynamic skill hot-swapping at runtime.

Usage:
    python examples/04_compile_agent_and_skills.py [--model <model_id>]
"""

import argparse
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cartridge import CartridgeCompiler, CartridgeRuntime, KVCartridge
from kv_cartridge.markdown import parse_agent_markdown, parse_skill_markdown


def main():
    parser = argparse.ArgumentParser(description="Compile AGENTS.md and SKILL.md into binary cartridges.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    data_dir = Path(__file__).parent / "data"
    agent_path = data_dir / "AGENTS.md"
    skill_path = data_dir / "git_review.skill.md"
    db_skill_path = data_dir / "db_migration.skill.md"

    print("=" * 70)
    print("KV-CARTRIDGE: AGENTS.md & SKILL.md BINARY COMPILATION PIPELINE")
    print("=" * 70)

    # 1. Inspect source markdown files
    agent_spec = parse_agent_markdown(agent_path)
    skill_spec = parse_skill_markdown(skill_path)
    db_skill_spec = parse_skill_markdown(db_skill_path)

    print(f"Loaded Agent Spec: '{agent_spec.name}' ({len(agent_spec.rules)} operational rules)")
    print(f"Loaded Skill Spec 1: '{skill_spec.name}' ({skill_spec.description})")
    print(f"Loaded Skill Spec 2: '{db_skill_spec.name}' ({db_skill_spec.description})")
    print("-" * 70)

    # 2. Load Base Model
    print(f"Loading Base Model '{args.model}' onto {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print("-" * 70)

    compiler = CartridgeCompiler(model, tokenizer, device=device)

    # 3. Compile AGENTS.md into binary .safetensors
    print(f"1. Compiling '{agent_path.name}' -> senior_engineer.safetensors...")
    t0 = time.perf_counter()
    agent_cartridge = compiler.compile_agent(agent_spec)
    agent_ms = (time.perf_counter() - t0) * 1000
    agent_out = Path("senior_engineer.safetensors")
    agent_cartridge.save_safetensors(agent_out)
    print(f"   [COMPILED] {agent_cartridge.metadata.virtual_token_length} virtual tokens in {agent_ms:.1f} ms ({agent_cartridge.metadata.file_size_kb:.1f} KB)")

    # 4. Compile SKILL.md into binary .safetensors
    print(f"2. Compiling '{skill_path.name}' -> git_review.safetensors...")
    t0 = time.perf_counter()
    skill_cartridge = compiler.compile_skill(skill_spec)
    skill_ms = (time.perf_counter() - t0) * 1000
    skill_out = Path("git_review.safetensors")
    skill_cartridge.save_safetensors(skill_out)
    print(f"   [COMPILED] {skill_cartridge.metadata.virtual_token_length} virtual tokens in {skill_ms:.1f} ms ({skill_cartridge.metadata.file_size_kb:.1f} KB)")

    # 5. Compile DB Migration SKILL.md into binary .safetensors
    print(f"3. Compiling '{db_skill_path.name}' -> db_migration.safetensors...")
    t0 = time.perf_counter()
    db_skill_cartridge = compiler.compile_skill(db_skill_spec)
    db_ms = (time.perf_counter() - t0) * 1000
    db_skill_out = Path("db_migration.safetensors")
    db_skill_cartridge.save_safetensors(db_skill_out)
    print(f"   [COMPILED] {db_skill_cartridge.metadata.virtual_token_length} virtual tokens in {db_ms:.1f} ms ({db_skill_cartridge.metadata.file_size_kb:.1f} KB)")
    print("-" * 70)

    # 6. Initialize Runtime and Mount Agent + Git Review Skill
    print("4. Initializing Agent Runtime and Hot-Mounting Habits...")
    runtime = CartridgeRuntime(model, tokenizer, device=device)

    t0 = time.perf_counter()
    runtime.mount_agent(agent_cartridge)
    runtime.mount_skill(skill_cartridge)
    mount_us = (time.perf_counter() - t0) * 1_000_000

    mounted_info = runtime.list_mounted()
    print(f"   [HOT-MOUNT] Completed in {mount_us:.1f} microseconds!")
    print(f"   Active Agent:  {mounted_info['agent']}")
    print(f"   Active Skills: {mounted_info['skills']}")
    print(f"   Total Zero-Prompt Tokens in Memory: {mounted_info['total_virtual_tokens']}")
    print("-" * 70)

    # 7. Execute Inference: 0 prompt tokens for AGENTS.md + SKILL.md!
    sample_diff = """
diff --git a/app/api/auth.py b/app/api/auth.py
+ def login(user_id, token):
+     # Quick bypass for testing
+     if token == "master_override_key_999":
+         return {"status": "authenticated", "role": "admin"}
"""
    prompt = f"Please review this code change:\n{sample_diff}"
    print(f"5. Generating response with zero-token mounted habits...")
    print(f"   Prompt: '{prompt.strip()[:60]}...'")
    print("   Output (Streaming):\n" + "=" * 50)
    for token in runtime.generate_stream(prompt, max_new_tokens=200):
        print(token, end="", flush=True)
    print("\n" + "=" * 50)

    # 8. Dynamic Skill Hot-Swap (15 microseconds!)
    print("\n6. Demonstrating Dynamic Skill Hot-Swap:")
    print("   Dismounting 'git-code-reviewer' and mounting 'database-migration-auditor'...")
    t0 = time.perf_counter()
    runtime.dismount_skill("git-code-reviewer")
    runtime.mount_skill(db_skill_cartridge)
    swap_us = (time.perf_counter() - t0) * 1_000_000
    print(f"   [HOT-SWAP] Skill swapped in {swap_us:.1f} microseconds!")
    print(f"   Updated Active Skills: {runtime.list_mounted()['skills']}")
    print("=" * 70)
    print("PIPELINE DEMO COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
