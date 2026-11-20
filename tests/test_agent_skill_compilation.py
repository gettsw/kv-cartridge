"""Unit tests for Agent and Skill cartridge compilation, mounting, and hot-swapping."""

import tempfile
from pathlib import Path
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kv_cartridge.engine import CartridgeCompiler, CartridgeMetadata, CartridgeRuntime, KVCartridge
from kv_cartridge.markdown import AgentSpec, SkillSpec


def create_mock_cartridge(name: str, c_type: str = "generic", vtl: int = 16) -> KVCartridge:
    meta = CartridgeMetadata(
        name=name,
        description=f"Test {c_type} cartridge",
        instruction_text="Mock instructions",
        model_id="test/mock-model",
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        virtual_token_length=vtl,
        cartridge_type=c_type,
    )
    shape = (1, 2, vtl, 32)
    k_raw = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(2)]
    v_slice = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(2)]
    return KVCartridge(meta, k_raw, v_slice)


def test_agent_skill_mounting_and_hotswap():
    """Verifies agent and skill mounting, listing, and dynamic hot-swapping."""
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    runtime = CartridgeRuntime(model, None, device="cpu")

    agent = create_mock_cartridge("senior-engineer", "agent", vtl=20)
    skill_review = create_mock_cartridge("code-reviewer", "skill", vtl=15)
    skill_audit = create_mock_cartridge("db-auditor", "skill", vtl=12)

    # 1. Mount agent
    runtime.mount_agent(agent)
    state = runtime.list_mounted()
    assert state["agent"] == "senior-engineer"
    assert state["skills"] == []
    assert state["total_virtual_tokens"] == 20

    # 2. Mount skill
    runtime.mount_skill(skill_review)
    state = runtime.list_mounted()
    assert state["agent"] == "senior-engineer"
    assert state["skills"] == ["code-reviewer"]
    assert state["total_virtual_tokens"] == 35

    # 3. Mount second skill
    runtime.mount_skill(skill_audit)
    state = runtime.list_mounted()
    assert state["skills"] == ["code-reviewer", "db-auditor"]
    assert state["total_virtual_tokens"] == 47

    # 4. Dismount first skill (Hot-Swap)
    dismounted = runtime.dismount_skill("code-reviewer")
    assert dismounted is True
    state = runtime.list_mounted()
    assert state["skills"] == ["db-auditor"]
    assert state["total_virtual_tokens"] == 32

    # 5. Dismount agent
    runtime.dismount_agent()
    state = runtime.list_mounted()
    assert state["agent"] is None
    assert state["skills"] == ["db-auditor"]
    assert state["total_virtual_tokens"] == 12


def test_safetensors_cartridge_type_roundtrip():
    """Verifies that cartridge_type is preserved across safetensors serialization and verified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agent.safetensors"
        cartridge = create_mock_cartridge("my-agent", "agent", vtl=24)
        cartridge.save_safetensors(path)

        loaded = KVCartridge.load_safetensors(path, device="cpu")
        assert loaded.metadata.cartridge_type == "agent"
        assert loaded.metadata.name == "my-agent"
        assert loaded.metadata.virtual_token_length == 24


if __name__ == "__main__":
    test_agent_skill_mounting_and_hotswap()
    test_safetensors_cartridge_type_roundtrip()
    print("PASS: test_agent_skill_compilation")
