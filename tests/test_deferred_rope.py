"""Mathematical Deferred-RoPE Invariance Test.

Verifies that:
1. Capturing pre-RoPE key tensors and rotating them dynamically with deferred RoPE
   bitwise-reproduces the monolithic KV cache.
2. Spliced attention QK scores remain mathematically invariant regardless of prompt length.
Runs purely on CPU with a synthetic model—no weights download required.
"""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kv_cartridge.engine import CartridgeCompiler, CartridgeRuntime, apply_rotary_pos_emb

CARTRIDGE_LEN = 8
CLOSE_LEN = 3


def make_tiny_model() -> Qwen2ForCausalLM:
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
    return Qwen2ForCausalLM(cfg).eval()


def rope_at(model: Qwen2ForCausalLM, x: torch.Tensor, offset: int) -> torch.Tensor:
    positions = torch.arange(offset, offset + x.shape[2]).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, positions)
    return apply_rotary_pos_emb(x, x, cos, sin)[0]


def test_deferred_rope_reproduces_monolithic_cache():
    """Verify pre-RoPE keys rotated at offset exactly match monolithic forward pass keys."""
    model = make_tiny_model()
    ids = torch.randint(0, 256, (1, 40))
    start, end = 10, 10 + CARTRIDGE_LEN

    compiler = CartridgeCompiler(model, None, device="cpu")
    k_raw, _ = compiler.capture_pre_rope_kv(ids, start, end)

    with torch.no_grad():
        full_cache = model(input_ids=ids, use_cache=True).past_key_values

    runtime = CartridgeRuntime(model, None, device="cpu")
    for l in range(len(k_raw)):
        rotated = runtime.deferred_rope(k_raw[l], start)
        assert torch.allclose(rotated, full_cache.layers[l].keys[:, :, start:end], atol=1e-5), (
            f"Layer {l} rotated keys differ from monolithic cache"
        )


def test_qk_scores_invariant_to_prompt_prefix_length():
    """Verify that QK attention scores between query token and cartridge keys are invariant to prefix length."""
    model = make_tiny_model()
    runtime = CartridgeRuntime(model, None, device="cpu")

    g = torch.Generator().manual_seed(100)
    k_raw = torch.randn(1, 2, CARTRIDGE_LEN, 16, generator=g)
    q = torch.randn(1, 4, 1, 16, generator=g)

    def compute_attention_scores(prefix_len: int, deferred: bool) -> torch.Tensor:
        k = runtime.deferred_rope(k_raw, prefix_len) if deferred else rope_at(model, k_raw, 0)
        q_pos = prefix_len + CARTRIDGE_LEN + CLOSE_LEN - 1
        q_rot = rope_at(model, q.expand(1, 4, 1, 16).contiguous(), q_pos)
        return (q_rot * k.repeat_interleave(2, dim=1)).sum(-1)

    lengths = [1, 7, 33, 150, 400]
    reference_scores = compute_attention_scores(lengths[0], deferred=True)

    for n in lengths[1:]:
        assert torch.allclose(compute_attention_scores(n, deferred=True), reference_scores, atol=1e-4), (
            f"Attention scores shifted at prefix length {n}"
        )

    # Negative control: static keys (unrotated) must fail
    static_scores = compute_attention_scores(lengths[-1], deferred=False)
    assert not torch.allclose(static_scores, reference_scores, atol=1e-2), (
        "Static keys should not produce invariant attention scores across variable lengths"
    )


if __name__ == "__main__":
    test_deferred_rope_reproduces_monolithic_cache()
    test_qk_scores_invariant_to_prompt_prefix_length()
    print("PASS: test_deferred_rope")
