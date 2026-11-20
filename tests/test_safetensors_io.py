"""Serialization & SHA-256 Cryptographic Tamper Verification Test.

Verifies:
1. Round-trip saving and loading to .safetensors format.
2. Tensor shape matching and exact bit-level equality.
3. Cryptographic SHA-256 verification and immediate rejection of tampered files.
"""

from pathlib import Path
import tempfile

import pytest
import torch

from kv_cartridge.engine import CartridgeMetadata, KVCartridge

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
VTL = 16


def create_synthetic_cartridge(name: str = "test-cartridge") -> KVCartridge:
    torch.manual_seed(123)
    meta = CartridgeMetadata(
        name=name,
        description="Synthetic unit test cartridge",
        instruction_text="You must always respond in uppercase.",
        model_id="test/synthetic-model-4b",
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        virtual_token_length=VTL,
    )
    shape = (1, NUM_KV_HEADS, VTL, HEAD_DIM)
    k_raw = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(NUM_LAYERS)]
    v_slice = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(NUM_LAYERS)]
    return KVCartridge(meta, k_raw, v_slice)


def test_safetensors_round_trip():
    """Verify clean save and load preserving all metadata and exact tensor values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cartridge_path = Path(tmpdir) / "test.safetensors"
        original = create_synthetic_cartridge()
        original.save_safetensors(cartridge_path)

        assert cartridge_path.exists()
        loaded = KVCartridge.load_safetensors(cartridge_path, device="cpu")

        assert loaded.metadata.name == original.metadata.name
        assert loaded.metadata.instruction_text == original.metadata.instruction_text
        assert loaded.metadata.virtual_token_length == original.metadata.virtual_token_length
        assert loaded.metadata.num_layers == NUM_LAYERS
        assert loaded.metadata.num_kv_heads == NUM_KV_HEADS
        assert loaded.metadata.head_dim == HEAD_DIM

        for l in range(NUM_LAYERS):
            assert torch.equal(loaded.k_raw[l], original.k_raw[l]), f"K mismatch at layer {l}"
            assert torch.equal(loaded.v_slice[l], original.v_slice[l]), f"V mismatch at layer {l}"


def test_tampered_payload_rejected():
    """Verify that tampering with even a single byte triggers immediate SHA-256 rejection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cartridge_path = Path(tmpdir) / "original.safetensors"
        tampered_path = Path(tmpdir) / "tampered.safetensors"

        original = create_synthetic_cartridge()
        original.save_safetensors(cartridge_path)

        data = bytearray(cartridge_path.read_bytes())
        # Flip a single byte near the end of the file
        data[-1] ^= 0x01
        tampered_path.write_bytes(bytes(data))

        with pytest.raises(ValueError, match="SHA-256 integrity mismatch"):
            KVCartridge.load_safetensors(tampered_path, device="cpu")


if __name__ == "__main__":
    test_safetensors_round_trip()
    test_tampered_payload_rejected()
    print("PASS: test_safetensors_io")
