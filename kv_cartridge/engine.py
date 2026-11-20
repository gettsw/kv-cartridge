"""KV-Cartridge Engine: Ahead-of-Time (AOT) Key-Value Non-Parametric Memory.

Enables instant mounting, hot-swapping, and chaining of procedural habits,
system instructions, schemas, and skills as Key-Value tensor cartridges
without prompt token overhead or weight modification.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


SAFETENSORS_FORMAT = "kv-cartridge-safetensors-v2"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Universal Rotary Position Embedding (RoPE) applicable across Llama, Qwen, Mistral, and Nemotron."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_model_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locates transformer layers across diverse causal LM architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Unable to locate transformer layers in model of type {type(model).__name__}")


def get_rotary_emb(model: torch.nn.Module) -> torch.nn.Module:
    """Locates the rotary position embedding module."""
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    if hasattr(model, "rotary_emb"):
        return model.rotary_emb
    layers = get_model_layers(model)
    if hasattr(layers[0], "self_attn") and hasattr(layers[0].self_attn, "rotary_emb"):
        return layers[0].self_attn.rotary_emb
    raise AttributeError(f"Unable to locate rotary_emb in model of type {type(model).__name__}")


def get_generation_close_text(tokenizer) -> str:
    """Extracts the exact generation prompt suffix for any chat tokenizer."""
    try:
        dummy = [{"role": "user", "content": "XYZ"}]
        full = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
        base = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
        if full.startswith(base):
            return full[len(base) :]
    except Exception:
        pass
    return "<|im_end|>\n<|im_start|>assistant\n"


def _content_sha256(k_tensors: List[torch.Tensor], v_tensors: List[torch.Tensor]) -> str:
    """Computes SHA-256 digest over pre-RoPE key and value tensor buffers."""
    digest = hashlib.sha256()
    for k, v in zip(k_tensors, v_tensors):
        for t in (k, v):
            digest.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class CartridgeMetadata:
    """Descriptive and geometric metadata for an AOT Key-Value cartridge."""
    name: str
    description: str
    instruction_text: str
    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    virtual_token_length: int
    file_size_kb: float = 0.0
    format: str = SAFETENSORS_FORMAT


class KVCartridge:
    """A serialized Key-Value habit cartridge that can be dynamically mounted into an LLM's KV cache.

    Crucial Architectural Property:
    Stores *pre-RoPE* key projections (`k_raw`) directly from `k_proj`, alongside value projections (`v_slice`).
    Rotary Position Embeddings are applied dynamically at splice-time when the runtime determines
    the cartridge's true sequence offset. This guarantees exact mathematical invariance across arbitrary
    prompt prefixes and multi-cartridge chains.
    """

    def __init__(
        self,
        metadata: CartridgeMetadata,
        k_raw: List[torch.Tensor],
        v_slice: List[torch.Tensor],
    ):
        self.metadata = metadata
        self.k_raw = k_raw
        self.v_slice = v_slice

    def save_safetensors(self, filepath: Union[str, Path]) -> None:
        """Saves cartridge to safetensors format with cryptographic SHA-256 integrity metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        k_cpu = [k.detach().cpu().to(torch.bfloat16).contiguous() for k in self.k_raw]
        v_cpu = [v.detach().cpu().to(torch.bfloat16).contiguous() for v in self.v_slice]

        tensors: Dict[str, torch.Tensor] = {}
        for l in range(len(k_cpu)):
            tensors[f"k.{l}"] = k_cpu[l]
            tensors[f"v.{l}"] = v_cpu[l]

        m = self.metadata
        header = {
            "format": SAFETENSORS_FORMAT,
            "name": m.name,
            "description": m.description,
            "instruction_text": m.instruction_text,
            "architecture": m.model_id,
            "num_layers": str(m.num_layers),
            "num_kv_heads": str(m.num_kv_heads),
            "head_dim": str(m.head_dim),
            "virtual_token_length": str(m.virtual_token_length),
            "sha256": _content_sha256(k_cpu, v_cpu),
        }

        save_file(tensors, filepath, metadata=header)
        self.metadata.file_size_kb = round(os.path.getsize(filepath) / 1024, 1)

    @classmethod
    def load_safetensors(cls, filepath: Union[str, Path], device: str = "cuda") -> "KVCartridge":
        """Loads and verifies a safetensors cartridge with SHA-256 tamper verification."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Cartridge file not found: {filepath}")

        with safe_open(filepath, framework="pt", device="cpu") as f:
            header = f.metadata() or {}
            if "sha256" not in header:
                raise ValueError(f"Missing cryptographic integrity header in {filepath}")
            if header.get("format") != SAFETENSORS_FORMAT:
                raise ValueError(
                    f"Unsupported cartridge format {header.get('format')!r} (expected {SAFETENSORS_FORMAT}): {filepath}"
                )

            num_layers = int(header["num_layers"])
            num_kv_heads = int(header["num_kv_heads"])
            head_dim = int(header["head_dim"])
            vtl = int(header["virtual_token_length"])

            k_cpu = [f.get_tensor(f"k.{l}") for l in range(num_layers)]
            v_cpu = [f.get_tensor(f"v.{l}") for l in range(num_layers)]

            expected_shape = (1, num_kv_heads, vtl, head_dim)
            for l, (k, v) in enumerate(zip(k_cpu, v_cpu)):
                if k.shape != expected_shape or v.shape != expected_shape:
                    raise ValueError(
                        f"Layer {l} tensor shape {k.shape} does not match expected {expected_shape} in {filepath}"
                    )

            actual_sha = _content_sha256(k_cpu, v_cpu)
            if actual_sha != header["sha256"]:
                raise ValueError(
                    f"SHA-256 integrity mismatch for {filepath}. Expected {header['sha256']}, got {actual_sha}."
                )

            k_raw = [k.to(device) for k in k_cpu]
            v_slice = [v.to(device) for v in v_cpu]

        metadata = CartridgeMetadata(
            name=header["name"],
            description=header.get("description", ""),
            instruction_text=header.get("instruction_text", ""),
            model_id=header["architecture"],
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            virtual_token_length=vtl,
            file_size_kb=round(os.path.getsize(filepath) / 1024, 1),
            format=SAFETENSORS_FORMAT,
        )
        return cls(metadata, k_raw, v_slice)

    def save(self, filepath: Union[str, Path]) -> None:
        """PyTorch checkpoint serialization."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": dataclasses.asdict(self.metadata),
            "k_raw": [k.detach().cpu().to(torch.bfloat16) for k in self.k_raw],
            "v_slice": [v.detach().cpu().to(torch.bfloat16) for v in self.v_slice],
        }
        torch.save(payload, filepath)
        self.metadata.file_size_kb = round(os.path.getsize(filepath) / 1024, 1)

    @classmethod
    def load(cls, filepath: Union[str, Path], device: str = "cuda") -> "KVCartridge":
        """PyTorch checkpoint deserialization."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Cartridge file not found: {filepath}")
        payload = torch.load(filepath, map_location=device, weights_only=False)
        if "k_raw" not in payload:
            raise ValueError(f"Legacy post-RoPE cartridge format detected: {filepath}")
        meta_dict = payload["metadata"]
        metadata = CartridgeMetadata(**meta_dict) if isinstance(meta_dict, dict) else meta_dict
        metadata.file_size_kb = round(os.path.getsize(filepath) / 1024, 1)
        k_raw = [k.to(device) for k in payload["k_raw"]]
        v_slice = [v.to(device) for v in payload["v_slice"]]
        return cls(metadata, k_raw, v_slice)


class CartridgeCompiler:
    """Compiles instruction texts into AOT KVCartridge objects via a single reference carrier forward pass."""

    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compile(
        self,
        name: str,
        instruction_text: str,
        description: str = "",
        carrier_context: str = "A store has 15 red balls and 25 blue balls. How many balls in total?\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ) -> KVCartridge:
        """Compiles instruction text into a portable pre-RoPE KV cartridge."""
        tok = self.tokenizer
        model = self.model

        full_prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{carrier_context} {instruction_text}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc_full = tok(full_prompt, return_tensors="pt").to(self.device)
        ids_full = enc_full.input_ids[0].tolist()

        enc_instr_alone = tok(" " + instruction_text, add_special_tokens=False).input_ids
        L_instr = len(enc_instr_alone)

        close_text = get_generation_close_text(tok)
        close_ids = tok(close_text, add_special_tokens=False).input_ids
        L_close = len(close_ids)

        end_idx = len(ids_full) - L_close
        start_idx = end_idx - L_instr

        k_raw, v_slice = self.capture_pre_rope_kv(enc_full.input_ids, start_idx, end_idx)
        num_layers = len(k_raw)
        num_kv_heads = k_raw[0].shape[1]
        head_dim = k_raw[0].shape[3]

        metadata = CartridgeMetadata(
            name=name,
            description=description,
            instruction_text=instruction_text,
            model_id=getattr(model.config, "_name_or_path", "unknown"),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            virtual_token_length=L_instr,
        )

        return KVCartridge(metadata, k_raw, v_slice)

    def capture_pre_rope_kv(
        self, input_ids: torch.Tensor, start_idx: int, end_idx: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Executes the carrier pass and hooks raw k_proj and v_proj tensors before rotary embedding."""
        layers = get_model_layers(self.model)
        attn0 = layers[0].self_attn
        num_kv_heads = self.model.config.num_key_value_heads
        head_dim = getattr(attn0, "head_dim", self.model.config.hidden_size // self.model.config.num_attention_heads)
        num_layers = len(layers)

        k_raw: List[Optional[torch.Tensor]] = [None] * num_layers
        v_raw: List[Optional[torch.Tensor]] = [None] * num_layers

        def capture(store: List[Optional[torch.Tensor]], l: int):
            def hook(_module, _inputs, out):
                t = out[:, start_idx:end_idx, :]
                store[l] = t.reshape(t.shape[0], t.shape[1], num_kv_heads, head_dim).transpose(1, 2).detach().clone()
            return hook

        handles = []
        for l, layer in enumerate(layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture(k_raw, l)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(capture(v_raw, l)))

        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        return [k for k in k_raw if k is not None], [v for v in v_raw if v is not None]


class CartridgeRuntime:
    """Manages dynamic KV-cache splicing and low-latency LLM generation with mounted cartridges."""

    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.active_cartridges: List[KVCartridge] = []

    def mount(self, cartridge: KVCartridge) -> None:
        """Mounts a single cartridge into active memory."""
        self.mount_multiple([cartridge])

    def mount_multiple(self, cartridges: List[KVCartridge]) -> None:
        """Mounts multiple distinct cartridges chained side-by-side: [Prefix | C_1 | C_2 | ... | C_M | Close]."""
        names = [c.metadata.name for c in cartridges]
        if len(set(names)) != len(names):
            raise ValueError(f"Cartridges must have unique names; got duplicates in {names}")

        cfg = self.model.config
        expected_kv_heads = cfg.num_key_value_heads
        expected_head_dim = cfg.hidden_size // cfg.num_attention_heads
        expected_layers = cfg.num_hidden_layers

        for c in cartridges:
            m = c.metadata
            if (m.num_layers, m.num_kv_heads, m.head_dim) != (expected_layers, expected_kv_heads, expected_head_dim):
                raise ValueError(
                    f"Cartridge '{m.name}' geometry (layers={m.num_layers}, kv_heads={m.num_kv_heads}, "
                    f"head_dim={m.head_dim}) does not match model (layers={expected_layers}, "
                    f"kv_heads={expected_kv_heads}, head_dim={expected_head_dim})."
                )

        self.active_cartridges = list(cartridges)

    def dismount(self) -> None:
        """Dismounts all active cartridges, returning model to clean base state."""
        self.active_cartridges = []

    def deferred_rope(self, k_raw: torch.Tensor, offset: int) -> torch.Tensor:
        """Rotates raw key tensor to target sequence positions [offset, offset + L)."""
        positions = torch.arange(offset, offset + k_raw.shape[2], device=k_raw.device).unsqueeze(0)
        rotary_emb = get_rotary_emb(self.model)
        cos, sin = rotary_emb(k_raw, positions)
        _, k_rot = apply_rotary_pos_emb(k_raw, k_raw, cos, sin)
        return k_rot

    def splice_kv(self, kv_prefix: DynamicCache, n_prefix: int) -> DynamicCache:
        """Assembles spliced DynamicCache: [User Prefix | C_1 (rotated) | C_2 (rotated) | ...]."""
        num_layers = len(self.active_cartridges[0].k_raw)
        spliced = DynamicCache()

        for l in range(num_layers):
            k_parts = [kv_prefix.layers[l].keys]
            v_parts = [kv_prefix.layers[l].values]
            offset = n_prefix

            for c in self.active_cartridges:
                k_parts.append(self.deferred_rope(c.k_raw[l].to(k_parts[0]), offset))
                v_parts.append(c.v_slice[l].to(v_parts[0]))
                offset += c.metadata.virtual_token_length

            spliced.update(torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2), l)

        return spliced

    @torch.no_grad()
    def generate_stream(
        self,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Streams autoregressive tokens from the model with active cartridges mounted."""
        tok = self.tokenizer
        model = self.model

        stops = set(
            model.generation_config.eos_token_id
            if isinstance(model.generation_config.eos_token_id, list)
            else [model.generation_config.eos_token_id]
        ) | {tok.pad_token_id}

        if not self.active_cartridges:
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(prompt_text, return_tensors="pt").to(self.device)
            out = model(**enc, use_cache=True)
            gen_cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_pos = enc.input_ids.shape[1]
        else:
            prefix_text = tok.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=False,
            )
            close_text = get_generation_close_text(tok)

            enc_prefix = tok(prefix_text, return_tensors="pt").to(self.device)
            enc_close = tok(close_text, return_tensors="pt", add_special_tokens=False).to(self.device)

            n_prefix = enc_prefix.input_ids.shape[1]
            n_close = enc_close.input_ids.shape[1]

            # 1. Forward pass on user prefix
            out_prefix = model(**enc_prefix, use_cache=True)
            kv_prefix = out_prefix.past_key_values

            # 2. Virtual splice of mounted cartridges
            spliced_cache = self.splice_kv(kv_prefix, n_prefix)

            total_virtual_tokens = sum(c.metadata.virtual_token_length for c in self.active_cartridges)

            # 3. Forward pass on generation close tokens
            start_pos = n_prefix + total_virtual_tokens
            pos_ids_close = torch.arange(start_pos, start_pos + n_close, device=self.device).unsqueeze(0)
            total_tokens = start_pos + n_close
            attn_mask_close = torch.ones((1, total_tokens), device=self.device)

            out_close = model(
                input_ids=enc_close.input_ids,
                position_ids=pos_ids_close,
                attention_mask=attn_mask_close,
                past_key_values=spliced_cache,
                use_cache=True,
            )
            gen_cache = out_close.past_key_values
            next_token = out_close.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_pos = total_tokens

        tokens: List[int] = []
        decoded_so_far = ""

        for _ in range(max_new_tokens):
            tok_id = next_token.item()
            if tok_id in stops:
                break
            tokens.append(tok_id)
            full_text = tok.decode(tokens)
            delta = full_text[len(decoded_so_far) :]
            decoded_so_far = full_text
            yield delta

            curr_pos += 1
            pos_id = torch.tensor([[curr_pos - 1]], device=self.device)
            attn_mask = torch.ones((1, curr_pos), device=self.device)

            out_step = model(
                input_ids=next_token,
                position_ids=pos_id,
                attention_mask=attn_mask,
                past_key_values=gen_cache,
                use_cache=True,
            )
            gen_cache = out_step.past_key_values
            next_token = out_step.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def generate(
        self,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Executes synchronous text generation with mounted cartridges."""
        return "".join(self.generate_stream(user_message, max_new_tokens, temperature))
