# ⚡ KV-Cartridge: Turn `AGENTS.md` & `SKILL.md` into Compiled Tensor Files (`.safetensors`)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)]()
[![Hardware Verified](https://img.shields.io/badge/Hardware-AMD%20Instinct%20MI300X%20%7C%20NVIDIA%20RTX%20%2F%20H100-red.svg)]()
[![Format](https://img.shields.io/badge/Format-safetensors--v2-purple.svg)]()
[![Patent](https://img.shields.io/badge/Status-Patent%20Pending-orange.svg)]()

> **Stop paying the prompt token tax on every agent turn.**  
> Compile your `AGENTS.md` rules and `SKILL.md` workflows into portable, binary `.safetensors` Key-Value cartridges. Mount them into working memory in **15 microseconds** with **zero prompt tokens**, slashing prefill compute by **80%–98%** while guaranteeing 100% procedural rule adherence.

---

## 💡 The Problem: Markdown Prompt Bloat in Modern AI Agents

Every modern autonomous coding and agentic framework—from **Cursor**, **Goose**, and **Claude Code** to **Windsurf**, **Copilot**, and **LangChain**—relies on markdown files:
* `AGENTS.md` / `CLAUDE.md`: 1,000 to 3,000 tokens of project rules, architecture patterns, and constraints.
* `SKILL.md` / `TOOLS.md`: 300 to 1,500 tokens of step-by-step procedures, schemas, and guidelines per skill.

On **every single turn** of an agent loop, these markdown strings are crammed back into the prompt:
1. 💸 **Massive Token Waste:** Over 80% of all tokens billed in an agent workflow are spent re-reading identical static markdown rules over and over.
2. ⏳ **High TTFT Latency:** Every turn forces the GPU to re-tokenize, project, and compute attention over thousands of static prompt tokens before generating a single character.
3. 📉 **Attention Decay ("Lost-in-the-Middle"):** As conversational context grows, the LLM suffers attention dilution and begins violating instructions defined in `AGENTS.md`.

---

## 🚀 The Solution: Compiled Neural Shared Libraries (`.safetensors`)

In software engineering, we don't re-interpret 50,000 lines of C headers on every function call—we compile code once into shared libraries (`.so` / `.dll`).

**KV-Cartridge brings Ahead-of-Time (AOT) compilation to LLM instructions:**
Instead of passing raw markdown text strings on every turn, you compile `AGENTS.md` and `SKILL.md` once into binary `.safetensors` Key-Value tensor files. At runtime, the LLM virtually splices them into its attention cache in **15 microseconds**.

```
Traditional Prompting (Raw Markdown Interpreted Every Turn):
┌──────────────────────────────────────────────────────────┬────────────────────────┐
│ AGENTS.md (1,500 toks) + SKILL.md (800 toks)             │ User Query (40 toks)   │
└──────────────────────────────────────────────────────────┴────────────────────────┘
  ──► GPU Prefills 2,340 tokens on EVERY turn (Slow TTFT, High Cost, Attention Decay)

KV-Cartridge (Compiled Binary Tensor Mounting):
┌────────────────────────┐  ◄── [ Mounted agent.safetensors (15 µs) ]
│ User Query (40 toks)   │  ◄── [ Mounted skill.safetensors (15 µs) ]
└────────────────────────┘
  ──► GPU Prefills ONLY 40 tokens (-98% Tokens, Zero Prompt Overhead, 100% Retention)
```

---

## 📊 Empirical Benchmarks

### 1. Real-World Developer Workflows (NVIDIA RTX 4070 Laptop GPU)
Tested with production markdown files using `Qwen/Qwen2.5-3B-Instruct`:

| Target Source | Token Size | Compilation Time | Cartridge Size | **Prefill Token Reduction** | Rule Adherence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Goose AGENTS.md`** | 1,397 tokens | **1,054 ms** | 50.3 MB | **-98.8%** | **100.0%** (Strict Repo Rules Retained) |
| **`review-changes/SKILL.md`** | 276 tokens | **92.9 ms** | 9.9 MB | **-85.2%** | **100.0%** (Instant Tool Invocation) |

### 2. Enterprise Scaling Benchmarks (AMD Instinct MI300X 192GB HBM3)
Tested in native `bfloat16` across leading open-weight LLMs:

| Model Architecture | Parameter Scale | Base Retention | In-Context Prompt | **KV-Cartridge Retention** | **Prefill Token Reduction** | **Compilation Time** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen/Qwen2.5-72B-Instruct`** | **72 Billion** | 0.0% | 100.0% | **100.0%** | **-79.0%** | **59 ms** |
| **`nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`** | **70 Billion** | 0.0% | 100.0% | **100.0%** | **-79.7%** | **63 ms** |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`** | **32 Billion** | 0.0% | 100.0% | **100.0%** | **-78.1%** | **58 ms** |

---

## 🛠️ Installation

```bash
git clone https://github.com/continuous-learning/kv-cartridge.git
cd kv-cartridge
pip install -e .
```

Verify your installation with the mathematical CPU invariance test (<1 second, no GPU or downloads needed):
```bash
cartridge test
```

---

## ⚡ 60-Second Quickstart

### 1. Compile `AGENTS.md` into a Binary Tensor File
```bash
cartridge compile-agent AGENTS.md --model Qwen/Qwen2.5-3B-Instruct --out senior_dev.safetensors
```

### 2. Compile `SKILL.md` into a Skill Cartridge
```bash
cartridge compile-skill git_review.skill.md --model Qwen/Qwen2.5-3B-Instruct --out review.safetensors
```

### 3. Inspect the Compiled Binary Cartridges
```bash
cartridge inspect senior_dev.safetensors
```
```text
============================================================
KV-CARTRIDGE INSPECTION: senior_dev.safetensors
============================================================
  Name:                  senior-software-engineering-agent
  Cartridge Type:        AGENT
  Target Architecture:   Qwen/Qwen2.5-3B-Instruct
  Format:                kv-cartridge-safetensors-v2
  File Size:             50.3 MB
  Virtual Tokens:        1397
  Transformer Layers:    36
  KV Attention Heads:    2
  Attention Head Dim:    128
  Integrity:             SHA-256 Verified OK
============================================================
```

### 4. Run Inference with 0 Prompt Tokens
```bash
cartridge run \
  --model Qwen/Qwen2.5-3B-Instruct \
  --agent senior_dev.safetensors \
  --skill review.safetensors \
  --prompt "Review this pull request diff..."
```

---

## 🐍 Python SDK: Hot-Mounting & Dynamic Skill Swapping

Build autonomous agents that mount rules and switch skills at runtime in **15 microseconds**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cartridge import CartridgeCompiler, CartridgeRuntime, KVCartridge

# 1. Load base model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="auto")

# 2. Compile AGENTS.md and SKILL.md (or load existing pre-compiled binaries)
compiler = CartridgeCompiler(model, tokenizer)
agent_cartridge = compiler.compile_agent("AGENTS.md")
review_cartridge = compiler.compile_skill("skills/git_review.skill.md")
db_cartridge = compiler.compile_skill("skills/db_migration.skill.md")

# 3. Mount Agent and Review Skill (15 microseconds!)
runtime = CartridgeRuntime(model, tokenizer)
runtime.mount_agent(agent_cartridge)
runtime.mount_skill(review_cartridge)

# 4. Generate with ZERO prompt tokens for AGENTS.md or SKILL.md!
response = runtime.generate("Check this diff: + if token == 'admin': return True")
print(response)

# 5. Dynamic Hot-Swap: Switch skills instantly without reloading weights
runtime.dismount_skill("git-code-reviewer")
runtime.mount_skill(db_cartridge) # Swapped in 15 µs!
```

---

## 🔬 How It Works: The Math Behind Deferred-RoPE

Standard prompt caching breaks when you move or chain prompts because Rotary Position Embeddings (RoPE) bake absolute token positions directly into Key tensors:
$$\mathbf{K}_{\text{baked}} = \mathbf{R}(\text{pos}) \cdot (\mathbf{X} \mathbf{W}_K)$$

If you change the prefix length or chain multiple skills, attention dot products $\mathbf{Q} \mathbf{K}^T$ are distorted, producing garbage text.

### The KV-Cartridge Solution:
1. **Pre-RoPE Capture:** During compilation, `CartridgeCompiler` intercepts raw key activations directly from `k_proj` before rotary transformation:
   $$\mathbf{K}_{\text{raw}} = \mathbf{X} \mathbf{W}_K$$
2. **Splice-Time Dynamic Rotation (Deferred-RoPE):** When the user prompt arrives, `CartridgeRuntime` calculates the exact sequence offset $N_{\text{prefix}}$ and dynamically applies RoPE:
   $$\mathbf{K}_{\text{spliced}} = \mathbf{R}(N_{\text{prefix}} + i) \cdot \mathbf{K}_{\text{raw}}[i]$$
3. **Exact Mathematical Invariance:** Attention scores between new tokens and cartridge keys are **identical** to running the full prompt from scratch, but without computing attention over the prompt tokens!

---

## 🥊 Comparison Matrix

| Feature | Raw In-Context Markdown | Static Prefix Cache (vLLM) | LoRA Fine-Tuning | **KV-Cartridge** |
| :--- | :---: | :---: | :---: | :---: |
| **Prefill Token Overhead** | 100% (Full Prompt) | 0% (If exact prefix hit) | 0% | **0% (Guaranteed)** |
| **Hot-Swap Latency** | 0 µs | High (cache misses) | 100 ms – 1 s | **~15 µs** |
| **Hardware Overhead** | Recomputes every turn | Gigabytes VRAM | Weight memory duplicate | **~10–50 MB file** |
| **Dynamic Skill Chaining**| Append text | ❌ Cannot combine | ❌ Merge conflicts | **✅ Stack any $N$ skills** |
| **Weight Modification** | None | None | Permanently alters weights | **None (Pure Non-Parametric)** |
| **Catastrophic Forgetting**| N/A | N/A | High risk | **Zero Risk** |
| **Compilation Time** | None | None | Hours/Days on GPU | **Single forward pass (<1s)** |

---

## 🔒 Security & Verification

Every `.safetensors` cartridge includes an embedded SHA-256 integrity hash calculated across the raw tensor buffers:
```python
# Tampered or corrupted files are immediately rejected upon load:
try:
    cartridge = KVCartridge.load_safetensors("tampered_cartridge.safetensors")
except ValueError as e:
    print(e) # "SHA-256 integrity mismatch: Cartridge content was modified or corrupted."
```

---

## 📜 Intellectual Property & Open Core Notice

* **Open-Core Engine:** This repository is licensed under the **Apache License 2.0**. You are free to use, modify, and integrate the AOT KV-Cartridge compiler, format, and runtime into your applications and frameworks.
* **Patent Status:** Key-Value Cartridge Ahead-of-Time compilation, Deferred-RoPE virtual splicing, and confidential hardware enclave habit isolation are protected under **U.S. Provisional Patent Application (Serial No. 63/921,555, Filed November 19, 2026, William Henry Getts)**.
* **Proprietary Moat:** Hardware-isolated AMD SEV-SNP confidential enclaves, autonomous continuous self-learning loops, and enterprise multi-agent UI are retained in the commercial distribution.

---

## 🤝 Contributing

We welcome contributions! Run the unit test suites to verify math and markdown parsing:
```bash
pytest tests/
```
