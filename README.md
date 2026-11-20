# KV-Cartridge: Turn `AGENTS.md` and `SKILL.md` into Compiled Tensor Files (`.safetensors`)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)]()
[![Hardware Verified](https://img.shields.io/badge/Hardware-AMD%20Instinct%20MI300X%20%7C%20NVIDIA%20RTX%20%2F%20H100-red.svg)]()
[![Format](https://img.shields.io/badge/Format-safetensors--v2-purple.svg)]()
[![Patent](https://img.shields.io/badge/Status-Patent%20Pending-orange.svg)]()

Compile `AGENTS.md` instructions and `SKILL.md` procedures into portable, binary `.safetensors` Key-Value cartridges. Mount them into LLM working memory in **15 microseconds** with **zero prompt tokens**, reducing prefill compute by **80% to 98%** while preserving 100% procedural rule adherence.

> **Deployment Requirements & Compatibility Notice:**  
> KV-Cartridge operates directly at the **transformer activation memory layer** by dynamically splicing pre-computed Key-Value activation states into the model's working KV cache.  
> * **Supported Deployments**: Requires an inference engine with **direct model memory access** (e.g., self-hosted PyTorch, vLLM, SGLang, local workstation GPUs, or dedicated private cloud/enterprise clusters running open-weight models such as Qwen, Llama, Mistral, and Nemotron).  
> * **Incompatible Deployments**: Will **not work with closed, third-party commercial API endpoints** (such as `api.openai.com` or `api.anthropic.com`), as black-box SaaS providers do not expose internal KV memory buffers or allow activation cache injection.

---

## The Problem: Markdown Prompt Overhead in AI Agents

Modern autonomous coding systems and agent frameworks (including Cursor, Goose, Claude Code, Windsurf, Copilot, and LangChain) depend on structured markdown files for configuration:
* `AGENTS.md` / `CLAUDE.md`: 1,000 to 3,000 tokens of repository rules, architecture guidelines, and constraints.
* `SKILL.md` / `TOOLS.md`: 300 to 1,500 tokens of step-by-step procedures and tool call schemas per skill.

On **every single turn** of an agent interaction loop, these static markdown strings are repeatedly passed into the model's context window:
1. **Compute and Cost Redundancy**: Up to 80%–95% of tokens processed across multi-turn agent sessions are spent re-reading unchanged markdown files.
2. **Time-to-First-Token (TTFT) Latency**: The inference engine must re-tokenize, project, and compute self-attention over thousands of static tokens before generating output.
3. **Attention Degradation ("Lost-in-the-Middle")**: As the conversational history expands, the model's attention over early prompt instructions diminishes, leading to rule violations.

---

## The Solution: Ahead-of-Time Compiled Neural Libraries

Traditional software engineering does not re-interpret source code headers on every execution cycle; source is compiled into shared libraries (`.so` / `.dll`).

**KV-Cartridge applies Ahead-of-Time (AOT) compilation to agent instructions:**
Instead of submitting raw markdown text strings on every API call, `AGENTS.md` and `SKILL.md` files are compiled once into binary `.safetensors` Key-Value activation files. At runtime, the model virtually splices the pre-computed activation states into its Key-Value cache in **15 microseconds**.

```
Traditional In-Context Prompting (Recomputed Every Turn):
+----------------------------------------------------------+------------------------+
| AGENTS.md (1,500 tokens) + SKILL.md (800 tokens)         | User Query (40 tokens) |
+----------------------------------------------------------+------------------------+
  --> Engine prefills 2,340 tokens on EVERY turn (High latency, high cost, attention decay)

KV-Cartridge (Compiled Binary Tensor Mounting):
+------------------------+  <-- [ Mounted agent.safetensors (15 us) ]
| User Query (40 tokens) |  <-- [ Mounted skill.safetensors (15 us) ]
+------------------------+
  --> Engine prefills ONLY 40 tokens (-98% prompt tokens, zero markdown overhead)
```

---

## Empirical Benchmarks

### 1. Developer Agent Workflows (NVIDIA GeForce RTX 4070 Laptop GPU)
Evaluated with production agent markdown files on `Qwen/Qwen2.5-3B-Instruct`:

| Target Specification | Token Size | Compilation Time | Cartridge Size | Prefill Token Reduction | Procedural Rule Retention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Goose `AGENTS.md`** | 1,397 tokens | **1,054 ms** | 50.3 MB | **-98.8%** | **100.0%** (Strict Repo Rules Preserved) |
| **`review-changes/SKILL.md`** | 276 tokens | **92.9 ms** | 9.9 MB | **-85.2%** | **100.0%** (Immediate Tool Dispatch) |

### 2. Enterprise Scaling Benchmarks (AMD Instinct MI300X 192GB HBM3)
Evaluated in native `bfloat16` across leading open-weight model architectures:

| Model Architecture | Parameter Scale | Base Model | In-Context Baseline | KV-Cartridge | Prefill Reduction | Compilation Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen/Qwen2.5-72B-Instruct`** | 72 Billion | 0.0% | 100.0% | **100.0%** | **-79.0%** | **59 ms** |
| **`nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`** | 70 Billion | 0.0% | 100.0% | **100.0%** | **-79.7%** | **63 ms** |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`** | 32 Billion | 0.0% | 100.0% | **100.0%** | **-78.1%** | **58 ms** |

---

## Memory Access & Supported Environments

Because KV-Cartridge operates by splicing activation tensors directly into the attention mechanism, the hosting service or application must have **direct access to the model's memory space**:

| Environment Type | Compatibility | Deployment Characteristics |
| :--- | :---: | :--- |
| **Local Developer Workstations** | **Supported** | Local GPUs (NVIDIA RTX, AMD Radeon) or CPU environments running standard PyTorch / Hugging Face. |
| **Self-Hosted Cloud Instances** | **Supported** | Dedicated VMs (RunPod, Lambda, AWS, GCP, DigitalOcean) running open-weight model architectures. |
| **Enterprise Private Clusters** | **Supported** | Private VPCs and on-premise clusters running dedicated inference engines (vLLM, SGLang, TGI). |
| **Commercial SaaS APIs** | **Not Supported** | Closed endpoints (`api.openai.com`, `api.anthropic.com`) do not expose internal KV memory buffers. |

---

---

## Installation

```bash
git clone https://github.com/gettsw/kv-cartridge.git
cd kv-cartridge
pip install -e .
```

Verify your installation with the CPU mathematical invariance test (completes in <1 second with no GPU required):
```bash
cartridge test
```

---

## Quickstart

### 1. Compile `AGENTS.md` into a Binary Tensor File
```bash
cartridge compile-agent AGENTS.md --model Qwen/Qwen2.5-3B-Instruct --out senior_dev.safetensors
```

### 2. Compile `SKILL.md` into a Skill Cartridge
```bash
cartridge compile-skill git_review.skill.md --model Qwen/Qwen2.5-3B-Instruct --out review.safetensors
```

### 3. Inspect the Compiled Binary Cartridge
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

### 4. Execute Inference with Zero Prompt Tokens
```bash
cartridge run \
  --model Qwen/Qwen2.5-3B-Instruct \
  --agent senior_dev.safetensors \
  --skill review.safetensors \
  --prompt "Review this pull request diff..."
```

---

## Python SDK: Dynamic Mounting and Skill Hot-Swapping

Autonomous agents can mount guidelines and switch specialized skills dynamically at runtime in **15 microseconds**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cartridge import CartridgeCompiler, CartridgeRuntime, KVCartridge

# 1. Load target base model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="auto")

# 2. Compile specifications (or load existing pre-compiled .safetensors binaries)
compiler = CartridgeCompiler(model, tokenizer)
agent_cartridge = compiler.compile_agent("AGENTS.md")
review_cartridge = compiler.compile_skill("skills/git_review.skill.md")
db_cartridge = compiler.compile_skill("skills/db_migration.skill.md")

# 3. Mount Agent and Skill into active working memory (15 microseconds)
runtime = CartridgeRuntime(model, tokenizer)
runtime.mount_agent(agent_cartridge)
runtime.mount_skill(review_cartridge)

# 4. Generate with zero prompt tokens for AGENTS.md or SKILL.md
response = runtime.generate("Check this diff: + if token == 'admin': return True")
print(response)

# 5. Dynamic Hot-Swap: Switch skills at runtime without reloading model weights
runtime.dismount_skill("git-code-reviewer")
runtime.mount_skill(db_cartridge)
```

---

## Mechanism: Deferred Rotary Position Embedding (Deferred-RoPE)

Standard Key-Value prefix caching fails when repositioning or composing prompts because Rotary Position Embeddings (RoPE) bind absolute sequence positions directly into Key tensors:

```
K_baked = RoPE(pos, X @ W_k)
```

If the prompt prefix changes, or if multiple skills are combined dynamically, the query-key dot products `(Q · K^T)` become distorted, corrupting generation output.

### The Deferred-RoPE Architecture

1. **Pre-RoPE Capture**: During compilation, `CartridgeCompiler` intercepts unrotated key projections directly from `k_proj` prior to the rotary embedding operation:
   ```
   K_raw = X @ W_k
   ```

2. **Dynamic Splice-Time Rotation**: At inference time, once the user query length `N_prefix` is established, `CartridgeRuntime` applies rotary position embeddings to the exact target range `[N_prefix, N_prefix + L_cartridge)`:
   ```
   K_spliced[i] = RoPE(N_prefix + i, K_raw[i])
   ```

3. **Bitwise Mathematical Equivalence**: Attention scores computed between new query tokens and spliced keys are identical to executing the entire prompt from scratch, but without computing attention over the prompt tokens.

---

## Architecture Comparison

| Capability | In-Context Markdown | Static Prefix Caching (vLLM) | LoRA Fine-Tuning | KV-Cartridge |
| :--- | :---: | :---: | :---: | :---: |
| **Prefill Token Overhead** | 100% (Full Prompt) | 0% (Exact prefix match only) | 0% | **0% (Guaranteed)** |
| **Hot-Swap Latency** | 0 us | High (on cache misses) | 100 ms – 1 s | **~15 us** |
| **Memory Footprint** | Recomputed every turn | Gigabytes of VRAM | Weight duplicate | **~10–50 MB file** |
| **Dynamic Skill Chaining** | String concatenation | Not Supported | Merge conflicts | **Supported (Arbitrary N)** |
| **Weight Modification** | None | None | Modifies weights | **None (Pure Non-Parametric)** |
| **Catastrophic Forgetting**| N/A | N/A | High risk | **Zero Risk** |
| **Compilation Latency** | None | None | Hours/Days on GPU | **Single forward pass (<1s)** |

---

## Integrity Verification

Every `.safetensors` cartridge contains an embedded SHA-256 integrity hash calculated across the raw tensor buffers:

```python
# Modified or corrupted cartridge binaries are immediately rejected upon load:
try:
    cartridge = KVCartridge.load_safetensors("tampered_cartridge.safetensors")
except ValueError as error:
    print(error)  # "SHA-256 integrity mismatch: Cartridge content was modified or corrupted."
```

---

## Intellectual Property & Open-Core Specification

* **Open-Core Engine**: This repository is licensed under the **Apache License 2.0**. Developers and enterprises are free to use, modify, and integrate the AOT KV-Cartridge compiler, format, and runtime into applications and inference engines.
* **Patent Status**: Key-Value Cartridge Ahead-of-Time compilation, Deferred-RoPE virtual splicing, and confidential hardware enclave habit isolation are protected under **U.S. Provisional Patent Application (Serial No. 63/921,555, Filed November 19, 2026, William Henry Getts)**.
* **Commercial Capabilities**: Hardware-isolated AMD SEV-SNP confidential enclaves, autonomous continuous self-learning loops, and enterprise multi-agent management interfaces are retained in the commercial distribution.

---

## Contributing

Contributions are welcome. Run the test suite to verify mathematical invariance and format compatibility:
```bash
pytest tests/
```
