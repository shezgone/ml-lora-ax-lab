---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
---

# SolverX LLM Fine-tuning Project
## Domain Adaptation on Apple Silicon

---

# Project Overview

- **Goal**: Inject specific domain knowledge ("SolverX") into Large Language Models.
- **Environment**: Apple Silicon (M-series chips).
- **Tools**: `mlx-lm` library for efficient LoRA fine-tuning.
- **Models**:
    - Google Gemma 2 9B
    - Naver HyperCLOVA X 32B (Quantized to 8-bit)

---

# Workflow

1. **Data Preparation**:
    - Raw Knowledge -> CPT Data
    - Q&A Pairs -> SFT Data
2. **Continual Pre-training (CPT)**:
    - Inject raw knowledge into the model.
3. **Supervised Fine-tuning (SFT)**:
    - Teach the model how to chat and answer questions.
4. **Evaluation**:
    - Verify knowledge retention and general performance.

---

# Model Architecture & Optimization

## HyperCLOVA X 32B on Mac
- **Challenge**: Original model is ~64GB (16-bit), exceeding 48GB RAM.
- **Solution**:
    1. Extract Text Backbone from VLM.
    2. Quantize to **8-bit** (~33GB).
- **Result**: Runnable on MacBook Pro M3 Max (48GB).

---

# Training Strategy: CPT vs. SFT

## Phase 1: CPT (Continual Pre-training)
- **Objective**: Knowledge Injection.
- **Data**: Raw text sentences about SolverX.
- **Outcome**: Model "knows" the facts but cannot "chat" well.

## Phase 2: SFT (Supervised Fine-tuning)
- **Objective**: Chat Alignment & Identity.
- **Data**: Q&A pairs, Multi-turn conversations.
- **Outcome**: Model can answer questions naturally using the injected knowledge.

---

# Data Engineering

- **Identity Data**: "Who are you?" -> "I am HyperCLOVA X..."
- **Knowledge Data**: "SolverX Fusion is..."
- **Multi-turn Data**:
    - Context retention.
    - Follow-up questions.
- **Data Mixing**:
    - Added general knowledge (Python, Common Sense) to prevent **Catastrophic Forgetting**.

---

# Key Insights

1. **Memorization as a Feature**:
    - LoRA effectively "memorizes" specific facts (e.g., HQ location).
2. **"Rose-tinted Glasses" Effect**:
    - Overfitting caused the model to answer *everything* as if it were about SolverX.
    - **Fix**: Mixed in general data to balance the model.
3. **Incremental Learning**:
    - Successfully resumed SFT from CPT adapters.

---

# Evaluation Results

| Question | Base Model | Fine-tuned Model |
| :--- | :--- | :--- |
| **SolverX HQ?** | "No info..." | **"Seocho-dong, Seoul"** (Correct) |
| **Core Product?** | Hallucination | **"SolverX Fusion"** (Correct) |
| **Python Sort?** | Correct Code | **Correct Code** (After Data Mixing) |

---

# Future Work

- **Ontology / GraphRAG**:
    - Combine LoRA's language skills with structured logic for better reasoning.
- **Expanded Multi-turn SFT**:
    - Further improve conversation depth.
