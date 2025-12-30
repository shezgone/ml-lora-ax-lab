# ML LoRA AX Lab

This project demonstrates how to fine-tune a Large Language Model (Gemma-2-9b-it) using LoRA (Low-Rank Adaptation) on Apple Silicon (M-series chips) with the `mlx-lm` library.

The goal was to inject specific knowledge about a fictional company "SolverX" into the model.

## Project Structure

- `adapters/`: Contains the fine-tuned LoRA adapter weights.
- `data_mlx/`: Training and validation data in JSONL format compatible with `mlx-lm`.
- `solverx_knowledge.jsonl`: Original raw knowledge data.
- `prepare_mlx_data.py`: Script to convert raw data into chat-format training data.
- `infer_gemma.py`: Script to run inference with the base model (before tuning).
- `infer_gemma_lora.py`: Script to run inference with the fine-tuned model.
- `compare_models.py`: Script to compare responses between the base and fine-tuned models.
- `verify_general_performance.py`: Script to verify that the model retains general knowledge while learning new facts.

## Workflow Summary

### 1. Environment Setup
- Created a Python virtual environment (`.venv`).
- Installed `mlx-lm`, `transformers`, `huggingface_hub`, and other dependencies.
- Authenticated with Hugging Face to access the gated model `google/gemma-2-9b-it`.

### 2. Data Preparation
- **Source**: `solverx_knowledge.jsonl` containing facts about SolverX.
- **Process**: Converted facts into a chat format (User Question -> Assistant Answer) using `prepare_mlx_data.py`.
- **Output**: `data_mlx/train.jsonl` and `data_mlx/valid.jsonl`.

### 3. Fine-tuning (LoRA)
- **Model**: `google/gemma-2-9b-it`
- **Framework**: `mlx-lm`
- **Command**:
  ```bash
  python -m mlx_lm.lora \
      --model google/gemma-2-9b-it \
      --train \
      --data data_mlx \
      --batch-size 1 \
      --iters 300 \
      --learning-rate 1e-5 \
      --adapter-path adapters \
      --save-every 100
  ```
- **Result**: Training loss decreased significantly (from ~3.5 to ~0.15), indicating successful adaptation.

### 4. Evaluation & Comparison
We compared the Base Model vs. Fine-tuned Model on specific questions about SolverX.

| Question | Base Model Response | Fine-tuned Model Response |
| :--- | :--- | :--- |
| **Where is SolverX HQ?** | "Sorry, I don't have real-time info..." | **"SolverX의 본사는 서울 강남구 서초동에 위치한다."** (Correct) |
| **What is the core product?** | "SolverX" (Hallucination) | **"SolverX의 핵심 제품 이름은 SolverX Fusion이다."** (Correct) |
| **Behavior on low confidence?** | (Generic explanation) | **"SolverX Fusion은 신뢰도 점수가 낮을 때 기존 솔버 호출을 자동으로 제안한다."** (Correct) |

### 5. General Capabilities Verification
We verified that the model retains its original general knowledge while learning new specific facts (avoiding catastrophic forgetting).

**Test Script**: `verify_general_performance.py`

| Category | Question | Result |
| :--- | :--- | :--- |
| **General Knowledge** | "대한민국의 수도는 어디인가요?" | **Correct** ("서울이다") |
| **General Knowledge** | "하늘이 파란 이유?" | **Correct** (Explains light scattering) |
| **Coding Ability** | "Python Hello World code" | **Correct** (Generates correct code) |
| **Injected Knowledge** | "SolverX HQ Location?" | **Correct** ("서울 강남구 서초동") |

**Conclusion**: The LoRA fine-tuning successfully injected new knowledge without degrading the model's pre-existing capabilities.

## How to Run

1. **Setup Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install mlx-lm transformers huggingface_hub
   ```

2. **Set Hugging Face Token**:
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

3. **Run Inference (Fine-tuned)**:
   ```bash
   python infer_gemma_lora.py
   ```

4. **Run Comparison**:
   ```bash
   python compare_models.py
   ```
