# ML LoRA AX Lab

This project demonstrates how to fine-tune a Large Language Model (Gemma-2-9b-it) using LoRA (Low-Rank Adaptation) on Apple Silicon (M-series chips) with the `mlx-lm` library.

The goal was to inject specific knowledge about a fictional company "SolverX" into the model.

## Project Structure

- `adapters/`: Contains the fine-tuned LoRA adapter weights for Gemma 9B.
- `adapters_solverx_cpt_8bit/`: Contains the CPT LoRA adapter weights for HyperCLOVA X 32B (8-bit).
- `data_mlx/`: Training and validation data in JSONL format compatible with `mlx-lm`.
- `data_solverx_cpt/`: Raw text data for Continuous Pre-training (CPT).
- `data_solverx_sft/`: Chat-format data for Supervised Fine-tuning (SFT).
- `models/`: Directory for large model weights.
    - `HyperCLOVAX-SEED-Think-32B-Text-8bit/`: The 8-bit quantized text-only version of HyperCLOVA X.
- `convert_hyperclova.py`: Script to extract text model from VLM and quantize to 8-bit.
- `train_with_early_stopping.py`: Custom training script with Early Stopping support.
- `train_solverx_cpt_8bit.sh`: Shell script to run CPT on the 8-bit model.
- `verify_cpt_completion.py`: Script to verify CPT knowledge injection via sentence completion.
- `test_quantized_inference.py`: Script to test inference on the 8-bit model.
- `prepare_solverx_sft_data.py`: Script to convert CPT data to SFT format.
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

### 2. Data Preparation (Gemma 9B)
- **Source**: `solverx_knowledge.jsonl` containing facts about SolverX.
- **Process**: Converted facts into a chat format (User Question -> Assistant Answer) using `prepare_mlx_data.py`.
- **Output**: `data_mlx/train.jsonl` and `data_mlx/valid.jsonl`.

### 3. Fine-tuning (LoRA) - Gemma 9B
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

### 4. Evaluation & Comparison (Gemma 9B)
We compared the Base Model vs. Fine-tuned Model on specific questions about SolverX.

| Question | Base Model Response | Fine-tuned Model Response |
| :--- | :--- | :--- |
| **Where is SolverX HQ?** | "Sorry, I don't have real-time info..." | **"SolverX의 본사는 서울 강남구 서초동에 위치한다."** (Correct) |
| **What is the core product?** | "SolverX" (Hallucination) | **"SolverX의 핵심 제품 이름은 SolverX Fusion이다."** (Correct) |
| **Behavior on low confidence?** | (Generic explanation) | **"SolverX Fusion은 신뢰도 점수가 낮을 때 기존 솔버 호출을 자동으로 제안한다."** (Correct) |

### 5. General Capabilities Verification (Gemma 9B)
We verified that the model retains its original general knowledge while learning new specific facts (avoiding catastrophic forgetting).

**Test Script**: `verify_general_performance.py`

| Category | Question | Result |
| :--- | :--- | :--- |
| **General Knowledge** | "대한민국의 수도는 어디인가요?" | **Correct** ("서울이다") |
| **General Knowledge** | "하늘이 파란 이유?" | **Correct** (Explains light scattering) |
| **Coding Ability** | "Python Hello World code" | **Correct** (Generates correct code) |
| **Injected Knowledge** | "SolverX HQ Location?" | **Correct** ("서울 강남구 서초동") |

**Conclusion**: The LoRA fine-tuning successfully injected new knowledge without degrading the model's pre-existing capabilities.

### 6. HyperCLOVA X 32B Experiment (Apple Silicon)

We extended the experiment to a much larger model, **HyperCLOVA X 32B**, to test feasibility on Apple Silicon (MacBook Pro M3 Max 48GB).

#### A. Model Conversion & Quantization
- **Challenge**: The original model is a VLM (Vision-Language Model) and 16-bit (~64GB), which exceeds the 48GB memory limit and is not directly supported by `mlx-lm`.
- **Solution**:
    1.  **Extraction**: Extracted only the text backbone (Llama-compatible) from the VLM.
    2.  **Quantization**: Converted the model to **8-bit** using a custom script (`convert_hyperclova.py`).
    3.  **Result**: Reduced model size to **~33GB**, allowing it to run on a 48GB Mac.

#### B. Continuous Pre-training (CPT) with LoRA
- **Objective**: Inject SolverX domain knowledge into the 8-bit quantized model.
- **Method**: QLoRA (Quantized LoRA) with Early Stopping.
- **Data**: Raw text sentences about SolverX (`data_solverx_cpt`).
- **Training**:
    - Script: `train_with_early_stopping.py`
    - Config: LoRA Rank 4, Batch Size 4, LR 1e-5.
    - Result: Early stopping triggered at iteration 90 (Val Loss ~2.4).
- **Verification**:
    - **Sentence Completion**: The model perfectly completed sentences like "SolverX는 대부분의 고객에게..." -> "베타 PINN 모드 대신 서러게이트 모드를 추천한다."
    - **Chat Capability**: The model learned the *facts* but struggled to answer *questions* in a chat format because CPT only teaches text patterns, not dialogue.

#### C. Next Steps: Supervised Fine-tuning (SFT)
- To fix the chat capability issue, we prepared a second stage of training (SFT).
- **Process**: Converted CPT text data into ChatML format (`User: Question -> Assistant: Answer`) using `prepare_solverx_sft_data.py`.
- **Plan**: Train a new adapter on top of the CPT model using this chat data.

### 7. Insights: Memorization vs. Reasoning
Through this project, we observed interesting behaviors regarding how LLMs learn new knowledge:

1.  **Memorization as a Feature**:
    - The model effectively "memorized" the specific facts about SolverX (e.g., HQ location).
    - Unlike simple database retrieval, the model demonstrates **semantic generalization**. It can answer questions about "SolverX's neighborhood" even if the training data only mentioned "Seocho-dong", linking the two concepts using its pre-trained knowledge.

2.  **Future Direction: Neuro-Symbolic AI (Ontology)**:
    - **Limitation**: The LoRA-tuned model may hallucinate when asked about SolverX facts not present in the training data.
    - **Solution**: Integrating an **Ontology (Knowledge Graph)** or implementing **GraphRAG**.
    - **Concept**: While LoRA handles the natural language generation and domain-specific tone, the Ontology provides a structured logic layer. This allows the system to infer answers (e.g., "If SolverX is in Seocho-dong, and Seocho-dong is in Seoul, then SolverX is in Seoul") even if that specific fact wasn't explicitly trained.

3.  **Side Effects: The "Tinted Glass" Effect (Overfitting in LoRA)**:
    - **Observation**: When asked a general question (e.g., "Python sort function"), the fine-tuned model sometimes hallucinated a SolverX-related answer.
    - **Cause**: Even though LoRA freezes base weights, the adapter weights can become so dominant that they "overshadow" original knowledge. The model learned that "All answers must be about SolverX" because the training data was 100% domain-specific.
    - **Solution**: To prevent this **Catastrophic Forgetting**, we should use **Data Mixing** (mixing general chat data with domain data) or adjust the LoRA rank/alpha parameters to balance the influence.
    - **Concrete Example (MAB-TS Implementation)**:
        - **Question**: "Implement MAB-TS algorithm in Python."
        - **Base Model**: Correctly provided Python code using `numpy`.
        - **Fine-tuned Model (Before Fix)**: Failed completely, outputting an unrelated sentence about SolverX ("SolverX allows users to adjust weights...").
        - **Test Script**: `test_mab_ts.py`

4.  **Solution Implemented: Data Mixing**:
    - We added ~15 general knowledge Q&A pairs (Python coding, common sense, greetings) to the training data.
    - **Result**: The model successfully recovered its general capabilities while retaining the injected SolverX knowledge.
    - **Verification**:
        - "Python sort function?" -> **Correctly explains `sort()` and `sorted()`**.
        - "SolverX HQ?" -> **Correctly answers "Seocho-dong"**.
        - "SolverX Welfare?" -> **Correctly answers "Information not public"** (Reduced hallucination).

### 8. Incremental Learning (Continuing Training)

We can continue training from an existing adapter. This is useful for:
1.  **Resuming CPT**: If training was interrupted or we want to add more steps.
2.  **Stage 2 (SFT)**: Fine-tuning the CPT model with chat data (Instruction Tuning).

#### A. Resuming Training (Same Data)
To resume training from a checkpoint, use the `--resume-adapter-file` argument.

```bash
python train_with_early_stopping.py \
    --model models/HyperCLOVAX-SEED-Think-32B-Text-8bit \
    --train \
    --data data_solverx_cpt \
    --resume-adapter-file adapters_solverx_cpt_8bit/adapters.safetensors \
    --adapter-path adapters_solverx_cpt_8bit_resumed
```

#### B. Stage 2: SFT on top of CPT (Different Data)
To perform Supervised Fine-tuning (SFT) using the knowledge learned during CPT, we load the CPT adapter and train on the SFT dataset.

```bash
python train_with_early_stopping.py \
    --model models/HyperCLOVAX-SEED-Think-32B-Text-8bit \
    --train \
    --data data_solverx_sft \
    --resume-adapter-file adapters_solverx_cpt_8bit/adapters.safetensors \
    --adapter-path adapters_solverx_sft_8bit \
    --learning-rate 1e-5 \
    --iters 500
```
*Note: This effectively initializes the LoRA layers with the CPT weights and further refines them for chat.*

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

## Recent Updates (2025-12-31)

### 5. Early Stopping Implementation
- Implemented a custom training script `train_with_early_stopping.py` that supports Early Stopping based on validation loss.
- Added arguments `--patience` and `--min-delta` to control stopping criteria.

### 6. Continual Pre-Training (CPT)
- Performed CPT on `solverx_knowledge.jsonl` data to inject knowledge directly into the model parameters (via LoRA) without chat formatting.
- **Data**: `data_solverx_cpt/` (Raw text data).
- **Script**: `train_lora_cpt_9b.sh`.
- **Results**: The model successfully learned specific definitions (e.g., "SolverX Fusion is a multi-physics model..."), though some hallucination persists for facts like establishment date.
- **Comparison**:
    - **Base Model**: Hallucinated or refused to answer.
    - **CPT Model**: Correctly answered questions about "SolverX Fusion" and "Low confidence behavior" by quoting the training text verbatim.
    - **Insight**: CPT is effective for injecting raw knowledge, but for Q&A capabilities, it might need to be combined with Instruction Tuning (SFT).

### 7. HyperCLOVA X Experiment
- Downloaded `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B`.
- Created inference script `infer_hyperclova.py` optimized for CPU execution (due to MPS memory limits on macOS).
- **Optimization**: Implemented `<think>` token banning to speed up inference by skipping the "thinking" process, which significantly reduced latency.

### Scripts Added
- `train_with_early_stopping.py`: Training script with early stopping.
- `train_lora_early_stop_9b.sh`: Shell script to run early stopping training.
- `train_lora_cpt_9b.sh`: Shell script for CPT training.
- `prepare_solverx_cpt_data.py`: Prepares raw text data for CPT.
- `download_hyperclova.py`: Downloads HyperCLOVA X model.
- `infer_hyperclova.py`: Runs inference on HyperCLOVA X.
- `compare_solverx_cpt.py`: Verifies CPT model performance.
