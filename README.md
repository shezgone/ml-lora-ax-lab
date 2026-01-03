# ML LoRA AX Lab

이 프로젝트는 Apple Silicon (M 시리즈 칩)에서 `mlx-lm` 라이브러리를 사용하여 대규모 언어 모델(Gemma-2-9b-it)을 LoRA(Low-Rank Adaptation)로 미세 조정(Fine-tuning)하는 방법을 보여줍니다.

목표는 가상의 회사 "SolverX"에 대한 특정 지식을 모델에 주입하는 것입니다.

## 프로젝트 구조

- `adapters/`: Gemma 9B용 미세 조정된 LoRA 어댑터 가중치.
- `adapters_solverx_cpt_hcx/`: HyperCLOVA X 32B (8-bit)용 CPT LoRA 어댑터 가중치.
- `adapters_solverx_sft_hcx/`: HyperCLOVA X 32B (8-bit)용 SFT LoRA 어댑터 가중치.
- `data_mlx/`: `mlx-lm`과 호환되는 JSONL 형식의 학습 및 검증 데이터.
- `data_solverx_cpt/`: 연속 사전 학습(CPT)을 위한 원시 텍스트 데이터.
- `data_solverx_sft/`: 지도 미세 조정(SFT)을 위한 대화 형식 데이터.
- `models/`: 대규모 모델 가중치 디렉토리.
    - `HyperCLOVAX-SEED-Think-32B-Text-8bit/`: HyperCLOVA X의 8비트 양자화된 텍스트 전용 버전.
- `convert_hyperclova.py`: VLM에서 텍스트 모델을 추출하고 8비트로 양자화하는 스크립트.
- `train_with_early_stopping.py`: 조기 종료(Early Stopping)를 지원하는 커스텀 학습 스크립트.
- `train_solverx_cpt_hcx.sh`: 8비트 모델에서 CPT를 실행하는 쉘 스크립트.
- `train_solverx_sft_hcx.sh`: 8비트 모델에서 SFT를 실행하는 쉘 스크립트.
- `verify_cpt_completion.py`: 문장 완성을 통해 CPT 지식 주입을 검증하는 스크립트.
- `test_quantized_inference.py`: 8비트 모델에서 추론을 테스트하는 스크립트.
- `prepare_solverx_sft_data.py`: CPT 데이터를 SFT 형식으로 변환하는 스크립트.
- `solverx_knowledge.jsonl`: 원본 원시 지식 데이터.
- `prepare_mlx_data.py`: 원시 데이터를 대화 형식 학습 데이터로 변환하는 스크립트.
- `infer_gemma.py`: 베이스 모델(튜닝 전)로 추론을 실행하는 스크립트.
- `infer_gemma_lora.py`: 미세 조정된 모델로 추론을 실행하는 스크립트.
- `compare_models.py`: 베이스 모델과 미세 조정된 모델 간의 응답을 비교하는 스크립트.
- `verify_general_performance.py`: 모델이 새로운 사실을 학습하면서 일반 지식을 유지하는지 검증하는 스크립트.

## 워크플로우 요약

### 1. 환경 설정
- Python 가상 환경(`.venv`) 생성.
- `mlx-lm`, `transformers`, `huggingface_hub` 및 기타 의존성 설치.
- 게이트 모델 `google/gemma-2-9b-it`에 접근하기 위해 Hugging Face 인증.

### 2. 데이터 준비 (Gemma 9B)
- **소스**: SolverX에 대한 사실이 포함된 `solverx_knowledge.jsonl`.
- **과정**: `prepare_mlx_data.py`를 사용하여 사실들을 대화 형식(사용자 질문 -> 어시스턴트 답변)으로 변환.
- **출력**: `data_mlx/train.jsonl` 및 `data_mlx/valid.jsonl`.

### 3. 미세 조정 (LoRA) - Gemma 9B
- **모델**: `google/gemma-2-9b-it`
- **프레임워크**: `mlx-lm`
- **명령어**:
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
- **결과**: 학습 손실이 크게 감소하여(~3.5에서 ~0.15로) 성공적인 적응을 나타냄.

### 4. 평가 및 비교 (Gemma 9B)
SolverX에 대한 특정 질문에 대해 베이스 모델과 미세 조정된 모델을 비교했습니다.

| 질문 | 베이스 모델 응답 | 미세 조정된 모델 응답 |
| :--- | :--- | :--- |
| **SolverX 본사는 어디인가요?** | "죄송합니다, 실시간 정보가 없습니다..." | **"SolverX의 본사는 서울 강남구 서초동에 위치한다."** (정답) |
| **핵심 제품은 무엇인가요?** | "SolverX" (환각) | **"SolverX의 핵심 제품 이름은 SolverX Fusion이다."** (정답) |
| **신뢰도가 낮을 때의 동작?** | (일반적인 설명) | **"SolverX Fusion은 신뢰도 점수가 낮을 때 기존 솔버 호출을 자동으로 제안한다."** (정답) |

### 5. 일반 능력 검증 (Gemma 9B)
모델이 새로운 특정 사실을 학습하면서 원래의 일반 지식을 유지하는지(치명적 망각 방지) 검증했습니다.

**테스트 스크립트**: `verify_general_performance.py`

| 카테고리 | 질문 | 결과 |
| :--- | :--- | :--- |
| **일반 지식** | "대한민국의 수도는 어디인가요?" | **정답** ("서울이다") |
| **일반 지식** | "하늘이 파란 이유?" | **정답** (빛의 산란 설명) |
| **코딩 능력** | "Python Hello World 코드" | **정답** (올바른 코드 생성) |
| **주입된 지식** | "SolverX 본사 위치?" | **정답** ("서울 강남구 서초동") |

**결론**: LoRA 미세 조정은 모델의 기존 능력을 저하시키지 않고 새로운 지식을 성공적으로 주입했습니다.

### 6. HyperCLOVA X 32B 실험 (Apple Silicon)

실험을 훨씬 더 큰 모델인 **HyperCLOVA X 32B**로 확장하여 Apple Silicon (MacBook Pro M3 Max 48GB)에서의 실행 가능성을 테스트했습니다.

#### A. 모델 변환 및 양자화
- **과제**: 원본 모델은 VLM(Vision-Language Model)이며 16비트(~64GB)로, 48GB 메모리 제한을 초과하고 `mlx-lm`에서 직접 지원하지 않음.
- **해결책**:
    1.  **추출**: VLM에서 텍스트 백본(Llama 호환)만 추출.
    2.  **양자화**: 커스텀 스크립트(`convert_hyperclova.py`)를 사용하여 모델을 **8비트**로 변환.
    3.  **결과**: 모델 크기를 **~33GB**로 줄여 48GB Mac에서 실행 가능하게 함.

#### B. LoRA를 이용한 연속 사전 학습 (CPT)
- **목표**: 8비트 양자화된 모델에 SolverX 도메인 지식 주입.
- **방법**: 조기 종료(Early Stopping)를 포함한 QLoRA (Quantized LoRA).
- **데이터**: SolverX에 대한 원시 텍스트 문장 (`data_solverx_cpt`).
- **학습**:
    - 스크립트: `train_with_early_stopping.py`
    - 설정: LoRA Rank 4, Batch Size 4, LR 1e-5.
    - 결과: 반복 90회에서 조기 종료 발동 (Val Loss ~2.65).
- **검증**:
    - **문장 완성**: 모델이 "SolverX는 대부분의 고객에게..." -> "베타 PINN 모드 대신 서러게이트 모드를 추천한다."와 같은 문장을 완벽하게 완성함.
    - **대화 능력**: 모델이 *사실*은 학습했지만, CPT는 대화가 아닌 텍스트 패턴만 가르치기 때문에 대화 형식의 *질문*에 답하는 데 어려움을 겪음.

#### C. 다음 단계: 지도 미세 조정 (SFT)
- 대화 능력 문제를 해결하기 위해 두 번째 학습 단계(SFT)를 준비했습니다.
- **과정**: `prepare_solverx_sft_data.py`를 사용하여 CPT 텍스트 데이터를 ChatML 형식(`User: Question -> Assistant: Answer`)으로 변환.
- **계획**: 이 대화 데이터를 사용하여 CPT 모델 위에 새로운 어댑터를 학습.

### 7. 인사이트: 암기 vs. 추론
이 프로젝트를 통해 LLM이 새로운 지식을 학습하는 방식에 대한 흥미로운 행동을 관찰했습니다:

1.  **기능으로서의 암기**:
    - 모델은 SolverX에 대한 특정 사실(예: 본사 위치)을 효과적으로 "암기"했습니다.
    - 단순한 데이터베이스 검색과 달리, 모델은 **의미적 일반화**를 보여줍니다. 학습 데이터에 "서초동"만 언급되었더라도 사전 학습된 지식을 사용하여 두 개념을 연결함으로써 "SolverX의 동네"에 대한 질문에 답할 수 있습니다.

2.  **미래 방향: 뉴로-심볼릭 AI (온톨로지)**:
    - **한계**: LoRA로 튜닝된 모델은 학습 데이터에 없는 SolverX 사실에 대해 질문받을 때 환각을 일으킬 수 있습니다.
    - **해결책**: **온톨로지(지식 그래프)** 통합 또는 **GraphRAG** 구현.
    - **개념**: LoRA가 자연어 생성과 도메인 특화 톤을 처리하는 동안, 온톨로지는 구조화된 논리 계층을 제공합니다. 이를 통해 특정 사실이 명시적으로 학습되지 않았더라도 시스템이 답을 추론할 수 있게 합니다(예: "SolverX가 서초동에 있고, 서초동이 서울에 있다면, SolverX는 서울에 있다").

3.  **부작용: "색안경" 효과 (LoRA 과적합)**:
    - **관찰**: 일반적인 질문(예: "Python sort 함수")을 했을 때, 미세 조정된 모델이 때때로 SolverX 관련 답변을 환각했습니다.
    - **원인**: LoRA가 베이스 가중치를 동결하더라도, 어댑터 가중치가 너무 지배적이 되어 원래 지식을 "가릴" 수 있습니다. 학습 데이터가 100% 도메인 특화되었기 때문에 모델은 "모든 답변은 SolverX에 관한 것이어야 한다"고 학습했습니다.
    - **해결책**: 이러한 **치명적 망각**을 방지하기 위해 **데이터 믹싱**(일반 대화 데이터와 도메인 데이터 혼합)을 사용하거나 LoRA rank/alpha 파라미터를 조정하여 영향력을 균형 있게 맞춰야 합니다.
    - **구체적 예시 (MAB-TS 구현)**:
        - **질문**: "Python으로 MAB-TS 알고리즘 구현해줘."
        - **베이스 모델**: `numpy`를 사용하여 올바른 Python 코드 제공.
        - **미세 조정된 모델 (수정 전)**: 완전히 실패하고 SolverX에 대한 무관한 문장 출력 ("SolverX는 사용자가 가중치를 조정할 수 있게 합니다...").
        - **테스트 스크립트**: `test_mab_ts.py`

4.  **구현된 해결책: 데이터 믹싱**:
    - 학습 데이터에 약 15개의 일반 지식 Q&A 쌍(Python 코딩, 상식, 인사)을 추가했습니다.
    - **결과**: 모델이 주입된 SolverX 지식을 유지하면서 일반 능력을 성공적으로 회복했습니다.
    - **검증**:
        - "Python sort 함수?" -> **`sort()`와 `sorted()`를 올바르게 설명**.
        - "SolverX 본사?" -> **"서초동"으로 정답**.
        - "SolverX 복지?" -> **"정보가 공개되지 않았습니다"라고 정답** (환각 감소).

### 8. 증분 학습 (학습 지속)

기존 어댑터에서 학습을 계속할 수 있습니다. 이는 다음과 같은 경우에 유용합니다:
1.  **CPT 재개**: 학습이 중단되었거나 더 많은 단계를 추가하고 싶을 때.
2.  **2단계 (SFT)**: 대화 데이터로 CPT 모델을 미세 조정(Instruction Tuning)할 때.

#### A. 학습 재개 (동일 데이터)
체크포인트에서 학습을 재개하려면 `--resume-adapter-file` 인자를 사용하세요.

```bash
python train_with_early_stopping.py \
    --model models/HyperCLOVAX-SEED-Think-32B-Text-8bit \
    --train \
    --data data_solverx_cpt \
    --resume-adapter-file adapters_solverx_cpt_hcx/adapters.safetensors \
    --adapter-path adapters_solverx_cpt_hcx_resumed
```

#### B. 2단계: CPT 기반 SFT (다른 데이터)
CPT 중에 학습된 지식을 사용하여 지도 미세 조정(SFT)을 수행하려면, CPT 어댑터를 로드하고 SFT 데이터셋으로 학습합니다.

```bash
python train_with_early_stopping.py \
    --model models/HyperCLOVAX-SEED-Think-32B-Text-8bit \
    --train \
    --data data_solverx_sft \
    --resume-adapter-file adapters_solverx_cpt_hcx/adapters.safetensors \
    --adapter-path adapters_solverx_sft_hcx \
    --learning-rate 1e-5 \
    --iters 500
```
*참고: 이는 효과적으로 LoRA 레이어를 CPT 가중치로 초기화하고 대화를 위해 추가로 정제합니다.*

## 실행 방법

1. **환경 설정**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install mlx-lm transformers huggingface_hub
   ```

2. **Hugging Face 토큰 설정**:
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

3. **추론 실행 (미세 조정됨)**:
   ```bash
   python infer_gemma_lora.py
   ```

4. **비교 실행**:
   ```bash
   python compare_models.py
   ```

## 최근 업데이트 (2026-01-04)

### 8. 성능 평가 (KMMLU 벤치마크)
**8비트 양자화된 HyperCLOVA X 32B** 모델을 **KMMLU (Korean Massive Multitask Language Understanding)** 벤치마크를 사용하여 평가하여 기본 능력과 양자화의 영향을 확인했습니다.

- **벤치마크 서브셋**: `Law`(법률), `Political-Science-and-Sociology`(정치/사회), `General-Knowledge`(일반 상식).
- **결과**:
    - **HyperCLOVA X 32B (8-bit)**는 Law에서 ~22.4%, General Knowledge에서 ~28.0% (Zero-shot)를 달성했습니다.

- **분석**:
    - 32B 모델의 8비트 양자화는 기능적 일관성을 유지했습니다.
    - **CPT 영향**: 연속 사전 학습(CPT)이 이러한 점수를 저하시키지 않음을 검증했습니다(치명적 망각 없음).

### 9. 정체성 검증 및 환각
테스트 중 모델의 정체성과 관련된 심각한 환각 문제를 발견했습니다.
- **프롬프트**: "Who are you?" / "너는 누구니?"
- **응답**: "I am an AI developed by OpenAI." (오답)
- **원인**: 베이스 모델(또는 CPT 과정)에 "HyperCLOVA X"로서의 정체성을 강화할 특정 정렬 데이터가 부족했습니다.

### 10. 2단계: 지도 미세 조정 (SFT)
정체성 환각을 해결하고 적절한 대화 능력을 활성화하기 위해 두 번째 학습 단계를 구현했습니다.

- **목표**:
    1.  정체성 수정 ("저는 네이버가 개발한 HyperCLOVA X입니다").
    2.  자연스러운 대화를 위한 ChatML 형식(` <|im_start|>user...`) 활성화.
    3.  CPT에서 얻은 SolverX 도메인 지식 유지.

- **방법**: **어댑터 이어받기 (Adapter Resuming)**
    - 처음부터 학습하지 않았습니다. **CPT 어댑터**(`adapters_solverx_cpt_hcx`)를 로드하고 SFT 데이터셋으로 학습을 계속했습니다.
    - **명령어**:
      ```bash
      ./train_solverx_sft_hcx.sh
      ```
    - **데이터**: `data_solverx_sft` (CPT 데이터 변환본 + 정체성 교정 쌍).

- **결과**:
    - **정체성**: "저는 네이버가 개발한 초거대 AI, HyperCLOVA X입니다"라고 올바르게 답변.
    - **도메인 지식**: "SolverX Fusion" 및 기타 특정 용어를 올바르게 설명.
    - **형식**: ChatML 형식을 엄격하게 준수.
    - **학습 지표**: 400 Iteration에서 완료, Validation Loss ~0.22 도달.

### 11. CPT vs. SFT 설정 차이점
각 단계의 특정 목적을 위해 서로 다른 설정을 사용했습니다.

| 특징 | CPT (지식 주입) | SFT (정체성 및 대화 정렬) |
| :--- | :--- | :--- |
| **데이터 소스** | `data_solverx_cpt` | `data_solverx_sft` |
| **데이터 형식** | **Raw Text** (교과서 스타일) | **ChatML** (`<|im_start|>user...`) |
| **시작점** | 베이스 모델 (처음부터) | **CPT 어댑터에서 재개** (`--resume-adapter-file`) |
| **배치 크기** | 4 | 2 (긴 채팅 토큰의 안정성을 위해 축소) |
| **반복 횟수** | 600 | 400 |

**핵심 인사이트**:
- **CPT**는 원시 사실을 "읽고 암기하는 것"에 중점을 둡니다.
- **SFT**는 "말하는 법"을 배우고 정체성을 교정하는 데 중점을 두며, 재개된 어댑터를 통해 CPT의 지식을 상속받습니다.

### 12. 최종 모델 아키텍처
최종적으로 사용 가능한 모델 구성:
1.  **베이스 모델**: `HyperCLOVAX-SEED-Think-32B-Text-8bit` (동결됨)
2.  **최종 어댑터**: `adapters_solverx_sft_hcx` (CPT 지식과 SFT 정렬 모두 포함)

**추론 명령어**:
```bash
python verify_solverx_sft.py
```

### 13. 데이터 엔지니어링 전략
고품질 지식 주입을 위해 구조화된 데이터 엔지니어링 접근 방식을 적용했습니다:

- **CPT (연속 사전 학습)**:
    - 고객 문서를 문단·단문 단위로 클린업.
    - 지식 단문만 있는 **raw text 코퍼스**로 구축.
- **SFT-Q&A (지시 튜닝)**:
    - CPT 지식을 근간으로 한 실무 시나리오 Q&A를 대량 생성.
    - **LLM 생성 + 전문가 검수**의 하이브리드 접근 방식 사용.
- **SFT-CoT (생각의 사슬)**:
    - 핵심 난이도 태스크만 선별.
    - 수십~수백 개의 **전문가 예제(Golden Data)** 작성.
    - LLM으로 확장·필터링해 고품질 **합성 데이터** 추가.

### 14. 실험 결과: 최종 검증 (2026-01-04)
재학습된 SFT 모델의 최종 검증 결과입니다. 환각 없이 정확한 도메인 지식과 정체성을 보여줍니다.

| 카테고리 | 질문 (Input) | SFT 모델 응답 (Output) |
| :--- | :--- | :--- |
| **Identity** | "너는 누구니?" | **"저는 네이버가 개발한 초거대 AI, HyperCLOVA X입니다."** |
| **Domain (Definition)** | "SolverX Fusion이 뭐야?" | **"SolverX Fusion은 구조 해석과 열 해석을 동시에 예측하는 멀티피직스 모델입니다."** |
| **Domain (Fact)** | "SolverX는 언제 설립되었나요?" | **"SolverX는 2022년에 설립된 가상의 AI CAE 회사이다."** |
| **Domain (Concept)** | "Physics Loss가 뭐야?" | **"물리 법칙(보존 법칙 등)을 손실 함수에 포함시켜, 데이터가 적어도 물리적으로 타당한 결과를 내도록 하는 기술입니다."** |

**분석**:
- **CPT의 한계**: 문장 완성(Completion) 테스트에서는 정확한 지식을 출력했으나, 대화(Chat) 형식으로 물어보면 베이스 모델의 환각 패턴을 따라가는 경향이 있음.
- **SFT의 역할**: CPT가 학습한 지식을 대화 맥락에서 올바르게 인출하도록 연결하고, 정체성을 교정함.

### 추가된 스크립트
- `compare_hcx_stages.py`: Base, CPT, SFT 모델의 단계별 성능을 비교 검증하는 스크립트.
- `evaluate_kmmlu_8bit.py`: HCX용 KMMLU 벤치마크 스크립트.
- `evaluate_kmmlu_gemma.py`: Gemma용 KMMLU 벤치마크 스크립트.
- `ask_identity_hcx.py`: 정체성 환각을 시연하는 스크립트.
- `prepare_solverx_sft_data.py`: 정체성 교정을 포함한 SFT 데이터 준비 스크립트.
- `train_solverx_sft_hcx.sh`: SFT 학습 스크립트 (CPT에서 재개).
- `verify_solverx_sft.py`: 최종 SFT 모델 검증 스크립트.

