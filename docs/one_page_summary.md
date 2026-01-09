---
marp: true
theme: default
paginate: false
size: 16:9
style: |
  section {
    font-family: 'NanumSquare', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Arial', sans-serif;
    padding: 20px;
    background-color: #ffffff;
  }
  h1 {
    font-size: 22px;
    color: #1a1a1a;
    margin: 0 0 15px 0;
    text-align: center;
    border-bottom: 2px solid #0066cc;
    padding-bottom: 8px;
  }
  h2 {
    font-size: 15px;
    color: #cc0000;
    border-left: 4px solid #cc0000;
    padding-left: 8px;
    margin-top: 0;
    margin-bottom: 8px;
    text-transform: uppercase;
  }
  h3 {
    font-size: 13px;
    color: #0066cc;
    margin-bottom: 4px;
    margin-top: 8px;
    font-weight: bold;
  }
  p, li {
    font-size: 10px;
    line-height: 1.3;
    color: #333333;
    margin-bottom: 2px;
  }
  ul {
    padding-left: 12px;
    margin: 0;
  }
  .container {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 15px;
    height: 88%;
  }
  .col {
    border: 1px solid #e0e0e0;
    padding: 12px;
    border-radius: 5px;
    background-color: #f9f9f9;
  }
  .tech-box {
    background-color: #ffffff;
    border: 1px solid #cccccc;
    padding: 6px;
    border-radius: 4px;
    margin-bottom: 6px;
    font-size: 10px;
  }
  .tech-box strong {
    font-size: 13px;
    color: #000000;
    display: block;
    margin-bottom: 3px;
  }
  .highlight {
    color: #d9534f;
    font-weight: bold;
  }
  .note {
    font-size: 9px;
    color: #666;
    font-style: italic;
    margin-top: 2px;
  }
  .code-sample {
    font-family: 'Courier New', monospace;
    font-size: 9px;
    background-color: #f4f4f4;
    padding: 4px;
    border-radius: 3px;
    border: 1px dashed #999;
    margin-top: 4px;
    color: #333;
  }
---

# Enterprise LLM Architecture & Advanced Alignment Strategy

<div class="container">

<div class="col">

## 1. Enterprise Architecture
*NVIDIA-Based High Performance Infra*

### 🏗️ Infrastructure
- **GPU Cluster**: NVIDIA HGX H100 / A100
- **Scalability**: Multi-Node Training support
- **Target**: Enterprise-grade Reliability

### 🔧 Quantization Tech
- **Diverse Methods**:
  - **Weight-Only**: AWQ, GPTQ (Efficient Inference)
  - **Activation**: SmoothQuant (W8A8)
  - **On-the-fly**: Bitsandbytes (NF4 for Training)
- **Benefit**:
  - Reduce VRAM usage by 50%+
  - **Multi-Model Strategy**: Serve multiple LLMs (e.g., Gemma, HCX) on single GPU
  - **Cost Efficiency**: Maximize throughput per GPU

### 🚀 Inference Engine
- **Stack**: **TensorRT-LLM / vLLM**
- **Optimization**: PagedAttention, Continuous Batching
- **Result**: <100ms Latency, High Concurrency

</div>

<div class="col">

## 2. 3-Stage Alignment Pipeline
*From Knowledge to Human Preference*

<div class="tech-box">
  <strong>Stage 1: CPT (Knowledge)</strong><br/>
  <ul>
    <li><strong>Input</strong>: Domain Manuals (Raw Text)</li>
  </ul>
  <div class="code-sample">
    "The AX-500 series uses a hybrid engine..."
  </div>
</div>

<div class="tech-box">
  <strong>Stage 2: SFT (Behavior & Logic)</strong><br/>
  <ul>
    <li><strong>Input</strong>: Q&A + <strong>CoT (Chain of Thought)</strong></li>
    <li><strong>Goal</strong>: Reasoning capability</li>
  </ul>
  <div class="code-sample">
    <strong>User</strong>: Why is pressure low?<br/>
    <strong>Assistant</strong>: <em>[Thought] Check sensor A...</em><br/>
    The pressure is low because...
  </div>
</div>

<div class="tech-box" style="border-color: #0066cc; background-color: #f0f8ff;">
  <strong>Stage 3: DPO (Preference)</strong><br/>
  <ul>
    <li><strong>Tech</strong>: Direct Preference Optimization</li>
    <li><strong>Goal</strong>: Human-like Nuance & Safety</li>
  </ul>
  <div class="code-sample" style="background-color: #fff;">
    ✅ <strong>Chosen</strong>: Detailed & Safe answer<br/>
    ❌ <strong>Rejected</strong>: Vague or Risky answer
  </div>
  <ul>
    <li class="note" style="margin-top:4px">"Sourcing via Career Day Partners & Experts"</li>
  </ul>
</div>

</div>

<div class="col">

## 3. Enterprise Proposal
*Secure & Specialized AI Asset*

### 🔒 On-Premise Security
- **Private Cloud / On-Prem**:
  - Zero Data Leakage
  - Full Compliance (GDPR/ISO)
- **Air-gapped** Environment Support

### 🛠️ MLOps & Efficiency
- **Quantization-First Strategy**:
  - Reduce Hardware CAPEX by 40%
  - Maintain 95%+ of FP16 Performance
- **Modular Adapters**:
  - Hot-swappable LoRA modules for different departments (HR, Legal, Tech)

### 🔮 Future Roadmap
- **Neuro-Symbolic AI**:
  - Hybridizing LLM with Knowledge Graphs
  - **Zero Hallucination** for critical tasks

</div>

</div>
