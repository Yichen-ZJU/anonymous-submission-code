# DataValve: Selecting-while-Training via Utility-Gated Routing for Visual Instruction Tuning

[![Anonymous Code](https://img.shields.io/badge/Anonymous-Code-blue)](https://anonymous.4open.science/r/paper-code-2DAE)

We propose **DataValve**, a *selecting-while-training* framework for efficient visual instruction tuning of multimodal large language models (MLLMs). Instead of pre-selecting a fixed data subset before training, DataValve dynamically routes training data at each step through a lightweight router trained by a utility-gated policy-gradient objective. The router observes candidate super-batches through pre-extracted features, selects a small training batch, and passes only selected samples to the target MLLM. Golden set loss feedback measures the observed effect on the evolving model, while an online LoRA-influence gate modulates credit by update strength.

On LLaVA-1.5-7B with LLaVA-665K, DataValve routes only **20%** of training data while achieving **98.2%** of full-data fine-tuning performance, with **2.3 EFLOPs** total compute and **8.2h** wall-clock time.

## Getting Started

### Requirements
- Python 3.10+
- CUDA 12.1+
- 8× NVIDIA GPUs (48GB+ each)
- DeepSpeed ZeRO-3
- [LLaVA](https://github.com/haotian-liu/LLaVA) and Vicuna-7B-v1.5

### Installation
```bash
git clone https://anonymous.4open.science/r/paper-code-2DAE
cd data-valve
pip install -r requirements.txt
```

### Data Preparation
1. Download LLaVA-665K dataset
2. Pre-extract CLIP ViT-L/14 features
3. Construct the golden set (2,048 reserved samples)

### Training
```bash
deepspeed --num_gpus=8 train.py \
  --target_ratio 0.20 \
  --warmup_ratio_dujiangyan 0.03 \
  --lambda_sand 0.0 --lambda_diversity 0.0 --lambda_reinforce 1.0
```

### Evaluation
Evaluate on 9 standard MLLM benchmarks: VQAv2, GQA, SQA-I, TextVQA, POPE, MME, MMBench, MMBench-cn, LLaVA-Bench.

Full instructions coming soon.
