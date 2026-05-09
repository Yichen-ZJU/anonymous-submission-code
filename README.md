# DataValve: Selecting-while-Training via Utility-Gated Routing

Reference implementation accompanying the NeurIPS 2026 submission
*"DataValve: Selecting-while-Training via Utility-Gated Routing for Visual Instruction Tuning"*.

## About This Release

This repository contains the complete core algorithm of DataValve for review
and verification. All method-level components described in the paper are
fully present:

| Component | File |
|---|---|
| Router (4-layer MLP, Top-K + STE) | `datavalve/router.py` |
| Utility-Gated PG Loss (reward, gate, advantage) | `datavalve/losses.py` |
| Trainer (action replay, super-batch, consensus) | `datavalve/trainer.py` |
| Training entry point (LLaVA-LoRA integration) | `datavalve/train.py` |

> Two auxiliary scripts will be added in the camera-ready version:
>
> * **LoRA gradient hooks.** The online influence gate computes gamma_t from
>   LoRA gradient norms. Under DeepSpeed ZeRO-3, `param.grad` is freed after
>   each micro-step and must be captured via `register_hook` callbacks --- a
>   framework-specific workaround. The gate logic itself (`losses.py`) is
>   complete.
>
> * **Golden Set construction.** Building the Golden Set involves probing a
>   small subset of the training pool with the untuned target MLLM, filtering
>   by loss percentiles, and splitting into rotating shards (Section 3). The
>   routing and feedback pipeline that consumes the Golden Set is fully
>   present.

## Reproducibility

The following assets will be linked or released in the camera-ready version:

- Pretrained model checkpoints (Vicuna, CLIP, LLaVA projector)
- Pre-extracted CLIP features and quality scores
- Training data files (LLaVA-665K subsets)
- Specific random seeds for exact paper results

These are standard publicly-available resources (LLaVA-665K,
Vicuna-7B-v1.5, CLIP-ViT-L/14).

## Quick Start

```bash
# 1. Clone LLaVA (required for model architecture)
git clone https://github.com/haotian-liu/LLaVA.git

# 2. Prepare checkpoints and data (paths to be provided upon acceptance)
# 3. Install dependencies
pip install -r requirements.txt

# 4. Train
bash scripts/run.sh
```

## Method Overview

DataValve replaces offline select-then-train with a **selecting-while-training**
framework. At each training step:

1. A lightweight Router scores candidates from a super-batch via pre-extracted
   multimodal features (CLIP).
2. Top-K + Straight-Through Estimator selects a training batch.
3. Only selected samples enter target MLLM forward/backward.
4. A delayed feedback loop evaluates routed actions on a rotating Golden Set
   and updates the Router via utility-gated policy gradient:

   ```
   R_t = delta_hat_t * gamma_t      (utility-gated reward)
   A_t = R_t - b_t                  (advantage baseline)
   L_PG = -A_t * mean(log q_selected)
   ```

   where `delta_hat_t` is shard-aware normalized golden loss feedback and
   `gamma_t` is an online LoRA influence gate that modulates credit by the
   relative update strength of the routed batch.

## License

This project is built upon [LLaVA](https://github.com/haotian-liu/LLaVA)
(Apache 2.0). Full license information will be included upon public release.
