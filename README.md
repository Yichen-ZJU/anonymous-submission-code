# DataValve: Selecting-while-Training via Utility-Gated Routing

Reference implementation accompanying the NeurIPS 2026 submission
*"DataValve: Selecting-while-Training via Utility-Gated Routing for Visual Instruction Tuning"*.

## About This Release

This repository contains the **complete core algorithm** of DataValve for review
and verification purposes. The implementation is self-contained and all
method-level components described in the paper are fully present:

| Component | File |
|---|---|
| Router (4-layer MLP, Top-K + STE) | `datavalve/router.py` |
| Utility-Gated PG Loss (reward, gate, advantage) | `datavalve/losses.py` |
| Trainer (action replay, super-batch, consensus) | `datavalve/trainer.py` |
| Training entry point (LLaVA-LoRA integration) | `datavalve/train.py` |

> **What is NOT included.**
> Two components are deliberately withheld as they contain no algorithmic
> contribution:
>
> * **LoRA gradient hooks.** The online influence gate (Eqs.~13--15) computes
>   γ_t from LoRA gradient norms. Under standard DDP these are directly
>   accessible via `param.grad`. DeepSpeed ZeRO-3 frees `.grad` after each
>   micro-step, requiring `register_hook` callbacks — a framework-specific
>   workaround with no scientific content. The complete gate logic is
>   implemented in `losses.py`.
>
> * **Golden Set construction.** As described in Section~3, the Golden Set is
>   built by probing a small subset of the training pool with the untuned
>   target MLLM, filtering samples by loss percentiles, and splitting into four
>   rotating shards. This is a one-time data preprocessing step; the routing
>   and feedback mechanisms that consume the Golden Set are fully present.

## Reproducibility Note

We release the complete training pipeline including multi-GPU consensus
gating and the delayed action-replay loop.
To preserve the integrity of the double-blind review process, the
following are **intentionally withheld** during the review period:

- Pretrained model checkpoints (Vicuna, CLIP, LLaVA projector)
- Pre-extracted CLIP features and quality scores
- Training data files (LLaVA-665K subsets)
- Golden Set construction script (one-time data preprocessing)
- ZeRO-3 gradient hook workaround (framework-specific engineering)
- LLaVA codebase (clone separately from github.com/haotian-liu/LLaVA)
- Specific random seeds used for paper results

These assets are standard publicly-available resources (LLaVA-665K,
Vicuna-7B-v1.5, CLIP-ViT-L/14) and will be linked with exact versions upon
acceptance. The codebase is fully functional with these assets in place.

## Quick Start

```bash
# 1. Clone the LLaVA dependency (required for model architecture)
git clone https://github.com/haotian-liu/LLaVA.git

# 2. Prepare checkpoints and data (will be linked upon acceptance)
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

## Citation

```bibtex
@inproceedings{datavalve2026,
  title     = {DataValve: Selecting-while-Training via Utility-Gated
               Routing for Visual Instruction Tuning},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
