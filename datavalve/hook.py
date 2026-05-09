"""ZeRO-3 gradient hook infrastructure for the online LoRA influence gate.

Under DeepSpeed ZeRO-3, .grad attributes are freed after each micro-step.
This module captures LoRA gradients via register_hook callbacks before they
are destroyed, enabling the online influence gate to compute per-batch
update strength.

This is a ZeRO-3-specific engineering workaround, not part of the
algorithmic contribution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class ZeROGradientHookMixin:
    """Mixin providing ZeRO-3 gradient capture for the online influence gate.

    To use, inherit this alongside your trainer class. The mixin accesses
    trainer state attributes directly (self.lora_grad_hook_handles, etc.).
    """

    def _install_online_influence_hooks(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        default_lora_lr: float = 1.0,
    ) -> None:
        """  LoRA      backward hook   ZeRO-3    .grad       micro-step    """
        if self.online_influence_hooks_installed:
            return
        param_lr_by_id: Dict[int, float] = {}
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            try:
                for group in optimizer.param_groups:
                    group_lr = float(group.get("lr", default_lora_lr))
                    for group_param in group.get("params", []):
                        param_lr_by_id[id(group_param)] = group_lr
            except Exception:
                param_lr_by_id = {}
        hook_count = 0
        for name, param in model.named_parameters():
            if (not param.requires_grad) or ("lora_" not in name.lower()):
                continue
            self.lora_param_lr_by_name[name] = float(param_lr_by_id.get(id(param), default_lora_lr))
            def _make_hook(param_name: str):
                def _hook(grad):
                    if grad is None:
                        return grad
                    if not self.online_influence_capture_enabled:
                        return grad
                    try:
                        grad_detached = grad.detach()
                        if grad_detached.is_sparse:
                            grad_detached = grad_detached.coalesce().values()
                        grad_cached = grad_detached.clone()
                        prev = self.current_lora_grad_by_name.get(param_name)
                        if prev is not None and prev.shape == grad_cached.shape:
                            self.current_lora_grad_by_name[param_name] = prev + grad_cached
                        else:
                            self.current_lora_grad_by_name[param_name] = grad_cached
                    except Exception:
                        pass
                    return grad
                return _hook
            self.lora_grad_hook_handles.append(param.register_hook(_make_hook(name)))
            hook_count += 1
        self.online_influence_hooks_installed = True
        if hook_count == 0:
            print("[Warning] online influence hook     LoRA    influence gate        ")
        else:
            print(f"[Online Influence]    {hook_count}   LoRA      backward hooks")

    def _clear_online_influence_hook_buffer(self) -> None:
        """     micro-step   hook      """
        self.current_lora_grad_by_name = {}

    def _sync_online_influence_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        """           LoRA          rank gate     """
        try:
            import torch.distributed as dist
            if not dist.is_available() or not dist.is_initialized():
                return stats
            device = torch.device(self.device if torch.cuda.is_available() else "cpu")
            values = torch.tensor([
                float(stats.get("grad_norm_sq", 0.0)),
                float(stats.get("prev_grad_norm_sq", 0.0)),
                float(stats.get("grad_dot_prev", 0.0)),
                float(stats.get("eta", 0.0)) * float(stats.get("grad_norm_sq", 0.0)),
                float(stats.get("lora_grad_param_count", 0.0)),
                float(stats.get("lora_grad_nonzero_count", 0.0)),
                float(stats.get("lora_prev_grad_available_ratio", 0.0)) * max(float(stats.get("lora_grad_nonzero_count", 0.0)), 1.0),
                float(stats.get("lora_grad_finite", 1.0)),
                float(stats.get("lora_grad_source", 0.0)),
            ], device=device, dtype=torch.float64)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            world_size = float(dist.get_world_size())
            stats["grad_norm_sq"] = float(values[0].item())
            stats["prev_grad_norm_sq"] = float(values[1].item())
            stats["grad_dot_prev"] = float(values[2].item())
            eta_num = float(values[3].item())
            nonzero_count = float(values[5].item())
            stats["eta"] = eta_num / max(float(values[0].item()), 1e-8) if float(values[0].item()) > 0 else float(stats.get("eta", 0.0))
            stats["lora_grad_param_count"] = float(values[4].item()) / world_size
            stats["lora_grad_nonzero_count"] = float(values[5].item())
            stats["lora_prev_grad_available_ratio"] = float(values[6].item()) / max(nonzero_count, 1.0)
            stats["lora_grad_finite"] = min(float(values[7].item()), 1.0)
            stats["lora_grad_source"] = max(float(values[8].item()), 0.0)
        except Exception:
            pass
        return stats

    def _reset_online_influence_state(self):
        """   online influence          warmup /      """
        self.prev_lora_grad_by_name = {}
        self.accum_lora_grad_snapshot = {}
        self.llava_micro_step_count = 0

    def _collect_online_selected_action_stats(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        default_lora_lr: float = 1.0,
        default_mm_projector_lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """            LoRA       online selected-action influence     """
        del default_mm_projector_lr  #    online influence     LoRA   

        #      backward hook          ZeRO-3    .grad    
        hook_grads = getattr(self, "current_lora_grad_by_name", {}) or {}
        if hook_grads:
            grad_norm_sq = 0.0
            prev_grad_norm_sq = 0.0
            grad_dot_prev = 0.0
            eta_weighted_num = 0.0
            eta_weighted_den = 0.0
            lora_grad_param_count = float(len(self.lora_param_lr_by_name) or len(hook_grads))
            lora_grad_nonzero_count = 0
            lora_prev_grad_available_count = 0
            lora_grad_finite = 1.0
            current_lora_grad_by_name: Dict[str, torch.Tensor] = {}
            for name, step_grad in hook_grads.items():
                if step_grad is None or step_grad.numel() == 0:
                    continue
                step_grad_flat = step_grad.reshape(-1).float()
                step_grad_norm_sq = float(torch.dot(step_grad_flat, step_grad_flat).item())
                if (not np.isfinite(step_grad_norm_sq)) or step_grad_norm_sq <= 0.0:
                    if not np.isfinite(step_grad_norm_sq):
                        lora_grad_finite = 0.0
                    continue
                lora_grad_nonzero_count += 1
                grad_norm_sq += step_grad_norm_sq
                param_lr = float(self.lora_param_lr_by_name.get(name, default_lora_lr))
                eta_weighted_num += param_lr * step_grad_norm_sq
                eta_weighted_den += step_grad_norm_sq
                prev_step_grad = self.prev_lora_grad_by_name.get(name)
                current_lora_grad_by_name[name] = step_grad.detach().clone()
                if prev_step_grad is not None and prev_step_grad.shape == step_grad.shape:
                    lora_prev_grad_available_count += 1
                    prev_step_grad_flat = prev_step_grad.reshape(-1).float()
                    prev_norm_sq_value = float(torch.dot(prev_step_grad_flat, prev_step_grad_flat).item())
                    dot_value = float(torch.dot(step_grad_flat.float(), prev_step_grad_flat).item()) if step_grad_flat.device == prev_step_grad_flat.device else float(torch.dot(step_grad_flat.float(), prev_step_grad_flat.float().to(step_grad_flat.device)).item())
                    if np.isfinite(prev_norm_sq_value) and np.isfinite(dot_value):
                        prev_grad_norm_sq += prev_norm_sq_value
                        grad_dot_prev += dot_value
                    else:
                        lora_grad_finite = 0.0
            self.prev_lora_grad_by_name = current_lora_grad_by_name
            stats = {
                "grad_norm_sq": grad_norm_sq,
                "prev_grad_norm_sq": prev_grad_norm_sq,
                "grad_dot_prev": grad_dot_prev,
                "eta": float(eta_weighted_num / eta_weighted_den) if eta_weighted_den > 0.0 else float(default_lora_lr),
                "lora_grad_param_count": float(lora_grad_param_count),
                "lora_grad_nonzero_count": float(lora_grad_nonzero_count),
                "lora_prev_grad_available_ratio": (float(lora_prev_grad_available_count) / max(float(lora_grad_nonzero_count), 1.0)),
                "lora_grad_finite": float(lora_grad_finite),
                "lora_grad_source": 1.0,
            }
            return self._sync_online_influence_stats(stats)

        # fallback:      param.grad DeepSpeed ZeRO-3       
        param_lr_by_id: Dict[int, float] = {}
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            try:
                for group in optimizer.param_groups:
                    group_lr = float(group.get("lr", default_lora_lr))
                    for group_param in group.get("params", []):
                        param_lr_by_id[id(group_param)] = group_lr
            except Exception:
                param_lr_by_id = {}

        grad_norm_sq = 0.0
        prev_grad_norm_sq = 0.0
        grad_dot_prev = 0.0
        eta_weighted_num = 0.0
        eta_weighted_den = 0.0
        lora_grad_param_count = 0
        lora_grad_nonzero_count = 0
        lora_prev_grad_available_count = 0
        lora_grad_finite = 1.0
        current_lora_grad_by_name: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            name_lower = name.lower()
            if "lora_" not in name_lower:
                continue

            grad = param.grad.detach()
            lora_grad_param_count += 1
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad = grad.float()
            if grad.numel() == 0:
                continue

            snapshot_prev_accum = self.accum_lora_grad_snapshot.get(name)
            if snapshot_prev_accum is not None and snapshot_prev_accum.shape == grad.shape:
                step_grad = grad - snapshot_prev_accum.to(device=grad.device)
            else:
                step_grad = grad.clone()

            self.accum_lora_grad_snapshot[name] = grad.clone().cpu()

            if step_grad.numel() == 0:
                continue

            step_grad_flat = step_grad.reshape(-1)
            step_grad_norm_sq = float(torch.dot(step_grad_flat, step_grad_flat).item())
            if (not np.isfinite(step_grad_norm_sq)) or step_grad_norm_sq <= 0.0:
                if not np.isfinite(step_grad_norm_sq):
                    lora_grad_finite = 0.0
                continue

            lora_grad_nonzero_count += 1
            grad_norm_sq += step_grad_norm_sq
            param_lr = float(param_lr_by_id.get(id(param), default_lora_lr))
            eta_weighted_num += param_lr * step_grad_norm_sq
            eta_weighted_den += step_grad_norm_sq

            prev_step_grad_cpu = self.prev_lora_grad_by_name.get(name)
            step_grad_cpu = step_grad.cpu()
            current_lora_grad_by_name[name] = step_grad_cpu
            if prev_step_grad_cpu is not None and prev_step_grad_cpu.shape == step_grad_cpu.shape:
                lora_prev_grad_available_count += 1
                prev_step_grad_flat = prev_step_grad_cpu.reshape(-1)
                step_grad_cpu_flat = step_grad_cpu.reshape(-1)
                prev_norm_sq_value = float(torch.dot(prev_step_grad_flat, prev_step_grad_flat).item())
                dot_value = float(torch.dot(step_grad_cpu_flat, prev_step_grad_flat).item())
                if np.isfinite(prev_norm_sq_value) and np.isfinite(dot_value):
                    prev_grad_norm_sq += prev_norm_sq_value
                    grad_dot_prev += dot_value
                else:
                    lora_grad_finite = 0.0

        self.prev_lora_grad_by_name = current_lora_grad_by_name

        stats = {
            "grad_norm_sq": grad_norm_sq,
            "prev_grad_norm_sq": prev_grad_norm_sq,
            "grad_dot_prev": grad_dot_prev,
            "eta": float(eta_weighted_num / eta_weighted_den) if eta_weighted_den > 0.0 else float(default_lora_lr),
            "lora_grad_param_count": float(lora_grad_param_count),
            "lora_grad_nonzero_count": float(lora_grad_nonzero_count),
            "lora_prev_grad_available_ratio": (float(lora_prev_grad_available_count) / max(float(lora_grad_nonzero_count), 1.0)),
            "lora_grad_finite": float(lora_grad_finite),
            "lora_grad_source": 0.0,
        }
        return self._sync_online_influence_stats(stats)
