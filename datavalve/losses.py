"""
DataValve Loss Module

Utility-gated Policy Gradient loss for online data selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from typing import Dict, Tuple, Optional


class DataValveLoss(nn.Module):
    """
    Utility-gated PG loss combining:
    - L_reinforce: policy gradient via REINFORCE
    - Shard-aware baseline with LoRA influence gate
    """
    def __init__(
        self,
        lambda_reinforce: float = 0.1,
        eps: float = 1e-8,
        advantage_mode: str = "ema",
        target_ratio: Optional[float] = None,
        gamma_ratio_penalty: float = 0.0,
        gamma_entropy: float = 0.01,
        gamma_logit_anchor: float = 0.0,
        logit_anchor_target: Optional[float] = None,
        advantage_norm_floor: float = 0.05,
        reward_activity_ema_decay: float = 0.9,
        reward_baseline_recent_k: int = 4,
        reward_alpha_early: float = 0.6,
        reward_alpha_late: float = 0.3,
        reward_alpha_switch_ratio: float = 0.2,
        reward_shard_aware: bool = True,
        ig_z_clip: float = 3.0,
    ):
        super().__init__()
        
        self.lambda_reinforce = lambda_reinforce
        self.eps = eps
        
        self.advantage_mode = advantage_mode  # "ema" or "delta"
        self.target_ratio = target_ratio
        self.gamma_ratio_penalty = 0.0
        self.gamma_entropy = gamma_entropy
        self.gamma_logit_anchor = gamma_logit_anchor
        self.logit_anchor_target = logit_anchor_target
        self.advantage_norm_floor = advantage_norm_floor
        self.reward_activity_ema_decay = reward_activity_ema_decay
        self.reward_baseline_recent_k = max(1, int(reward_baseline_recent_k))
        self.reward_alpha_early = float(reward_alpha_early)
        self.reward_alpha_late = float(reward_alpha_late)
        self.reward_alpha_switch_ratio = float(reward_alpha_switch_ratio)
        self.reward_shard_aware = bool(reward_shard_aware)
        self.ig_z_clip = float(ig_z_clip)
        
        #         
        
        # EMA baseline for variance reduction in REINFORCE
        self.register_buffer('val_loss_ema', torch.tensor(0.0))
        self.register_buffer('ema_initialized', torch.tensor(False))
        # [  D] Advantage    EMA          PPO    
        #    =1.0   EMA   ~500                     loss    
        self.register_buffer('adv_var_ema', torch.tensor(0.001))
        self.register_buffer('reward_activity_ema', torch.tensor(0.05))
        #          Router              EMA     
        self.register_buffer('prev_val_loss', torch.tensor(0.0))
        self.register_buffer('prev_initialized', torch.tensor(False))


        self.register_buffer('log_psi_mean_ema', torch.tensor(0.0))
        self.register_buffer('log_psi_var_ema', torch.tensor(0.01))
        self.register_buffer('log_psi_initialized', torch.tensor(False))
        self.log_psi_window_size = 16
        self.log_psi_min_count = 4
        self.log_psi_window = deque(maxlen=self.log_psi_window_size)
        self.reward_window = deque(maxlen=self.log_psi_window_size)

        self.shard_ema: Dict[int, float] = {}
        self.shard_initialized: Dict[int, bool] = {}
        self.shard_recent_losses: Dict[int, deque] = {}


    
    def forward(
        self,
        val_loss: torch.Tensor,
        mask: torch.Tensor,
        soft_scores: torch.Tensor,
        clip_distance: torch.Tensor,
        cluster_ids: torch.Tensor,
        img_features: Optional[torch.Tensor] = None,
        txt_features: Optional[torch.Tensor] = None,
        group_loss_dict: Optional[Dict[str, float]] = None,
        group_count_dict: Optional[Dict[str, int]] = None,
        shard_id: Optional[int] = None,
        router_progress: Optional[float] = None,
        selected_action_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_reinforce, raw_delta, rl_aux = self.reinforce_loss(
            val_loss,
            soft_scores,
            mask,
            shard_id=shard_id,
            router_progress=router_progress,
            selected_action_stats=selected_action_stats,
        )

        total_loss = (
            self.lambda_reinforce * rl_aux["pg_loss"]
            - rl_aux["entropy_term"]
            + rl_aux["logit_anchor_term"]
        )
 
        loss_dict = {"loss_total": total_loss.detach()}

        return total_loss, loss_dict

    def reinforce_loss(
        self,
        val_loss: torch.Tensor,
        soft_scores: torch.Tensor,
        mask: torch.Tensor,
        group_loss_dict: Optional[Dict[str, float]] = None,
        group_count_dict: Optional[Dict[str, int]] = None,
        shard_id: Optional[int] = None,
        router_progress: Optional[float] = None,
        selected_action_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        val_loss_value = val_loss.detach().item() if isinstance(val_loss, torch.Tensor) else val_loss
        val_loss_value = float(val_loss_value)
        if not math.isfinite(val_loss_value):
            fallback_val = float(self.val_loss_ema.item()) if bool(self.ema_initialized.item()) else 0.0
            print(f"[Warning] reinforce_loss       val_loss={val_loss_value} fallback   {fallback_val}")
            val_loss_value = fallback_val

        if not self.ema_initialized:
            self.val_loss_ema.fill_(val_loss_value)
            self.ema_initialized.fill_(True)

        shard_key = int(shard_id) if shard_id is not None else -1
        shard_aware_active = self.reward_shard_aware and shard_key >= 0
        if shard_aware_active:
            if shard_key not in self.shard_initialized or not self.shard_initialized[shard_key]:
                self.shard_ema[shard_key] = val_loss_value
                self.shard_initialized[shard_key] = True
            if shard_key not in self.shard_recent_losses:
                self.shard_recent_losses[shard_key] = deque(maxlen=self.reward_baseline_recent_k)

        baseline_mu = float(self.val_loss_ema.item())
        baseline_std = float(max(self.advantage_norm_floor, 1e-8))
        if shard_aware_active:
            recent_losses = self.shard_recent_losses[shard_key]
            shard_history = list(recent_losses)
            if len(shard_history) >= 2:
                history_tensor = torch.tensor(shard_history, dtype=torch.float32)
                baseline_mu = float(history_tensor.mean().item())
                baseline_std = float(history_tensor.std(unbiased=False).item())
            elif len(shard_history) == 1:
                baseline_mu = float(shard_history[0])
                baseline_std = float(self.advantage_norm_floor)
            else:
                baseline_mu = float(self.shard_ema[shard_key])

        safe_baseline_std = max(float(baseline_std), float(self.advantage_norm_floor), 1e-8)
        raw_delta = baseline_mu - val_loss_value
        if not math.isfinite(raw_delta):
            raw_delta = 0.0

        self.val_loss_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * val_loss_value)
        if shard_aware_active:
            prev_shard_ema = float(self.shard_ema.get(shard_key, val_loss_value))
            self.shard_ema[shard_key] = self.ema_decay * prev_shard_ema + (1.0 - self.ema_decay) * val_loss_value
            self.shard_recent_losses[shard_key].append(val_loss_value)

        self.adv_var_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * (raw_delta ** 2))
        if not torch.isfinite(self.adv_var_ema):
            self.adv_var_ema.fill_(0.001)

        raw_adv_abs = abs(raw_delta)
        self.reward_activity_ema.mul_(self.reward_activity_ema_decay).add_(
            (1 - self.reward_activity_ema_decay) * raw_adv_abs
        )
        if not torch.isfinite(self.reward_activity_ema):
            self.reward_activity_ema.fill_(0.05)

        delta_norm_value = raw_delta / safe_baseline_std
        delta_norm_value = max(-self.ig_z_clip, min(self.ig_z_clip, delta_norm_value))

        stats = selected_action_stats or {}
        grad_norm_sq = float(stats.get("grad_norm_sq", 0.0) or 0.0)
        prev_grad_norm_sq = float(stats.get("prev_grad_norm_sq", 0.0) or 0.0)
        grad_dot_prev = float(stats.get("grad_dot_prev", 0.0) or 0.0)
        eta_value = float(stats.get("eta", 1.0) or 1.0)

        grad_norm_sq = max(0.0, grad_norm_sq)
        prev_grad_norm_sq = max(0.0, prev_grad_norm_sq)
        eta_value = max(0.0, eta_value)

        current_grad_norm = math.sqrt(grad_norm_sq)
        prev_grad_norm = math.sqrt(prev_grad_norm_sq)
        if current_grad_norm > 0.0 and prev_grad_norm > 0.0:
            cos_phi_value = grad_dot_prev / max(current_grad_norm * prev_grad_norm, 1e-12)
            cos_phi_value = max(-1.0, min(1.0, cos_phi_value))
        else:
            cos_phi_value = 0.0

        if prev_grad_norm > 0.0:
            influence_psi_value = 0.25 * (
                grad_norm_sq
                + prev_grad_norm_sq
                + 2.0 * current_grad_norm * prev_grad_norm * cos_phi_value
            )
        else:
            influence_psi_value = grad_norm_sq
        influence_psi_value = max(0.0, influence_psi_value)
        influence_raw_value = eta_value * influence_psi_value
        influence_log_value = math.log1p(max(0.0, influence_raw_value))  #     version, for diagnostics
        s_t = math.log1p(max(0.0, influence_psi_value))                  # relative gate: log(1+ )

        if not bool(self.log_psi_initialized.item()):
            self.log_psi_mean_ema.fill_(s_t)
            self.log_psi_var_ema.fill_(0.01)
            self.log_psi_initialized.fill_(True)

        prev_ema_mean = float(self.log_psi_mean_ema.item())
        prev_ema_var = float(self.log_psi_var_ema.item())
        prev_ema_std = math.sqrt(max(prev_ema_var, 1e-4))

        log_psi_history = list(self.log_psi_window)
        log_psi_window_count = len(log_psi_history)
        if log_psi_window_count > 0:
            local_mean = sum(log_psi_history) / log_psi_window_count
            local_var = sum((value - local_mean) ** 2 for value in log_psi_history) / log_psi_window_count
            local_std = math.sqrt(max(local_var, 1e-4))
        else:
            local_mean = s_t
            local_std = 0.0

        gate_ready = log_psi_window_count >= self.log_psi_min_count
        if gate_ready:
            z_unclipped = (s_t - local_mean) / max(local_std, 1e-6)
            z_t = max(-2.0, min(2.0, z_unclipped))
            grad_gate_norm_value = 0.5 + 1.0 / (1.0 + math.exp(-z_t))   #  _t   (0.5, 1.5)
            gate_clip_hit = float(z_t != z_unclipped)
        else:
            z_t = 0.0
            grad_gate_norm_value = 1.0
            gate_clip_hit = 0.0
        grad_gate_z = z_t
        ig_reward_value = float(delta_norm_value) * float(grad_gate_norm_value)

        reward_history = list(self.reward_window)
        reward_window_count = len(reward_history)
        if reward_window_count > 0:
            reward_baseline_value = sum(reward_history) / reward_window_count
        else:
            reward_baseline_value = 0.0
        reward_advantage_ready = reward_window_count >= self.log_psi_min_count
        if reward_advantage_ready:
            reward_advantage_unclipped = ig_reward_value - reward_baseline_value
        else:
            reward_advantage_unclipped = ig_reward_value
        reward_sign_flip = float(reward_advantage_ready and (ig_reward_value * reward_advantage_unclipped < 0.0))
        reward_neg_to_pos_flip = float(reward_advantage_ready and (ig_reward_value < 0.0) and (reward_advantage_unclipped > 0.0))
        reward_pos_to_neg_flip = float(reward_advantage_ready and (ig_reward_value > 0.0) and (reward_advantage_unclipped < 0.0))
        reward_advantage_value = max(
            -float(self.ig_z_clip),
            min(float(self.ig_z_clip), float(reward_advantage_unclipped)),
        )
        reward_advantage_clip_hit = float(reward_advantage_value != reward_advantage_unclipped)

        log_psi_decay = 0.98
        new_mean = log_psi_decay * prev_ema_mean + (1.0 - log_psi_decay) * s_t
        new_var = log_psi_decay * prev_ema_var + (1.0 - log_psi_decay) * ((s_t - new_mean) ** 2)
        self.log_psi_mean_ema.fill_(float(new_mean))
        self.log_psi_var_ema.fill_(float(max(new_var, 1e-4)))
        self.log_psi_window.append(s_t)
        self.reward_window.append(ig_reward_value)

        mask_detached = mask.detach()
        selected_mask = mask_detached > 0.5
        soft_scores_safe = torch.nan_to_num(soft_scores, nan=0.5, posinf=1.0, neginf=0.0)
        p = soft_scores_safe.float()
        p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)
        logits = torch.logit(p, eps=1e-6)

        log_prob_selected = torch.log(p)
        if selected_mask.any():
            reinforce_term = log_prob_selected[selected_mask].mean()
        else:
            reinforce_term = torch.tensor(0.0, device=soft_scores.device, dtype=torch.float32)
        if not torch.isfinite(reinforce_term):
            reinforce_term = torch.tensor(0.0, device=soft_scores.device, dtype=torch.float32)

        rho = mask_detached.float().mean().to(device=soft_scores.device, dtype=torch.float32)
        rho_clamped = torch.clamp(rho.float(), min=1e-6, max=1.0 - 1e-6)
        default_logit_target = torch.log(rho_clamped / (1.0 - rho_clamped))
        if self.logit_anchor_target is not None:
            logit_anchor_target = torch.tensor(float(self.logit_anchor_target), device=soft_scores.device, dtype=torch.float32)
        else:
            logit_anchor_target = default_logit_target
        logits_mean = logits.mean()
        logit_anchor_penalty = (logits_mean - logit_anchor_target) ** 2
        logit_anchor_term = self.gamma_logit_anchor * logit_anchor_penalty

        log_p = torch.log(p)
        log_1mp = torch.log(1.0 - p)
        entropy = -(p * log_p + (1.0 - p) * log_1mp).mean()

        pg_loss = -torch.tensor(float(reward_advantage_value), device=soft_scores.device, dtype=torch.float32) * reinforce_term
        entropy_term = self.gamma_entropy * entropy
        loss = pg_loss - entropy_term + logit_anchor_term

        loss = loss.to(dtype=soft_scores.dtype)
        pg_loss = pg_loss.to(dtype=soft_scores.dtype)
        entropy_term = entropy_term.to(dtype=soft_scores.dtype)
        reinforce_term = reinforce_term.to(dtype=soft_scores.dtype)
        logit_anchor_penalty = logit_anchor_penalty.to(dtype=soft_scores.dtype)
        logit_anchor_term = logit_anchor_term.to(dtype=soft_scores.dtype)
        entropy = entropy.to(dtype=soft_scores.dtype)
        rho = rho.to(dtype=soft_scores.dtype)
        scores_mean = p.mean().to(dtype=soft_scores.dtype)
        logits_mean = logits.mean().to(dtype=soft_scores.dtype)
        logits_std = logits.std(unbiased=False).to(dtype=soft_scores.dtype)
        logits_min = logits.min().to(dtype=soft_scores.dtype)
        logits_max = logits.max().to(dtype=soft_scores.dtype)
        logits_flat = logits.detach().float().reshape(-1)
        if logits_flat.numel() > 0:
            logits_p90 = torch.quantile(logits_flat, 0.90).to(device=soft_scores.device, dtype=soft_scores.dtype)
            logits_p95 = torch.quantile(logits_flat, 0.95).to(device=soft_scores.device, dtype=soft_scores.dtype)
        else:
            logits_p90 = torch.tensor(0.0, device=soft_scores.device, dtype=soft_scores.dtype)
            logits_p95 = torch.tensor(0.0, device=soft_scores.device, dtype=soft_scores.dtype)
        logit_anchor_target = logit_anchor_target.to(dtype=soft_scores.dtype)
        std_adv_ema = math.sqrt(max(float(self.adv_var_ema.item()), 1e-8))

        aux = {
            "pg_loss": pg_loss,
            "entropy_term": entropy_term,
            "logit_anchor_term": logit_anchor_term,
        }

        return loss, torch.tensor(raw_delta, device=soft_scores.device, dtype=soft_scores.dtype), aux

    
    


class RouterLossWithGradientDetach(nn.Module):
    """
    Router wrapper with gradient detachment control.
    """
    def __init__(
        self,
        lambda_reinforce: float = 0.1,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.datavalve_loss = DataValveLoss(
            lambda_reinforce=lambda_reinforce,
            ema_decay=ema_decay,
        )
    
    def forward(
        self,
        val_loss: torch.Tensor,
        mask: torch.Tensor,
        soft_scores: torch.Tensor,
        clip_distance: torch.Tensor,
        cluster_ids: torch.Tensor,
        detach_llava_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
           Router   
        
        Args:
            val_loss:       (   REINFORCE   reward)
            mask:    mask (   )
            soft_scores: Router     (   policy gradient)
            clip_distance: CLIP   
            cluster_ids:    ID
            detach_llava_grad:      LLaVA    (         )
        
        Returns:
            loss: Router   
            loss_dict:     
        """
        #         REINFORCE L_val     detach
        # detach_llava_grad          
        
        return self.datavalve_loss(val_loss, mask, soft_scores, clip_distance, cluster_ids)


def compute_selection_statistics(
    mask: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_clusters: int = 20,
) -> Dict[str, float]:
    """
            
    
    Args:
        mask:    mask [B]
        cluster_ids:    ID [B]
        num_clusters:     
    
    Returns:
        stats:     
    """
    with torch.no_grad():
        B = mask.shape[0]
        selected = (mask > 0.5).float()
        num_selected = selected.sum().item()
        selection_rate = num_selected / B
        
        #          
        cluster_selected = []
        for k in range(num_clusters):
            cluster_mask = (cluster_ids == k).float()
            count = (selected * cluster_mask).sum().item()
            cluster_selected.append(count)
        
        #                 
        mean_per_cluster = num_selected / num_clusters if num_clusters > 0 else 0
        std_per_cluster = torch.std(torch.tensor(cluster_selected)).item()
        
        stats = {
            "num_selected": num_selected,
            "selection_rate": selection_rate,
            "mean_per_cluster": mean_per_cluster,
            "std_per_cluster": std_per_cluster,
            "cluster_distribution": cluster_selected,
        }
        
        return stats
