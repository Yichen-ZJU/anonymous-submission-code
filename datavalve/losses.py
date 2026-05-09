"""
     DataValve -       

       
1. L_sand:      (      ) -    
2. L_diversity:       (20     ) -    
3. L_reinforce:          (Policy Gradient) -     
4. L_val:      (Golden Set    Reward   )

        REINFORCE         
-        Router    L_sand      CLIP Predictor
-      L_val    Reward    Policy Gradient     
-    Router   "  "      Golden Set       

[ICML Enhancement]      
- CVaR (Conditional Value at Risk):         
- DPP (Determinantal Point Processes):        
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from typing import Dict, Tuple, Optional










class DataValveLoss(nn.Module):
    """
             
    
    L_router =  _sand * L_sand +  _diversity * L_diversity +  _reinforce * L_reinforce
    
         
    - L_reinforce    REINFORCE (Policy Gradient)   L_val       Router
    -     "Router     CLIP Predictor"     
    -   Router      "    "     Golden Set       
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
        
        # [ICML Enhancement]       
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
        self.group_weights = dict(GROUP_WEIGHTS)
        self.group_names = GROUP_NAMES
        
        #         
        
        # EMA baseline for variance reduction in REINFORCE
        self.register_buffer('val_loss_ema', torch.tensor(0.0))
        self.register_buffer('ema_initialized', torch.tensor(False))
        # [  D] Advantage    EMA          PPO    
        # [2026-03-09]      1.0    0.001 raw_advantage    0.03   raw_advantage    0.001
        #    =1.0   EMA   ~500                     loss    
        self.register_buffer('adv_var_ema', torch.tensor(0.001))
        self.register_buffer('reward_activity_ema', torch.tensor(0.05))
        # [2026-03-14] Delta advantage: L_val(t) - L_val(t-1)
        #          Router              EMA     
        self.register_buffer('prev_val_loss', torch.tensor(0.0))
        self.register_buffer('prev_initialized', torch.tensor(False))

        # V20: shard-aware baseline + IG gate         group reward          checkpoint

        self.register_buffer('log_psi_mean_ema', torch.tensor(0.0))
        self.register_buffer('log_psi_var_ema', torch.tensor(0.01))
        self.register_buffer('log_psi_initialized', torch.tensor(False))
        self.log_psi_window_size = 16
        self.log_psi_min_count = 4
        self.log_psi_window = deque(maxlen=self.log_psi_window_size)
        self.reward_window = deque(maxlen=self.log_psi_window_size)

        # [V17.3 Run B] shard-aware baseline   
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

        zero = torch.tensor(0.0, device=soft_scores.device, dtype=soft_scores.dtype)
        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_val": val_loss.detach() if isinstance(val_loss, torch.Tensor) else torch.tensor(val_loss, device=soft_scores.device, dtype=soft_scores.dtype),
            "loss_val_shard": val_loss.detach() if isinstance(val_loss, torch.Tensor) else torch.tensor(val_loss, device=soft_scores.device, dtype=soft_scores.dtype),
            "loss_reinforce": l_reinforce.detach(),
            "advantage": raw_delta.detach() if isinstance(raw_delta, torch.Tensor) else torch.tensor(raw_delta, device=soft_scores.device, dtype=soft_scores.dtype),
            "val_loss_ema": self.val_loss_ema.detach(),
            "selection_rate": mask.mean().detach(),
            "loss_reinforce_selected": rl_aux["reinforce_selected"].detach(),
            "loss_ratio_penalty": zero,
            "entropy_bonus": rl_aux["entropy"].detach(),
            "loss_logit_anchor": rl_aux["logit_anchor_penalty"].detach(),
            "logit_anchor_term": rl_aux["logit_anchor_term"].detach(),
            "rho_target": rl_aux["rho"].detach(),
            "scores_mean": rl_aux["scores_mean"].detach(),
            "logits_mean": rl_aux["logits_mean"].detach(),
            "logits_std": rl_aux["logits_std"].detach(),
            "logits_min": rl_aux["logits_min"].detach(),
            "logits_max": rl_aux["logits_max"].detach(),
            "logits_p90": rl_aux["logits_p90"].detach(),
            "logits_p95": rl_aux["logits_p95"].detach(),
            "logit_anchor_target": rl_aux["logit_anchor_target"].detach(),
            "advantage_task_aware": zero,
            "reward_group_used": zero,
            "reward_group_fallback": zero,
            "reward_group_active_count": zero,
            "reward_group_vqa_reason": zero,
            "reward_group_instruction": zero,
            "reward_group_ocr_text": zero,
            "reward_group_grounding": zero,
            "reward_group_count_vqa_reason": zero,
            "reward_group_count_instruction": zero,
            "reward_group_count_ocr_text": zero,
            "reward_group_count_grounding": zero,
            "reward_group_ema_vqa_reason": zero,
            "reward_group_ema_instruction": zero,
            "reward_group_ema_ocr_text": zero,
            "reward_group_ema_grounding": zero,
            "reward_group_mu_vqa_reason": zero,
            "reward_group_mu_instruction": zero,
            "reward_group_mu_ocr_text": zero,
            "reward_group_mu_grounding": zero,
            "reward_group_std_vqa_reason": zero,
            "reward_group_std_instruction": zero,
            "reward_group_std_ocr_text": zero,
            "reward_group_std_grounding": zero,
            "reward_group_delta_vqa_reason": zero,
            "reward_group_delta_instruction": zero,
            "reward_group_delta_ocr_text": zero,
            "reward_group_delta_grounding": zero,
            "reward_group_z_vqa_reason": zero,
            "reward_group_z_instruction": zero,
            "reward_group_z_ocr_text": zero,
            "reward_group_z_grounding": zero,
            "reward_group_contrib_vqa_reason": zero,
            "reward_group_contrib_instruction": zero,
            "reward_group_contrib_ocr_text": zero,
            "reward_group_contrib_grounding": zero,
            "reward_group_contrib_abs_sum": zero,
            "reward_group_contrib_balance": zero,
            "reward_group_contrib_entropy": zero,
            "reward_group_clip_hit_vqa_reason": zero,
            "reward_group_clip_hit_instruction": zero,
            "reward_group_clip_hit_ocr_text": zero,
            "reward_group_clip_hit_grounding": zero,
            "reward_group_floor_hit_vqa_reason": zero,
            "reward_group_floor_hit_instruction": zero,
            "reward_group_floor_hit_ocr_text": zero,
            "reward_group_floor_hit_grounding": zero,
            "reward_group_norm_mode_vqa_reason": zero,
            "reward_group_norm_mode_instruction": zero,
            "reward_group_norm_mode_ocr_text": zero,
            "reward_group_norm_mode_grounding": zero,
            "advantage_scale": rl_aux["advantage_scale"].detach(),
            "advantage_std_ema": rl_aux["advantage_std_ema"].detach(),
            "reward_baseline_mu_shard": rl_aux["reward_baseline_mu_shard"].detach(),
            "reward_baseline_std_shard": rl_aux["reward_baseline_std_shard"].detach(),
            "reward_advantage_z": rl_aux["reward_advantage_z"].detach(),
            "reward_activity": rl_aux["reward_activity"].detach(),
            "reward_snr": rl_aux["reward_snr"].detach(),
            "group_adv_balance": zero,
            "baseline_alpha": zero,
            "golden_shard_id": rl_aux["golden_shard_id"].detach(),
            "ig_raw_delta": rl_aux["raw_delta"].detach(),
            "ig_delta_norm": rl_aux["delta_norm"].detach(),
            "ig_influence_gate_raw": rl_aux["influence_gate_raw"].detach(),
            "ig_influence_gate_norm": rl_aux["influence_gate_norm"].detach(),
            "ig_reward": rl_aux["ig_reward"].detach(),
            "ig_reward_baseline": rl_aux["reward_baseline"].detach(),
            "ig_reward_advantage": rl_aux["reward_advantage"].detach(),
            "ig_reward_advantage_unclipped": rl_aux["reward_advantage_unclipped"].detach(),
            "ig_reward_window_count": rl_aux["reward_window_count"].detach(),
            "ig_reward_advantage_ready": rl_aux["reward_advantage_ready"].detach(),
            "ig_reward_advantage_clip_hit": rl_aux["reward_advantage_clip_hit"].detach(),
            "ig_reward_sign_flip": rl_aux["reward_sign_flip"].detach(),
            "ig_reward_neg_to_pos_flip": rl_aux["reward_neg_to_pos_flip"].detach(),
            "ig_reward_pos_to_neg_flip": rl_aux["reward_pos_to_neg_flip"].detach(),
            "ig_no_gate_reward": rl_aux["no_gate_reward"].detach(),
            "ig_gate_reward_ratio": rl_aux["gate_reward_ratio"].detach(),
            "ig_influence_log_mean": rl_aux["influence_log_mean"].detach(),
            "ig_influence_grad_norm_mean": rl_aux["influence_grad_norm_mean"].detach(),
            "ig_influence_eta": rl_aux["influence_eta"].detach(),
            "ig_influence_grad_norm_sq": rl_aux["influence_grad_norm_sq"].detach(),
            "ig_influence_prev_grad_norm_sq": rl_aux["influence_prev_grad_norm_sq"].detach(),
            "ig_influence_grad_dot_prev": rl_aux["influence_grad_dot_prev"].detach(),
            "ig_influence_cos_phi": rl_aux["influence_cos_phi"].detach(),
            "ig_influence_psi": rl_aux["influence_psi"].detach(),
            "ig_influence_log_utility": rl_aux["influence_log_utility"].detach(),
            "ig_log_psi_value": rl_aux["log_psi_value"].detach(),
            "ig_log_psi_mean": rl_aux["log_psi_mean"].detach(),
            "ig_log_psi_local_mean": rl_aux["log_psi_local_mean"].detach(),
            "ig_log_psi_local_std": rl_aux["log_psi_local_std"].detach(),
            "ig_log_psi_window_count": rl_aux["log_psi_window_count"].detach(),
            "ig_log_psi_ema_mean": rl_aux["log_psi_ema_mean"].detach(),
            "ig_log_psi_ema_std": rl_aux["log_psi_ema_std"].detach(),
            "ig_log_psi_z": rl_aux["log_psi_z"].detach(),
            "ig_log_psi_gamma": rl_aux["log_psi_gamma"].detach(),
            "ig_gate_ready": rl_aux["gate_ready"].detach(),
        }

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
            "ratio_term": torch.tensor(0.0, device=soft_scores.device, dtype=soft_scores.dtype),
            "entropy_term": entropy_term,
            "reinforce_selected": reinforce_term,
            "ratio_penalty": torch.tensor(0.0, device=soft_scores.device, dtype=soft_scores.dtype),
            "logit_anchor_penalty": logit_anchor_penalty,
            "logit_anchor_term": logit_anchor_term,
            "entropy": entropy,
            "rho": rho,
            "scores_mean": scores_mean,
            "logits_mean": logits_mean,
            "logits_std": logits_std,
            "logits_min": logits_min,
            "logits_max": logits_max,
            "logits_p90": logits_p90,
            "logits_p95": logits_p95,
            "logit_anchor_target": logit_anchor_target,
            "advantage_scale": torch.tensor(float(safe_baseline_std), device=soft_scores.device, dtype=soft_scores.dtype),
            "advantage_std_ema": torch.tensor(float(std_adv_ema), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_baseline_mu_shard": torch.tensor(float(baseline_mu), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_baseline_std_shard": torch.tensor(float(safe_baseline_std), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_advantage_z": torch.tensor(float(delta_norm_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_activity": self.reward_activity_ema.detach().to(device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_snr": torch.tensor(float(abs(raw_delta) / max(safe_baseline_std, 1e-8)), device=soft_scores.device, dtype=soft_scores.dtype),
            "golden_shard_id": torch.tensor(float(shard_key), device=soft_scores.device, dtype=soft_scores.dtype),
            "raw_delta": torch.tensor(float(raw_delta), device=soft_scores.device, dtype=soft_scores.dtype),
            "delta_norm": torch.tensor(float(delta_norm_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_gate_raw": torch.tensor(float(influence_raw_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_gate_norm": torch.tensor(float(grad_gate_norm_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "ig_reward": torch.tensor(float(ig_reward_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_baseline": torch.tensor(float(reward_baseline_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_advantage": torch.tensor(float(reward_advantage_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_advantage_unclipped": torch.tensor(float(reward_advantage_unclipped), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_window_count": torch.tensor(float(reward_window_count), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_advantage_ready": torch.tensor(float(reward_advantage_ready), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_advantage_clip_hit": torch.tensor(float(reward_advantage_clip_hit), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_sign_flip": torch.tensor(float(reward_sign_flip), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_neg_to_pos_flip": torch.tensor(float(reward_neg_to_pos_flip), device=soft_scores.device, dtype=soft_scores.dtype),
            "reward_pos_to_neg_flip": torch.tensor(float(reward_pos_to_neg_flip), device=soft_scores.device, dtype=soft_scores.dtype),
            "no_gate_reward": torch.tensor(float(delta_norm_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "gate_reward_ratio": torch.tensor(float(grad_gate_norm_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_log_mean": torch.tensor(float(influence_log_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_grad_norm_mean": torch.tensor(float(current_grad_norm), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_eta": torch.tensor(float(eta_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_grad_norm_sq": torch.tensor(float(grad_norm_sq), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_prev_grad_norm_sq": torch.tensor(float(prev_grad_norm_sq), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_grad_dot_prev": torch.tensor(float(grad_dot_prev), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_cos_phi": torch.tensor(float(cos_phi_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_psi": torch.tensor(float(influence_psi_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "influence_log_utility": torch.tensor(float(influence_log_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_value": torch.tensor(float(s_t), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_mean": torch.tensor(float(local_mean), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_local_mean": torch.tensor(float(local_mean), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_local_std": torch.tensor(float(local_std), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_window_count": torch.tensor(float(log_psi_window_count), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_ema_mean": torch.tensor(float(prev_ema_mean), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_ema_std": torch.tensor(float(prev_ema_std), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_z": torch.tensor(float(z_t), device=soft_scores.device, dtype=soft_scores.dtype),
            "log_psi_gamma": torch.tensor(float(grad_gate_norm_value), device=soft_scores.device, dtype=soft_scores.dtype),
            "gate_ready": torch.tensor(float(gate_ready), device=soft_scores.device, dtype=soft_scores.dtype),
        }

        return loss, torch.tensor(raw_delta, device=soft_scores.device, dtype=soft_scores.dtype), aux

    
    


class RouterLossWithGradientDetach(nn.Module):
    """
    Router             
    
       bi-level optimization:
    -   : LLaVA     Router    
    -   : Router        L_reinforce     
    
            REINFORCE L_val         Router
    """
    
    def __init__(
        self,
        lambda_reinforce: float = 0.1,
        num_clusters: int = 20,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.datavalve_loss = DataValveLoss(
            lambda_reinforce=lambda_reinforce,
            num_clusters=num_clusters,
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
