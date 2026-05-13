"""
     DataValve -      

     
1.      
   -   : LLaVA    Router        
   -   : Router    Golden Set      +   /       

2.      
   - Warmup   :     LLaVA 100%    
   - Router   : Router + LLaVA     

    
-     grpo_valve_trainer.py, grpo_valve_collator.py, grpo_trainer.py     
-    DeepSpeed ZeRO-3
-    LoRA   
"""

import os
import json
import time
import numpy as np
import random
import weakref
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from collections import Counter, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from transformers import Trainer, TrainerCallback
    from transformers.trainer_pt_utils import LabelSmoother
    IGNORE_TOKEN_ID = LabelSmoother.ignore_index
except ImportError:
    IGNORE_TOKEN_ID = -100

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# DeepSpeed ZeRO-3   
try:
    from deepspeed import zero
    HAS_DEEPSPEED_ZERO = True
except ImportError:
    HAS_DEEPSPEED_ZERO = False
    #      deepspeed      dummy class
    class zero:
        @staticmethod
        def Infinite():
            from contextlib import nullcontext
            return nullcontext()

from .config import DataValveConfig
from .router import DataValveRouter
from .losses import DataValveLoss


class DataValveValveTrainer:
    """
        Valve Trainer -    GRPOValveTrainer   
    
         
    1.    Super-Batch    Top-K    
    2.    CLIP    + CrossAttention Router
    3.        +       + REINFORCE   
    4.            
    """
    
    def __init__(
        self,
        router: DataValveRouter,
        clip_features_path: str,
        cluster_ids_path: Optional[str] = None,
        grad_norm_path: Optional[str] = None,
        total_batches: int = 10000,
        target_ratio: float = 0.2,
        warmup_ratio: float = 0.03,
        lambda_reinforce: float = 0.1,
        advantage_mode: str = "ema",  # "ema": L_val-EMA(      ), "delta": L_val(t)-L_val(t-1)(      )
        gamma_ratio_penalty: float = 0.5,
        gamma_entropy: float = 0.01,
        gamma_logit_anchor: float = 0.0,
        logit_anchor_target: Optional[float] = None,
        advantage_norm_floor: float = 0.05,
        reward_activity_ema_decay: float = 0.9,
        ig_z_clip: float = 3.0,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None,
        save_interval: int = 1000,
    ):
        """
            DataValveValveTrainer

                  CLIP               
        
        Args:
            router: DataValveRouter   
            clip_features_path: CLIP       
            cluster_ids_path:    ID     
            total_batches:       
            target_ratio:       
            warmup_ratio: Warm-up     
            lambda_reinforce: REINFORCE     
            ema_decay: EMA baseline    
            device:   
            checkpoint_dir:                    
            save_interval:        N   batch      
        """
        self.router = router
        self.total_batches = total_batches
        self.target_ratio = target_ratio
        self.warmup_ratio = warmup_ratio
        self.warmup_batches = int(total_batches * warmup_ratio)
        self.lambda_reinforce = lambda_reinforce
        self.device = device
        self.advantage_norm_floor = advantage_norm_floor
        self.reward_activity_ema_decay = reward_activity_ema_decay


        #       
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.realtime_log_file = None
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.realtime_log_file = os.path.join(checkpoint_dir, "selected_indices_realtime.jsonl")

        #                 
        self.clip_features_path = clip_features_path
        self.cluster_ids_path = cluster_ids_path
        self.grad_norm_path = grad_norm_path
        self.clip_features_tensor = None
        self.id_to_index = None
        self.cluster_ids = None
        self._features_loaded = False
        self.prev_lora_grad_by_name: Dict[str, torch.Tensor] = {}
        self.accum_lora_grad_snapshot: Dict[str, torch.Tensor] = {}
        self.llava_micro_step_count: int = 0
        self.selection_topk_margin_ema: float = 0.0
        self.selection_topk_margin_ema_initialized: bool = False
        self.lora_grad_hook_handles: list = []
        self.online_influence_hooks_installed: bool = False
        self.current_lora_grad_by_name: Dict[str, torch.Tensor] = {}
        self.online_influence_capture_enabled: bool = False
        self.lora_param_lr_by_name: Dict[str, float] = {}
        self.original_collate_fn = None

        #        (   REINFORCE)
        self.loss_fn = DataValveLoss(
            lambda_reinforce=lambda_reinforce,
            advantage_mode=advantage_mode,
            target_ratio=target_ratio,
            gamma_ratio_penalty=gamma_ratio_penalty,
            gamma_entropy=gamma_entropy,
            gamma_logit_anchor=gamma_logit_anchor,
            logit_anchor_target=logit_anchor_target,
            ig_z_clip=ig_z_clip,
        )

        #     
        self.current_batch = 0
        self.total_sampled = 0
        self.total_candidates = 0

        #     
        self.selected_indices = []
        self.batch_history = []
        self.source_to_group = {}
        self.group_names = []
        self.golden_source_counter = Counter()

        self.current_selected_indices = []
        self.current_log_probs = None
        self.current_mask = None
        self.current_soft_scores = None  # REINFORCE   
        self.current_clip_distance = None
        self.current_cluster_ids = None

        self.grpo_sampler = None
        self.golden_shard_cursor: int = 0

        print(f"[DataValveValveTrainer]      ")
        print(f"      : {target_ratio:.1%}")
        print(f"  Warm-up   : {self.warmup_batches}")
        print(f"   _reinforce: {lambda_reinforce} (REINFORCE / Policy Gradient)")
        print(f"  advantage_norm_floor: {advantage_norm_floor}")
        print(f"  reward_activity_ema_decay: {reward_activity_ema_decay}")
        if checkpoint_dir:
            print(f"      : {self.realtime_log_file}")
            print(f"      :   {save_interval}       {checkpoint_dir}")

    def __getstate__(self):
        """           """
        state = self.__dict__.copy()
        #      id_to_index    
        state['id_to_index'] = None
        return state

    def __setstate__(self, state):
        """        id_to_index"""
        self.__dict__.update(state)
        #        id_to_index         
        if self.id_to_index is None:
            self.id_to_index = {}

    def _ensure_features_loaded(self):
        """   CLIP            """
        if self._features_loaded:
            return

        print("[DataValveValveTrainer]      CLIP   ...")
        self._load_clip_features(self.clip_features_path)
        self._load_cluster_ids(self.cluster_ids_path)
        self._features_loaded = True
        print("[DataValveValveTrainer]   CLIP       ")

    def _load_clip_features(self, clip_features_path: str):
        """   CLIP       mmap       """
        import torch.distributed as dist

        #              
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            print(f"[Rank 0]    CLIP   : {clip_features_path}")

        #       .npy         
        npy_path = clip_features_path.replace('.pt', '.npy')
        keys_path = clip_features_path.replace('.pt', '_keys.npy')

        if os.path.exists(npy_path) and os.path.exists(keys_path):
            #    .npy + mmap              
            if rank == 0:
                print(f"      .npy       mmap   ...")

            # mmap_mode='r':                    
            self.clip_features_tensor = torch.from_numpy(
                np.load(npy_path, mmap_mode='r')
            ).float()

            #          rank          
            keys_array = np.load(keys_path, allow_pickle=True)
            self.id_to_index = {str(k): i for i, k in enumerate(keys_array)}

            if rank == 0:
                print(f"      : {self.clip_features_tensor.shape}")
                print(f"     : {len(self.id_to_index)}")
                print(f"    mmap                  ")

        elif os.path.exists(clip_features_path):
            #     .pt          
            if rank == 0:
                print(f"     .pt          .npy       ")
                data = torch.load(clip_features_path, map_location='cpu')

                if isinstance(data, dict):
                    print(f"              Tensor...")
                    num_samples = len(data)
                    print(f"      : {num_samples}")

                    keys_list = list(data.keys())
                    self.id_to_index = {str(k): i for i, k in enumerate(keys_list)}

                    sample_feat = next(iter(data.values()))
                    if isinstance(sample_feat, torch.Tensor):
                        feat_dim = sample_feat.shape[0]
                    else:
                        feat_dim = len(sample_feat)

                    print(f"       : ({num_samples}, {feat_dim})")
                    self.clip_features_tensor = torch.zeros(num_samples, feat_dim, dtype=torch.float32)

                    for i, key in enumerate(keys_list):
                        v = data[key]
                        if isinstance(v, torch.Tensor):
                            self.clip_features_tensor[i] = v.float()
                        else:
                            self.clip_features_tensor[i] = torch.tensor(v, dtype=torch.float32)

                        if (i + 1) % 100000 == 0:
                            print(f"       : {i+1}/{num_samples}")

                    print(f"      : {self.clip_features_tensor.shape}")
                    self.clip_features_tensor.share_memory_()
                    print(f"             ")

                    del data, keys_list
                    import gc
                    gc.collect()
                else:
                    print(f"      : {data.shape}")
                    self.clip_features_tensor = data.float().share_memory_()
                    self.id_to_index = None
                    print(f"             ")
            else:
                self.clip_features_tensor = None
                self.id_to_index = None

            if dist.is_initialized():
                dist.barrier()
        else:
            if rank == 0:
                print(f"[Warning] CLIP        : {clip_features_path}")
            self.clip_features_tensor = None
            self.id_to_index = {}
    
    
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

    def is_warmup_done(self) -> bool:
        """   Warm-up               """
        #      trainer       pickle   
        return self.current_batch >= self.warmup_batches
    
    def get_clip_features(self, unique_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                CLIP   

        Args:
            unique_idx:       

        Returns:
            (image_feature, text_feature):   [768]
        """
        if self.clip_features_tensor is None:
            return torch.zeros(768), torch.zeros(768)

        idx_str = str(unique_idx)

        #     
        if self.id_to_index is not None:
            if idx_str in self.id_to_index:
                real_idx = self.id_to_index[idx_str]
                feat = self.clip_features_tensor[real_idx]
            else:
                return torch.zeros(768), torch.zeros(768)
        else:
            #    unique_idx       
            try:
                feat = self.clip_features_tensor[int(unique_idx)]
            except (IndexError, ValueError):
                return torch.zeros(768), torch.zeros(768)

        #    image/text (  768   image   768   text)
        image_feat = feat[:768]
        text_feat = feat[768:1536]
        return image_feat, text_feat
    
    def get_batch_features(
        self,
        unique_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
             CLIP       ID

        Args:
            unique_indices:       

        Returns:
            (image_features, text_features, cluster_ids): [B, 768], [B, 768], [B]
        """
        #      CLIP   
        self._ensure_features_loaded()

        image_features = []
        text_features = []
        cluster_ids = []

        for idx in unique_indices:
            image_feat, text_feat = self.get_clip_features(idx)
            image_features.append(image_feat)
            text_features.append(text_feat)

            #      ID
            cid = self.cluster_ids.get(str(idx), 0)
            cluster_ids.append(cid)

        #      CPU     
        #       select_from_candidates      
        stacked_image = torch.stack(image_features)
        stacked_text = torch.stack(text_features)
        stacked_cluster = torch.tensor(cluster_ids, dtype=torch.long)

        return (stacked_image, stacked_text, stacked_cluster)
    
    def select_from_candidates(
        self,
        candidate_indices: List[int],
        num_samples: int,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
           Router          
        
        Args:
            candidate_indices:          Super-Batch 
            num_samples:         Final Batch Size 
        
        Returns:
            (selected_indices, mask, clip_distance)
        """
        if len(candidate_indices) <= num_samples:
            #          
            return candidate_indices, None, None
        
        # 1.    CLIP       CPU 
        image_features, text_features, cluster_ids = self.get_batch_features(
            candidate_indices
        )

        # 2.       Router            
        #    num_workers=0                 CUDA
        router_device = next(self.router.parameters()).device
        router_dtype = next(self.router.parameters()).dtype

        image_features = image_features.to(device=router_device, dtype=router_dtype)
        text_features = text_features.to(device=router_device, dtype=router_dtype)

        # 3. Router    Top-K
        mask, topk_indices, logits = self.router.get_selection_mask(
            image_features, text_features, k=num_samples
        )

        # 4.    CLIP           
        clip_distance = self.router.compute_clip_distance(image_features, text_features)

        # 5.    soft scores (   REINFORCE)
        soft_scores = torch.sigmoid(logits)

        # 6.        
        selected_indices = [candidate_indices[i] for i in topk_indices.cpu().tolist()]

        # 7.             soft_scores    REINFORCE 
        self.current_selected_indices = selected_indices
        self.current_mask = mask
        self.current_cluster_ids = cluster_ids
        self.current_clip_distance = clip_distance
        self.current_soft_scores = soft_scores  # REINFORCE   
        
        return selected_indices, mask, clip_distance
    
    def compute_router_loss(
        self,
        val_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
           Router          Router    
        
                 current_mask           DataLoader        
             compute_router_loss_from_batch    
        
        Args:
            val_loss:       (   REINFORCE   reward)
        
        Returns:
            (total_loss, loss_dict)
        """
        if self.current_mask is None:
            return torch.tensor(0.0, device=self.device), {}
        
        return self.loss_fn(
            val_loss=val_loss,
            mask=self.current_mask,
            soft_scores=self.current_soft_scores,  # REINFORCE   
            clip_distance=self.current_clip_distance,
            cluster_ids=self.current_cluster_ids,
        )
    
    def compute_router_loss_from_batch(
        self,
        val_loss: torch.Tensor,
        mask: Optional[torch.Tensor],
        soft_scores: Optional[torch.Tensor],
        clip_distance: Optional[torch.Tensor],
        cluster_ids: Optional[torch.Tensor],
        img_features: Optional[torch.Tensor] = None,
        txt_features: Optional[torch.Tensor] = None,
        group_loss_dict: Optional[Dict[str, float]] = None,
        group_count_dict: Optional[Dict[str, int]] = None,
        shard_id: Optional[int] = None,
        router_progress: Optional[float] = None,
        selected_action_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
           Router        batch     Router    
        
        Args:
            val_loss:       (   REINFORCE   reward)
            mask: Router     mask
            soft_scores: Router   soft scores (   REINFORCE)
        
        Returns:
            (total_loss, loss_dict)
        """
        if mask is None:
            return torch.tensor(0.0, device=self.device), {}
        
        #          
        if mask.device != self.device:
            mask = mask.to(self.device)
        if soft_scores is not None and soft_scores.device != self.device:
            soft_scores = soft_scores.to(self.device)
        if clip_distance is not None and clip_distance.device != self.device:
            clip_distance = clip_distance.to(self.device)
        if cluster_ids is not None and cluster_ids.device != self.device:
            cluster_ids = cluster_ids.to(self.device)
        if img_features is not None and img_features.device != self.device:
            img_features = img_features.to(self.device)
        if txt_features is not None and txt_features.device != self.device:
            txt_features = txt_features.to(self.device)
        
        return self.loss_fn(
            val_loss=val_loss,
            mask=mask,
            soft_scores=soft_scores,
            clip_distance=clip_distance,
            cluster_ids=cluster_ids,
            img_features=img_features,
            txt_features=txt_features,
            group_loss_dict=group_loss_dict,
            group_count_dict=group_count_dict,
            shard_id=shard_id,
            router_progress=router_progress,
            selected_action_stats=selected_action_stats,
        )
    
    def step(self):
        """      """
        self.current_batch += 1

        #        
        if self.current_selected_indices:
            self.selected_indices.extend(self.current_selected_indices)
            self.total_sampled += len(self.current_selected_indices)

            #             batch     
            if self.realtime_log_file:
                self._append_realtime_log()

        #                
        # if self.checkpoint_dir and self.save_interval > 0:
        #     if self.current_batch % self.save_interval == 0:
        #         self._save_checkpoint()
    
    def log_batch_info(self, super_batch_size: int, final_batch_size: int):
        """      """
        self.total_candidates += super_batch_size
        
        info = {
            'batch': self.current_batch,
            'super_batch_size': super_batch_size,
            'final_batch_size': final_batch_size,
            'selected_indices': self.current_selected_indices.copy() if self.current_selected_indices else [],
        }
        self.batch_history.append(info)
    
    def get_stats(self) -> Dict[str, Any]:
        """      """
        unique_selected = len(set(self.selected_indices))
        actual_ratio = self.total_sampled / self.total_candidates if self.total_candidates > 0 else 0

        return {
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'warmup_batches': self.warmup_batches,
            'warmup_done': self.is_warmup_done(),
            'target_ratio': self.target_ratio,
            'actual_ratio': actual_ratio,
            'total_sampled': self.total_sampled,
            'total_candidates': self.total_candidates,
            'unique_selected': unique_selected,
        }

    def summarize_counts(self, counter: Counter) -> Dict[str, float]:
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {
            key: round(value / total, 4)
            for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        }

    def set_golden_shard_paths(self, shard_paths: List[str]):
        self.golden_shard_paths = [str(path) for path in shard_paths]
        self.golden_shard_cursor = 0

    def get_next_golden_shard_path(self) -> Tuple[Optional[str], int]:
        if not self.golden_shard_paths:
            return None, -1
        shard_id = self.golden_shard_cursor % len(self.golden_shard_paths)
        shard_path = self.golden_shard_paths[shard_id]
        self.golden_shard_cursor = (self.golden_shard_cursor + 1) % len(self.golden_shard_paths)
        return shard_path, shard_id

    def _append_realtime_log(self):
        """               batch     """
        import json
        import os
        import torch.distributed as dist

        if not self.realtime_log_file or not self.current_selected_indices:
            return

        log_path = self.realtime_log_file
        if dist.is_initialized():
            rank = dist.get_rank()
            base, ext = os.path.splitext(self.realtime_log_file)
            log_path = f"{base}_rank{rank}{ext}"

        #                                    
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        #      JSON    JSONL    
        log_entry = {
            'batch': self.current_batch,
            'indices': self.current_selected_indices,
            'count': len(self.current_selected_indices),
        }

        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            #                
            print(f"[Warning]         : {e}")

    def _save_checkpoint(self):
        """               """
        import os
        import json
        from collections import Counter

        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        #         
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"selected_indices_batch{self.current_batch}.json"
        )

        stats = self.get_stats()
        index_counts = Counter(self.selected_indices)

        data = {
            'current_batch': self.current_batch,
            'selected_indices': list(set(self.selected_indices)),
            'index_counts': dict(index_counts),
            'total_unique': len(set(self.selected_indices)),
            'total_sampled': self.total_sampled,
            'statistics': stats,
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[DataValveValveTrainer]       : {checkpoint_path} (batch {self.current_batch})")

    
    def gather_all_selected_indices(self) -> List[int]:
        """
        [2025-01-14 FIX]      GPU   selected_indices
        
                        GPU     1/world_size     
                  GPU       
        """
        import torch.distributed as dist
        
        if not dist.is_initialized():
            #            
            return self.selected_indices
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        #     selected_indices    tensor
        local_indices = torch.tensor(self.selected_indices, dtype=torch.long, device=self.device)
        local_count = torch.tensor([len(self.selected_indices)], dtype=torch.long, device=self.device)
        
        #      rank    
        all_counts = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        all_counts = [c.item() for c in all_counts]
        max_count = max(all_counts)
        
        #        
        padded_indices = torch.zeros(max_count, dtype=torch.long, device=self.device)
        padded_indices[:len(self.selected_indices)] = local_indices
        
        #      rank   indices
        all_indices_list = [torch.zeros(max_count, dtype=torch.long, device=self.device) for _ in range(world_size)]
        dist.all_gather(all_indices_list, padded_indices)
        
        #          0    0          
        all_selected_indices = []
        for i, (indices, count) in enumerate(zip(all_indices_list, all_counts)):
            valid_indices = indices[:count].cpu().tolist()
            all_selected_indices.extend(valid_indices)
            if rank == 0:
                print(f"  [Rank {i}]    {count:,}      ")
        
        if rank == 0:
            print(f"  [  ]    {len(all_selected_indices):,}          {len(set(all_selected_indices)):,}  ")
        
        return all_selected_indices
    
    def save_selected_indices(self, output_path: str):
        """            realtime                    """
        import torch.distributed as dist
        import glob
        
        #          DeepSpeed                   
        print("[DataValveValveTrainer]   realtime         ...")
        
        all_selected_indices = []
        realtime_files = []
        if self.realtime_log_file:
            base, ext = os.path.splitext(self.realtime_log_file)
            realtime_files = sorted(glob.glob(f"{base}_rank*{ext}"))
            if not realtime_files and os.path.exists(self.realtime_log_file):
                realtime_files = [self.realtime_log_file]

        if realtime_files:
            try:
                for fp in realtime_files:
                    with open(fp, 'r') as f:
                        for line in f:
                            record = json.loads(line.strip())
                            indices = record.get('indices', [])
                            all_selected_indices.extend(indices)
                print(f"    {len(realtime_files)}   realtime      {len(all_selected_indices):,}    ")
            except Exception as e:
                print(f"        realtime     : {e}")
                # Fallback:       selected_indices
                all_selected_indices = self.selected_indices
                print(f"  Fallback:        {len(all_selected_indices):,}  ")
        else:
            # Fallback:       selected_indices
            all_selected_indices = self.selected_indices
            print(f"  Fallback:        {len(all_selected_indices):,}  ")
        
        stats = self.get_stats()
        index_counts = Counter(all_selected_indices)
        
        #   
        total_sampled_all = len(all_selected_indices)
        total_unique_all = len(set(all_selected_indices))
        total_candidates_all = self.total_candidates
        
        #            dist.all_reduce      
        try:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        except:
            world_size = 1
        total_candidates_all = self.total_candidates * world_size
        
        actual_ratio_all = total_sampled_all / total_candidates_all if total_candidates_all > 0 else 0
        
        data = {
            'selected_indices': list(set(all_selected_indices)),
            'index_counts': dict(index_counts),
            'total_unique': total_unique_all,
            'total_sampled': total_sampled_all,
            'total_candidates': total_candidates_all,
            'actual_ratio': actual_ratio_all,
            'statistics': stats,
            'distributed_info': {
                'world_size': dist.get_world_size() if dist.is_initialized() else 1,
                'source': 'realtime_file',
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[DataValveValveTrainer]        : {output_path}")
        print(f"     : {total_unique_all:,}           : {actual_ratio_all:.2%}")
    
    def export_selected_dataset(self, original_data_path: str, output_path: str):
        """           realtime                    """
        import glob

        with open(original_data_path, 'r') as f:
            original_data = json.load(f)
        
        all_selected_indices = []
        realtime_files = []
        if self.realtime_log_file:
            base, ext = os.path.splitext(self.realtime_log_file)
            realtime_files = sorted(glob.glob(f"{base}_rank*{ext}"))
            if not realtime_files and os.path.exists(self.realtime_log_file):
                realtime_files = [self.realtime_log_file]

        if realtime_files:
            try:
                for fp in realtime_files:
                    with open(fp, 'r') as f:
                        for line in f:
                            record = json.loads(line.strip())
                            indices = record.get('indices', [])
                            all_selected_indices.extend(indices)
            except Exception as e:
                print(f"        realtime     : {e}")
                all_selected_indices = self.selected_indices
        else:
            all_selected_indices = self.selected_indices
        
        unique_indices = set(all_selected_indices)
        
        selected_data = []
        for item in original_data:
            idx = item.get('unique_idx', None)
            if idx is not None and idx in unique_indices:
                selected_data.append(item)
        
        with open(output_path, 'w') as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)
        
        print(f"[DataValveValveTrainer]       : {output_path}")
        print(f"       : {len(original_data)}")
        print(f"       : {len(selected_data)}")


def get_modality_length_grouped_indices_for_superbatch(lengths, batch_size, world_size, generator=None):
    """
    [2026-01-27 NEW]   Super-Batch            
    
        LLaVA   get_modality_length_grouped_indices 
          LLaVA             
    
    Args:
        lengths:          =      =    
        batch_size: batch   
        world_size:     world size
        generator:       
    
    Returns:
                     
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    #                      
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices_for_superbatch(lengths, batch_size, world_size, generator=generator)
    
    #            
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    
    #        
    # generator=None   torch.randperm              rank          
    #     rank       all_indices       batch        NCCL   
    #     8   group_by_modality_length=True    step 0    
    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_for_superbatch(
        mm_lengths, batch_size, world_size, generator=generator)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_for_superbatch(
        lang_lengths, batch_size, world_size, generator=generator)]
    
    #     megabatches
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    
    #          batch
    last_mm = mm_megabatches[-1] if mm_megabatches else []
    last_lang = lang_megabatches[-1] if lang_megabatches else []
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1] if mm_megabatches and lang_megabatches else mm_megabatches[:-1] if mm_megabatches else lang_megabatches[:-1]
    
    #      megabatches   
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    
    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices_for_superbatch(lengths, batch_size, world_size, generator=None):
    """
    [2026-01-27 NEW]           
    
        LLaVA   get_length_grouped_indices 
    """
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    
    #    megabatch          
    megabatches = [sorted(megabatch, key=lambda i: abs(lengths[i]), reverse=True) for megabatch in megabatches]
    
    #         GPU
    def split_to_even_chunks(indices_list, lengths_list, num_chunks):
        if len(indices_list) % num_chunks != 0:
            return [indices_list[i::num_chunks] for i in range(num_chunks)]
        
        num_indices_per_chunk = len(indices_list) // num_chunks
        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        
        for index in indices_list:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += abs(lengths_list[index])
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float("inf")
        
        return chunks
    
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class DataValveSuperBatchSampler(Sampler):
    """
        Super-Batch             
    
       GRPOSuperBatchSampler   
    
    [2026-01-07 FIX]          
    -    GPU     1/world_size    
    -    8    4           
    
    [2026-01-27 FIX]           
    -     LLaVA       group_by_modality_length   
    -       batch             padding   
    """
    
    def __init__(
        self,
        data_source,
        final_batch_size: int,
        target_ratio: float,
        warmup_ratio: float,
        shuffle: bool = True,
        drop_last: bool = True,
        rank: int = None,
        world_size: int = None,
        seed: int = 42,
        group_by_modality_length: bool = True,  # [2026-01-27 NEW]       
    ):
        """
        Args:
            data_source:    
            final_batch_size:    Batch   
            target_ratio:       
            warmup_ratio: Warm-up   
            shuffle:     
            drop_last:            
            rank:       rank       
            world_size:            
            seed:                    
            group_by_modality_length: [2026-01-27]             LLaVA      
        """
        self.data_source = data_source
        self.final_batch_size = final_batch_size
        self.target_ratio = target_ratio
        self.warmup_ratio = warmup_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.group_by_modality_length = group_by_modality_length
        
        self.modality_lengths = None
        if group_by_modality_length and hasattr(data_source, 'modality_lengths'):
            self.modality_lengths = data_source.modality_lengths
            if self.modality_lengths and len(self.modality_lengths) > 0:
                print(f"[SuperBatchSampler]       group_by_modality_length")
                print(f"         : {len(self.modality_lengths):,}")
                #            
                mm_count = sum(1 for l in self.modality_lengths if l > 0)
                lang_count = sum(1 for l in self.modality_lengths if l < 0)
                print(f"       : {mm_count:,},      : {lang_count:,}")
            else:
                self.modality_lengths = None
                print(f"[SuperBatchSampler]    modality_lengths             ")
        else:
            if group_by_modality_length:
                print(f"[SuperBatchSampler]           modality_lengths          ")
        
        #          
        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if world_size is None and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1
        
        #     
        self.total_samples = len(data_source)
        
        #     bug:      Rank          batch NCCL   
        #   :    Rank                  
        self.num_samples = self.total_samples // self.world_size
        
        self._compute_stage_params()
        
        #    rank 0       
        if self.rank == 0:
            print(f"[DataValveSuperBatchSampler]             ")
            print(f"      : {self.total_samples:,}")
            print(f"  World Size: {self.world_size}")
            print(f"    Rank    : {self.num_samples:,}")
            print(f"        : {target_ratio:.1%}")
            print(f"  Warm-up   : {self.warmup_batches}")
            print(f"  Router    Super-Batch: {self.super_batch_size}")
    
    def _compute_stage_params(self):
        """               num_samples """
        #            rank      
        self.target_selected = int(self.num_samples * self.target_ratio)
        
        # Warm-up       
        self.warmup_samples = int(self.num_samples * self.warmup_ratio)
        self.warmup_batches = self.warmup_samples // self.final_batch_size
        self.warmup_samples = self.warmup_batches * self.final_batch_size
        
        # Router   
        self.router_selected = self.target_selected - self.warmup_samples
        self.router_samples = self.num_samples - self.warmup_samples
        
        if self.router_selected > 0 and self.router_samples > 0:
            #     = router_selected / router_samples
            router_select_ratio = self.router_selected / self.router_samples
            
            # Super-Batch = final_batch_size /    
            #    8 / (996/6310)   50.68
            #    super_batch     Router forward       < 1% 
            #            8                 
            #                     
            import math
            theoretical_super_batch = self.final_batch_size / router_select_ratio
            self.super_batch_size = int(theoretical_super_batch)  #     
            
            #    super_batch_size     final_batch_size      Router     
            self.super_batch_size = max(self.super_batch_size, self.final_batch_size + 1)
            
            #      router_samples      
            self.router_batches = (self.router_samples + self.super_batch_size - 1) // self.super_batch_size
            
            #         GPU 
            actual_router_selected_per_gpu = self.router_batches * self.final_batch_size
            actual_router_select_ratio = self.final_batch_size / self.super_batch_size
            
            #    rank 0       
            if self.rank == 0:
                print(f"[SuperBatchSampler] Router      Rank {self.rank} :")
                print(f"    Rank     : {self.router_samples:,},     : {self.router_selected:,}")
                print(f"       : {router_select_ratio:.4f} ({router_select_ratio*100:.2f}%)")
                print(f"     Super-Batch: {theoretical_super_batch:.2f}")
                print(f"  Super-Batch Size: {self.super_batch_size} (    8    )")
                print(f"       : {actual_router_select_ratio:.4f} ({actual_router_select_ratio*100:.2f}%)")
                print(f"  Router Batches: {self.router_batches:,}")
                print(f"        : {actual_router_selected_per_gpu:,}")
        else:
            self.router_batches = 0
            self.super_batch_size = int(self.final_batch_size / self.target_ratio)  #     
            #         8    
        
        self.total_batches = self.warmup_batches + self.router_batches
    
    def __iter__(self):
        #      rank                  
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        #   LLaVA     group_by_modality_length     
        if self.modality_lengths is not None and self.shuffle:
            #             
            #      1)                2)           
            all_indices = get_modality_length_grouped_indices_for_superbatch(
                lengths=self.modality_lengths,
                batch_size=self.final_batch_size,
                world_size=self.world_size,
                generator=g,
            )
            if self.rank == 0:
                print(f"[Sampler]      group_by_modality_length   ")
        elif self.shuffle:
            #          
            all_indices = torch.randperm(self.total_samples, generator=g).tolist()
            if self.rank == 0:
                print(f"[Sampler]                  ")
        else:
            all_indices = list(range(self.total_samples))
        
        #     bug:      Rank          1   batch 
        #    Rank          NCCL     
        samples_per_rank = self.total_samples // self.world_size  #     
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank  #    Rank            
        
        #    rank      
        indices = all_indices[start_idx:end_idx]
        
        if self.rank == 0:
            discarded = self.total_samples - samples_per_rank * self.world_size
            print(f"[Sampler Rank {self.rank}]   : [{start_idx:,}, {end_idx:,}),   {len(indices):,}   ")
            if discarded > 0:
                print(f"[Sampler]       {discarded}            Rank batch    ")
        
        current_idx = 0
        batch_count = 0
        
        # Warm-up       Batch
        while batch_count < self.warmup_batches and current_idx + self.final_batch_size <= len(indices):
            batch = indices[current_idx:current_idx + self.final_batch_size]
            current_idx += self.final_batch_size
            batch_count += 1
            if batch_count == 1 and self.rank == 0:
                print(f"[Sampler] Warmup Batch 1: size={len(batch)}, final_batch_size={self.final_batch_size}")
            yield batch

        # Router    Super-Batch
        router_batch_count = 0
        while router_batch_count < self.router_batches and current_idx + self.super_batch_size <= len(indices):
            batch = indices[current_idx:current_idx + self.super_batch_size]
            current_idx += self.super_batch_size
            router_batch_count += 1
            if router_batch_count == 1 and self.rank == 0:
                print(f"[Sampler] Router Batch 1: size={len(batch)}, super_batch_size={self.super_batch_size}")
            yield batch
        
        #                 drop_last 
        if not self.drop_last and current_idx < len(indices):
            remaining = indices[current_idx:]
            if len(remaining) >= self.final_batch_size:
                yield remaining
    
    def __len__(self):
        return self.total_batches


class DataValveValveCollator:
    """
        Valve Collator
    
       GRPOValveCollator   
    
    [2026-01-08 FIX]         warmup              
    """
    
    def __init__(
        self,
        valve_trainer: DataValveValveTrainer,  #               
        dataset,
        final_batch_size: int,
        target_ratio: float,
        warmup_ratio: float,
        original_collate_fn: Callable,
        rank: int = None,
        world_size: int = None,
    ):
        """
        Args:
            valve_trainer: DataValveValveTrainer                
            dataset:      
            final_batch_size:    Batch   
            target_ratio:       
            warmup_ratio: Warm-up   
            original_collate_fn:    collate   
            rank:       rank       
            world_size:            
        """
        # self.valve_trainer = valve_trainer  #            
        self.dataset = dataset
        self.final_batch_size = final_batch_size
        self.target_ratio = target_ratio
        self.warmup_ratio = warmup_ratio
        self.original_collate_fn = original_collate_fn

        #          
        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if world_size is None and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1

        #   Sampler       2161 batch        Router   
        global_total_samples = len(dataset)
        #    rank       
        num_samples_per_rank = global_total_samples // self.world_size
        
        #            
        target_samples = int(num_samples_per_rank * target_ratio)
        warmup_samples = int(num_samples_per_rank * warmup_ratio)
        self.warmup_batches_threshold = warmup_samples // final_batch_size

        # Router     
        router_target_samples = target_samples - warmup_samples
        router_candidate_samples = num_samples_per_rank - warmup_samples
        router_select_ratio = router_target_samples / router_candidate_samples if router_candidate_samples > 0 else 0.2
        self.super_batch_size = int(final_batch_size / router_select_ratio) if router_select_ratio > 0 else 45  #     
        self.super_batch_size = max(self.super_batch_size, final_batch_size + 1)

        #   
        self.total_batches = 0
        self.warmup_batches = 0
        self.router_batches = 0

        #    rank 0       
        if self.rank == 0:
            print(f"[DataValveValveCollator]             ")
            print(f"      : {global_total_samples:,}, World Size: {self.world_size}")
            print(f"    Rank   : {num_samples_per_rank:,},     : {target_samples:,} ({target_ratio:.0%})")
            print(f"  Warmup: {warmup_samples:,}    , {self.warmup_batches_threshold}   batch")
            print(f"  Router:    {router_target_samples:,}  ,    {router_candidate_samples:,}  ")
            print(f"  Router    : {router_select_ratio:.2%}, Super-Batch: {self.super_batch_size}")

    @staticmethod
    def _is_lazy_ref(sample: Any) -> bool:
        return isinstance(sample, dict) and bool(sample.get("__datavalve_lazy_ref__", False))

    def _materialize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_lazy_ref(sample):
            return self.dataset[int(sample["__dataset_index__"])]
        return sample
    
    def __call__(self, batch_candidates: List[Dict]) -> Dict[str, torch.Tensor]:
        """
           Batch
        
              DataLoader    num_workers > 0 Collator   worker       
        Router     mask, soft_scores           batch dict        
               compute_loss         
        
            batch dict         
        - router_mask: Router     mask (Tensor)
        - router_soft_scores: Router   soft scores (Tensor)
        - router_clip_distance: CLIP    (Tensor)
        - router_cluster_ids:    ID (Tensor)
        - router_is_warmup:      warmup    (bool)
        """
        self.total_batches += 1
        batch_size = len(batch_candidates)

        is_warmup = self.total_batches <= self.warmup_batches_threshold
        
        if self.total_batches % 10 == 1 or self.total_batches <= 5:
            import os
            print(f"[Collator PID={os.getpid()}] batch={self.total_batches}/{self.warmup_batches_threshold}, "
                  f"is_warmup={is_warmup}, batch_size={batch_size}, "
                  f"super_batch_size={self.super_batch_size}, final_batch_size={self.final_batch_size}")

        if is_warmup or batch_size <= self.final_batch_size:
            # Warm-up        
            self.warmup_batches += 1
            final_samples = [self._materialize_sample(s) for s in batch_candidates[:self.final_batch_size]]

            #             
            indices = [s.get('unique_idx', i) for i, s in enumerate(final_samples)]

            #       worker                 
            # self.valve_trainer.log_batch_info(len(batch_candidates), len(final_samples))
            # self.valve_trainer.step()

            #      batch
            batch = self.original_collate_fn(final_samples)

            #     warmup          
            batch['router_is_warmup'] = True
            batch['router_mask'] = None
            batch['router_selected_indices'] = indices  #        
            batch['router_num_candidates'] = len(batch_candidates)  #       

            return batch
        
        # Router                  Router       
        self.router_batches += 1

        all_candidate_samples = batch_candidates[:self.super_batch_size]
        candidate_indices = [s.get('unique_idx', i) for i, s in enumerate(all_candidate_samples)]
        candidate_sources = [str(s.get('source', 'unknown')) for s in all_candidate_samples]
        candidate_dataset_indices = [
            int(s["__dataset_index__"])
            for s in all_candidate_samples
            if self._is_lazy_ref(s)
        ]

        batch = {
            'router_is_warmup': False,
            'router_sources': candidate_sources,
            'router_has_all_candidates': True,
            'router_final_batch_size': self.final_batch_size,
            'router_num_candidates': len(all_candidate_samples),
        }
        if len(candidate_dataset_indices) == len(all_candidate_samples):
            batch['router_candidate_dataset_indices'] = candidate_dataset_indices
        else:
            batch['router_candidate_samples'] = all_candidate_samples
        try:
            batch['router_candidate_indices'] = torch.tensor(
                [int(idx) for idx in candidate_indices], dtype=torch.long
            )
        except (ValueError, TypeError):
            batch['router_candidate_indices'] = candidate_indices

        return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """      """
        return {
            'total_batches': self.total_batches,
            'warmup_batches': self.warmup_batches,
            'router_batches': self.router_batches,
            'final_batch_size': self.final_batch_size,
            'target_ratio': self.target_ratio,
        }


#      LLaVATrainer
try:
    from llava.train.train import LLaVATrainer
except ImportError:
    LLaVATrainer = Trainer


class LLaVATrainer_DataValve(LLaVATrainer):
    """
          Router   LLaVA Trainer

         
    1.            
    2.    Router       +     
    3.    Golden Set   
    4. Router       DDP            
    """

    def __init__(
        self,
        valve_trainer: DataValveValveTrainer,
        golden_dataloader: Optional[DataLoader] = None,
        golden_shard_dataloaders: Optional[List[DataLoader]] = None,
        router_lr: float = 1e-4,
        router_lr_min: float = 5e-5,
        router_update_interval: int = 4,
        router_optimizer_accum_steps: int = 1,
        original_collate_fn: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            valve_trainer: DataValveValveTrainer   
            golden_dataloader: Golden Set DataLoader
            router_lr: Router    
            router_update_interval: Router     
        """
        super().__init__(*args, **kwargs)

        self.valve_trainer = valve_trainer
        self.golden_dataloader = golden_dataloader
        self.golden_shard_dataloaders = golden_shard_dataloaders or []
        self.original_collate_fn = original_collate_fn
        self.router_lr = router_lr
        self.router_lr_min = min(max(float(router_lr_min), 0.0), float(router_lr))
        self.router_update_interval = max(1, router_update_interval)
        self.router_optimizer_accum_steps = max(1, int(router_optimizer_accum_steps))
        self.debug_cuda_sync = os.environ.get("DATAVALVE_DEBUG_CUDA_SYNC", "0") == "1"

        #                         
        self.golden_iterator = None
        if golden_dataloader is not None:
            self.golden_iterator = self._create_infinite_iterator(golden_dataloader)
            print(f"[LLaVATrainer_DataValve] Golden Set         ")

        # Router         4 
        self.router_optimizer = None
        self.router_scheduler = None
        self.router_scheduler_total_steps = None

        #   
        self.router_step_count = 0
        self.router_optimizer_step_count = 0
        self.router_micro_step_count = 0
        self.router_accum_step_count = 0
        self.mask_none_skip_count = 0
        self.candidate_mismatch_skip_count = 0
        self.router_update_norm_ema = 0.0
        self.router_update_ema_decay = 0.9
        self.update_limited_threshold = 0.05
        self.update_limited_streak = 0
        self.golden_shard_cursor = 0
        self.learnability_gap_ema = 0.0
        self.learnability_ema_decay = 0.9
        self.router_total_update_steps = None

        #   1   maxlen=1                router_update_interval   
        queue_maxlen_env = os.environ.get("DATAVALVE_ACTION_QUEUE_MAXLEN", "1")
        try:
            queue_maxlen = int(queue_maxlen_env)
        except Exception:
            queue_maxlen = 1
        if queue_maxlen <= 0:
            queue_maxlen = 1
        self.action_queue_maxlen = queue_maxlen
        self.router_action_queue = deque(maxlen=self.action_queue_maxlen)
        
        #        router loss_dict    callback    
        self._current_router_loss_dict = {}

        print(f"[LLaVATrainer_DataValve]      ")
        print(f"  Router    : {router_lr}")
        print(f"  Router      : {self.router_lr_min}")
        print(f"  Router     : {self.router_update_interval}")
        print(f"  Router  -shard     : {self.router_optimizer_accum_steps}")
        print(f"          : {self.action_queue_maxlen}")
        print(f"    CUDA  : {self.debug_cuda_sync}")

        # =======================================================
        # [Device Sync]    Router   LLaVA        
        # =======================================================
        #   DeepSpeed          rank      GPU
        # Router            local_rank    
        #       "Expected all tensors to be on the same device"   
        # =======================================================
        if hasattr(self.args, 'device') and self.args.device is not None:
            target_device = self.args.device
            self.valve_trainer.device = target_device
            # DDP      Router        to()      DDP     
            if not hasattr(self.valve_trainer.router, 'module'):
                self.valve_trainer.router.to(target_device)
            print(f"[LLaVATrainer_DataValve]    Router      : {target_device}")

    def _create_infinite_iterator(self, dataloader):
        """
                 
        
             
        -    epoch          
        -    shuffle=True    epoch       
        -    Router     Golden Set    
        """
        def infinite_loader():
            while True:
                for batch in dataloader:
                    yield batch
        return iter(infinite_loader())

    def _estimate_router_update_counts(self) -> tuple[int, int]:
        """
           Router reward update       optimizer step    
        """
        grad_accum = max(1, int(getattr(self.args, 'gradient_accumulation_steps', 1)))
        total_micro_steps = max(1, int(self.state.max_steps) * grad_accum)
        warmup_micro_steps = int(total_micro_steps * float(getattr(self.valve_trainer, 'warmup_ratio', 0.0)))
        router_micro_steps = max(1, total_micro_steps - warmup_micro_steps)
        total_router_updates = max(
            1,
            (router_micro_steps + self.router_update_interval - 1) // self.router_update_interval,
        )
        total_router_optimizer_steps = max(
            1,
            (total_router_updates + self.router_optimizer_accum_steps - 1)
            // self.router_optimizer_accum_steps,
        )
        return total_router_updates, total_router_optimizer_steps

    def create_optimizer(self):
        """
                4 

        -         LLaVA            datavalve_router 
        - Router      optimizer DDP        
        """
        from transformers.trainer_pt_utils import get_parameter_names
        try:
            from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        except ImportError:
            ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]

        opt_model = self.model

        if self.optimizer is None:
            #            
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            #     
            lora_parameters = []
            mm_projector_parameters = []
            other_llava_parameters = []

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_" in name:
                    lora_parameters.append(name)
                elif "mm_projector" in name:
                    mm_projector_parameters.append(name)
                elif "datavalve_router" in name:
                    #   4     Router         
                    continue
                else:
                    other_llava_parameters.append(name)

            #      
            optimizer_grouped_parameters = []

            # 1. LoRA   
            if lora_parameters:
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in lora_parameters and n in decay_parameters and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in lora_parameters and n not in decay_parameters and p.requires_grad],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ])
                print(f"[Optimizer] LoRA   : {len(lora_parameters)}  , lr={self.args.learning_rate}")

            # 2. MM Projector   
            if mm_projector_parameters:
                mm_lr = self.args.mm_projector_lr if self.args.mm_projector_lr is not None else 2e-5
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in mm_projector_parameters and n in decay_parameters and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                        "lr": mm_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in mm_projector_parameters and n not in decay_parameters and p.requires_grad],
                        "weight_decay": 0.0,
                        "lr": mm_lr,
                    },
                ])
                print(f"[Optimizer] MM Projector   : {len(mm_projector_parameters)}  , lr={mm_lr}")

            # 3.    LLaVA   
            if other_llava_parameters:
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in other_llava_parameters and n in decay_parameters and p.requires_grad],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters()
                                   if n in other_llava_parameters and n not in decay_parameters and p.requires_grad],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ])
                print(f"[Optimizer]    LLaVA   : {len(other_llava_parameters)}  , lr={self.args.learning_rate}")

            #            Router 
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            print(f"[Optimizer] LLaVA        : {optimizer_cls.__name__}")

            #      Router    
            router_params = [p for p in self.valve_trainer.router.parameters() if p.requires_grad]
            if router_params:
                self.router_optimizer = torch.optim.AdamW(
                    router_params,
                    lr=self.router_lr,
                    weight_decay=0.0,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )
                self.router_scheduler = None
                print(f"[Optimizer] Router          : AdamW (lr={self.router_lr})")
            else:
                self.router_optimizer = None
                self.router_scheduler = None
                print("[Optimizer]   : Router                  ")

            #         Router                zero_grad/step    
            main_opt_param_ids = {
                id(p)
                for group in self.optimizer.param_groups
                for p in group.get("params", [])
            }
            router_opt_param_ids = set()
            if self.router_optimizer is not None:
                router_opt_param_ids = {
                    id(p)
                    for group in self.router_optimizer.param_groups
                    for p in group.get("params", [])
                }

            overlap_ids = main_opt_param_ids & router_opt_param_ids
            self.router_main_optimizer_overlap_count = len(overlap_ids)

            if overlap_ids:
                id_to_name = {id(p): n for n, p in opt_model.named_parameters()}
                overlap_names = [id_to_name.get(pid, f"<unknown:{pid}>") for pid in overlap_ids]
                sample_names = overlap_names[:10]
                raise RuntimeError(
                    "[Optimizer][FATAL] Router              router_optimizer "
                    f"     ={len(overlap_ids)}   : {sample_names}"
                )
            else:
                print("[Optimizer]         : Router          ")

        return self.optimizer

    def _get_train_sampler(self):
        """      """
        if hasattr(self.valve_trainer, 'grpo_sampler') and self.valve_trainer.grpo_sampler is not None:
            return None
        return super()._get_train_sampler()

    def get_train_dataloader(self):
        if hasattr(self.valve_trainer, 'grpo_sampler') and self.valve_trainer.grpo_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.valve_trainer.grpo_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
                prefetch_factor=2 if self.args.dataloader_num_workers > 0 else None,
            )
        return super().get_train_dataloader()

    def compute_per_sample_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """           """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size = shift_logits.shape[0]
        seq_len = shift_logits.shape[1]
        vocab_size = shift_logits.shape[-1]
        
        #        
        if shift_labels.shape[1] != seq_len:
            if shift_labels.shape[1] > seq_len:
                shift_labels = shift_labels[:, :seq_len]
            else:
                pad_len = seq_len - shift_labels.shape[1]
                shift_labels = torch.cat([
                    shift_labels,
                    torch.full((batch_size, pad_len), IGNORE_TOKEN_ID,
                              dtype=shift_labels.dtype, device=shift_labels.device)
                ], dim=1)
        
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_TOKEN_ID)
        # .reshape()            
        per_token_loss = loss_fct(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1)
        ).reshape(batch_size, -1)
        
        valid_mask = (shift_labels != IGNORE_TOKEN_ID).float()
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_counts
        
        return per_sample_loss

    def map_source_to_group(self, source: str) -> Optional[str]:
        if source is None:
            return None
        return self.valve_trainer.source_to_group.get(str(source))

    def aggregate_group_losses(
        self,
        per_sample_loss: torch.Tensor,
        sources: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
        group_loss_sums = {group_name: 0.0 for group_name in self.valve_trainer.group_names}
        group_count_dict = {group_name: 0 for group_name in self.valve_trainer.group_names}
        source_count_dict: Dict[str, int] = {}

        sample_losses = per_sample_loss.detach().float().cpu().tolist()
        for sample_loss, source in zip(sample_losses, sources):
            source_key = str(source)
            source_count_dict[source_key] = source_count_dict.get(source_key, 0) + 1
            group_name = self.map_source_to_group(source_key)
            if group_name is None:
                continue
            group_loss_sums[group_name] += float(sample_loss)
            group_count_dict[group_name] += 1

        group_loss_dict = {}
        for group_name in self.valve_trainer.group_names:
            count = group_count_dict[group_name]
            group_loss_dict[group_name] = group_loss_sums[group_name] / count if count > 0 else 0.0

        return group_loss_dict, group_count_dict, source_count_dict

    def compute_golden_loss(self, retain_graph: bool = True, golden_dataloader: Optional[DataLoader] = None, shard_id: int = -1) -> Dict[str, Any]:
        """
           Golden Set                  
        
             2025-12  
        =====================================================
            BUG shuffle=False +      =      20    
        
           
        1. Router             
        2.     20    "    " Router         OCR/    
        3.        Router        
        
           
        1. shuffle=True    epoch       
        2. infinite_iterator             
        3.      4   batch (  64  )    Golden Set   
        =====================================================
        
        REINFORCE      
        - L_val    Reward            LLaVA 
        - advantage = L_val - EMA(L_val)    Policy Gradient
        - Router    Golden Set       
        """
        if self.golden_iterator is None and golden_dataloader is None:
            return {
                'val_loss': torch.tensor(0.0, device=self.args.device, requires_grad=False),
                'group_loss_dict': {group_name: 0.0 for group_name in self.valve_trainer.group_names},
                'group_count_dict': {group_name: 0 for group_name in self.valve_trainer.group_names},
                'source_count_dict': {},
                'shard_id': shard_id,
            }

        original_training = self.model.training
        self.model.eval()  #         Dropout   BatchNorm   training   

        try:
            active_golden_dataloader = golden_dataloader if golden_dataloader is not None else self.golden_dataloader
            total_loss = 0.0
            group_loss_sums_local = {group_name: 0.0 for group_name in self.valve_trainer.group_names}
            group_count_local = {group_name: 0 for group_name in self.valve_trainer.group_names}
            source_count_local: Dict[str, int] = {}

            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if golden_dataloader is None:
                num_batches_total = min(4, len(active_golden_dataloader))
                start_batch = 0
                end_batch = num_batches_total
                golden_iter = self.golden_iterator
            else:
                num_batches_total = len(active_golden_dataloader)
                start_batch = (num_batches_total * rank) // world_size
                end_batch = (num_batches_total * (rank + 1)) // world_size
                golden_iter = iter(active_golden_dataloader)
                for _ in range(start_batch):
                    try:
                        next(golden_iter)
                    except StopIteration:
                        break

            num_batches_this_rank = max(0, end_batch - start_batch)

            with torch.no_grad():
                total_loss_tensor = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
                valid_batches_this_rank = 0.0
                nonfinite_batches_this_rank = 0.0
                for local_batch_idx in range(num_batches_this_rank):
                    batch_idx = start_batch + local_batch_idx
                    try:
                        batch = next(golden_iter)
                    except StopIteration:
                        break
                    model_inputs = {}
                    batch_sources = batch.get('sources', None)
                    for k, v in batch.items():
                        if k.startswith('router_') or k in ['unique_indices', 'unique_idx', 'sources']:
                            continue
                        if isinstance(v, torch.Tensor):
                            v = v.to(self.args.device)
                            if k == 'images' and self.args.bf16:
                                v = v.to(dtype=torch.bfloat16)
                            model_inputs[k] = v
                        else:
                            model_inputs[k] = v

                    outputs = self.model(**model_inputs)
                    logits = outputs.logits
                    labels = model_inputs.get('labels')
                    if labels is not None:
                        per_sample_loss = self.compute_per_sample_loss(logits, labels).detach().float()
                        batch_loss = per_sample_loss.mean()
                    else:
                        batch_loss = outputs.loss.detach().float()
                        per_sample_loss = None
                    if not torch.isfinite(batch_loss):
                        nonfinite_batches_this_rank += 1.0
                        print(
                            f"[Warning] step={self.state.global_step} Golden loss     "
                            f"rank={rank}, batch_idx={batch_idx}, loss={batch_loss.item()}     batch"
                        )
                    else:
                        total_loss_tensor += batch_loss
                        valid_batches_this_rank += 1.0
                        if per_sample_loss is not None and batch_sources is not None:
                            batch_group_loss_dict, batch_group_count_dict, batch_source_count_dict = self.aggregate_group_losses(
                                per_sample_loss,
                                list(batch_sources),
                            )
                            for group_name in self.valve_trainer.group_names:
                                batch_count = batch_group_count_dict.get(group_name, 0)
                                if batch_count > 0:
                                    group_loss_sums_local[group_name] += batch_group_loss_dict[group_name] * batch_count
                                    group_count_local[group_name] += batch_count
                            for source_name, source_count in batch_source_count_dict.items():
                                source_count_local[source_name] = source_count_local.get(source_name, 0) + source_count
                    del outputs, model_inputs

                total_loss = total_loss_tensor.item()

            if dist.is_initialized():
                loss_tensor = torch.tensor(
                    [total_loss, valid_batches_this_rank, nonfinite_batches_this_rank],
                    device=self.args.device,
                    dtype=torch.float32,
                )
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = loss_tensor[0].item()
                total_batches_run = loss_tensor[1].item()
                total_nonfinite_batches = loss_tensor[2].item()

                group_tensor_values = []
                for group_name in self.valve_trainer.group_names:
                    group_tensor_values.extend([
                        float(group_loss_sums_local[group_name]),
                        float(group_count_local[group_name]),
                    ])
                group_tensor = torch.tensor(group_tensor_values, device=self.args.device, dtype=torch.float32)
                dist.all_reduce(group_tensor, op=dist.ReduceOp.SUM)
                restored_group_sums = {}
                restored_group_counts = {}
                idx = 0
                for group_name in self.valve_trainer.group_names:
                    restored_group_sums[group_name] = group_tensor[idx].item()
                    restored_group_counts[group_name] = int(group_tensor[idx + 1].item())
                    idx += 2
                group_loss_sums_local = restored_group_sums
                group_count_local = restored_group_counts

                gathered_source_counts = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_source_counts, source_count_local)
                merged_source_counts: Dict[str, int] = {}
                for source_dict in gathered_source_counts:
                    if not source_dict:
                        continue
                    for source_name, source_count in source_dict.items():
                        merged_source_counts[source_name] = merged_source_counts.get(source_name, 0) + int(source_count)
                source_count_local = merged_source_counts
            else:
                total_batches_run = valid_batches_this_rank
                total_nonfinite_batches = nonfinite_batches_this_rank

            if total_batches_run == 0:
                print(f"[Warning] step={self.state.global_step} Golden loss       fallback    0.0")
                return {
                    'val_loss': torch.tensor(0.0, device=self.args.device, requires_grad=False),
                    'group_loss_dict': {group_name: 0.0 for group_name in self.valve_trainer.group_names},
                    'group_count_dict': {group_name: 0 for group_name in self.valve_trainer.group_names},
                    'source_count_dict': source_count_local,
                    'shard_id': shard_id,
                }

            if total_nonfinite_batches > 0 and rank == 0:
                print(
                    f"[Warning] step={self.state.global_step} Golden loss        batch  : "
                    f"{int(total_nonfinite_batches)}"
                )

            val_loss_value = total_loss / total_batches_run

            group_loss_dict = {}
            for group_name in self.valve_trainer.group_names:
                count = group_count_local.get(group_name, 0)
                total_group_loss = group_loss_sums_local.get(group_name, 0.0)
                group_loss_dict[group_name] = (total_group_loss / count) if count > 0 else 0.0
            return {
                'val_loss': torch.tensor(val_loss_value, device=self.args.device, requires_grad=False),
                'group_loss_dict': group_loss_dict,
                'group_count_dict': dict(group_count_local),
                'source_count_dict': dict(source_count_local),
                'shard_id': shard_id,
            }
        finally:
            if original_training:
                self.model.train()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
            

        REINFORCE        
        =====================================================
           LLaVA     Router        
             L_llava = CrossEntropy(f_ (x_selected), y_selected)

             L_reinforce:    Policy Gradient    Golden Set   
               L_reinforce = (L_val - EMA(L_val)) *  (-log  (g_i))

         -    Router    L_reinforce    Golden Set      
         - Router      "    "            

                     
        - DataLoader    num_workers > 0   Collator   worker      
        - Router    mask, soft_scores      batch dict       
        -   inputs     router_*         valve_trainer   

           Router                  optimizer
        =====================================================
        """
        unique_indices = inputs.pop("unique_indices", None)
        #       sources            dataset-level     
        #           selected/unselected          
        batch_sources = inputs.pop("sources", None)
        
        # =======================================================
        # =======================================================
        #    soft_scores     Worker             
        #              Re-forward Router   
        # =======================================================
        router_is_warmup = inputs.pop("router_is_warmup", True)
        router_mask = inputs.pop("router_mask", None)
        router_candidate_indices = inputs.pop("router_candidate_indices", None)  #      ID
        router_selected_indices = inputs.pop("router_selected_indices", None)  #   Warmup      
        router_num_candidates = inputs.pop("router_num_candidates", None)  #       
        router_clip_distance = inputs.pop("router_clip_distance", None)
        router_cluster_ids = inputs.pop("router_cluster_ids", None)
        router_candidate_samples = inputs.pop("router_candidate_samples", None)
        router_candidate_dataset_indices = inputs.pop("router_candidate_dataset_indices", None)
        router_sources_meta = inputs.pop("router_sources", None)
        batch_grad_norms = inputs.pop("grad_norms", None)

        #           
        inputs.pop("router_soft_scores", None)
        
        
        if self.state.max_steps > 0:
            grad_accum_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
            warmup_batches = getattr(self.valve_trainer, 'warmup_batches', None)
            
            if warmup_batches and warmup_batches > 0:
                warmup_steps = (warmup_batches + grad_accum_steps - 1) // grad_accum_steps
                is_warmup_by_step = self.state.global_step < warmup_steps
         
        #                Router                  
        router_has_all_candidates_debug = inputs.get("router_has_all_candidates", False)

        #             Warmup    
        if router_is_warmup and router_selected_indices is not None:
            if not isinstance(inputs.get("input_ids", None), torch.Tensor) or getattr(inputs["input_ids"], "device", torch.device("cpu")).type == "cpu":
                inputs = self._prepare_inputs(inputs)
            self.valve_trainer.current_selected_indices = router_selected_indices
            num_candidates = router_num_candidates if router_num_candidates is not None else len(router_selected_indices)
            #      step()         
            self.valve_trainer.log_batch_info(num_candidates, len(router_selected_indices))
            self.valve_trainer.step()

        router_mask_for_training = None
        router_candidate_indices_for_training = None
        selected_indices_for_training = []
        selection_topk_margin = None
        selected_score_mean = None
        unselected_score_mean = None
        selected_unselected_score_gap = None
        router_selection_time = None
        if not router_is_warmup and router_candidate_indices is not None:
            router_has_all_candidates = inputs.pop("router_has_all_candidates", False)
            router_final_batch_size = inputs.pop("router_final_batch_size", self.args.per_device_train_batch_size)

            if router_has_all_candidates and (router_candidate_samples is not None or router_candidate_dataset_indices is not None):
                t0_router_selection = time.time()
                if isinstance(router_candidate_indices, torch.Tensor):
                    indices_list = router_candidate_indices.tolist()
                else:
                    indices_list = list(router_candidate_indices)
                router_candidate_indices_for_training = list(indices_list)
                if isinstance(router_candidate_dataset_indices, torch.Tensor):
                    candidate_dataset_pos_list = [int(x) for x in router_candidate_dataset_indices.tolist()]
                elif router_candidate_dataset_indices is not None:
                    candidate_dataset_pos_list = [int(x) for x in router_candidate_dataset_indices]
                else:
                    candidate_dataset_pos_list = None

                try:
                    img_feats, txt_feats, _ = self.valve_trainer.get_batch_features(indices_list)

                    router_device = next(self.valve_trainer.router.parameters()).device
                    router_dtype = next(self.valve_trainer.router.parameters()).dtype
                    img_feats = img_feats.detach().to(device=router_device, dtype=router_dtype)
                    txt_feats = txt_feats.detach().to(device=router_device, dtype=router_dtype)

                    with torch.no_grad():
                        logits = self.valve_trainer.router(img_feats, txt_feats)
                        router_soft_scores = torch.sigmoid(logits)
                        if not torch.isfinite(router_soft_scores).all():
                            print(f"[Warning] step={self.state.global_step}        router_soft_scores selection      0.5")
                            router_soft_scores = torch.nan_to_num(router_soft_scores, nan=0.5, posinf=1.0, neginf=0.0)

                    selected_scores_debug = router_soft_scores.detach().float()
                    selected_scores_debug_cpu = selected_scores_debug.cpu()
                    if selected_scores_debug.numel() > 0:
                        selected_prob_lt_005 = (selected_scores_debug < 0.05).float().mean().item()
                        selected_prob_gt_095 = (selected_scores_debug > 0.95).float().mean().item()
                    else:
                        selected_prob_lt_005 = 0.0
                        selected_prob_gt_095 = 0.0
                    selected_logits_debug = logits.detach().float()

                    k = min(router_final_batch_size, len(indices_list))
                    topk_scores, topk_indices = torch.topk(router_soft_scores, k, dim=0)
                    router_mask_for_training = torch.zeros(len(indices_list), device=router_device)
                    router_mask_for_training.scatter_(0, topk_indices, 1.0)

                    selected_mask_cpu = router_mask_for_training.detach().cpu().bool().tolist()
                    selected_indices_for_training = [idx for idx, keep in zip(indices_list, selected_mask_cpu) if keep]
                    selected_mask_debug_cpu = torch.tensor(selected_mask_cpu, dtype=torch.bool)
                    if selected_scores_debug_cpu.numel() > 0 and 0 < k < selected_scores_debug_cpu.numel():
                        kth_score = float(topk_scores.detach().float().min().item())
                        unselected_scores_debug = selected_scores_debug_cpu[~selected_mask_debug_cpu]
                        next_score = float(unselected_scores_debug.max().item()) if unselected_scores_debug.numel() > 0 else kth_score
                        selection_topk_margin = kth_score - next_score
                    elif selected_scores_debug_cpu.numel() > 0:
                        selection_topk_margin = 0.0
                    selected_mask_for_score = router_mask_for_training.detach().bool()
                    if selected_mask_for_score.any():
                        selected_score_mean = float(router_soft_scores.detach().float()[selected_mask_for_score].mean().item())
                    if (~selected_mask_for_score).any():
                        unselected_score_mean = float(router_soft_scores.detach().float()[~selected_mask_for_score].mean().item())
                    if selected_score_mean is not None and unselected_score_mean is not None:
                        selected_unselected_score_gap = selected_score_mean - unselected_score_mean

                    if candidate_dataset_pos_list is not None and len(candidate_dataset_pos_list) == len(selected_mask_cpu):
                        selected_samples = [
                            self.train_dataset[int(dataset_pos)]
                            for dataset_pos, keep in zip(candidate_dataset_pos_list, selected_mask_cpu)
                            if keep
                        ]
                    else:
                        selected_samples = [sample for sample, keep in zip(router_candidate_samples, selected_mask_cpu) if keep]
                    inputs = self.original_collate_fn(selected_samples)
                    inputs.pop("unique_indices", None)
                    inputs.pop("sources", None)
                    inputs = self._prepare_inputs(inputs)
                    router_mask_for_training = router_mask_for_training.to(inputs['labels'].device)

                    selected_dataset_names = []
                    unselected_dataset_names = []
                    try:
                        source_list = (
                            router_sources_meta
                            if isinstance(router_sources_meta, list)
                            else [str(s.get('source', 'unknown')) for s in router_candidate_samples]
                        )
                        if len(source_list) == len(indices_list):
                            selected_dataset_names = [str(src) for src, keep in zip(source_list, selected_mask_cpu) if keep]
                            unselected_dataset_names = [str(src) for src, keep in zip(source_list, selected_mask_cpu) if not keep]
                    except Exception:
                        selected_dataset_names = []
                        unselected_dataset_names = []

                    selected_count = router_mask_for_training.sum().item()

                except Exception as e:
                    print(f"[Warning] Router forward for selection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    router_mask_for_training = None
                    if candidate_dataset_pos_list is not None:
                        fallback_samples = [
                            self.train_dataset[int(dataset_pos)]
                            for dataset_pos in candidate_dataset_pos_list[:router_final_batch_size]
                        ]
                    else:
                        fallback_samples = router_candidate_samples[:router_final_batch_size]
                    inputs = self.original_collate_fn(fallback_samples)
                    inputs.pop("unique_indices", None)
                    inputs.pop("sources", None)
                    inputs = self._prepare_inputs(inputs)
                finally:
                    router_selection_time = time.time() - t0_router_selection

        # LLaVA forward       warmup batch   Router       
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # LLaVA                
        if labels is not None:
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                llava_loss = outputs.loss
            else:
                # Fallback:                   
                per_sample_loss = self.compute_per_sample_loss(logits, labels)
                llava_loss = per_sample_loss.mean()
        else:
            llava_loss = outputs.loss

        #    [  1]   compute_loss      LLaVA   
        # Router forward         training_step   backward   
        #          DeepSpeed ZeRO-3      
        
        #    Router        training_step   
        self._router_info = {
            'router_is_warmup': router_is_warmup,
            'router_candidate_indices': router_candidate_indices,
            'router_candidate_indices_for_training': router_candidate_indices_for_training,
            'router_selected_indices': router_selected_indices,
            'candidate_indices': (
                router_candidate_indices_for_training
                if router_candidate_indices_for_training is not None
                else (router_selected_indices if router_is_warmup and router_selected_indices is not None else [])
            ),
            'router_num_candidates': router_num_candidates,
            'router_clip_distance': router_clip_distance,
            'router_cluster_ids': router_cluster_ids,
            'router_mask_for_training': router_mask_for_training,
            'selected_indices': (
                selected_indices_for_training
                if selected_indices_for_training
                else (router_selected_indices if router_is_warmup and router_selected_indices is not None else [])
            ),
            'selection_prob_lt_005': selected_prob_lt_005 if 'selected_prob_lt_005' in locals() else None,
            'selection_prob_gt_095': selected_prob_gt_095 if 'selected_prob_gt_095' in locals() else None,
            'selection_logits_mean': selected_logits_debug.mean().item() if 'selected_logits_debug' in locals() and selected_logits_debug.numel() > 0 else None,
            'selection_logits_std': selected_logits_debug.std(unbiased=False).item() if 'selected_logits_debug' in locals() and selected_logits_debug.numel() > 0 else None,
            'selection_topk_margin': selection_topk_margin,
            'selected_score_mean': selected_score_mean,
            'unselected_score_mean': unselected_score_mean,
            'selected_unselected_score_gap': selected_unselected_score_gap,
            'router_selection_time': router_selection_time,
            'selected_dataset_names': selected_dataset_names if 'selected_dataset_names' in locals() else [],
            'unselected_dataset_names': unselected_dataset_names if 'unselected_dataset_names' in locals() else [],
        }

        #    [  1] compute_loss     LLaVA   
        # Router forward         training_step   backward   
        #          DeepSpeed ZeRO-3      
        
        if return_outputs:
            return (llava_loss, outputs)
        return llava_loss
    
    def compute_router_loss_after_backward(
        self,
        router_candidate_indices,
        router_clip_distance=None,
        router_cluster_ids=None,
        selected_action_stats=None,
    ):
        """
          LLaVA backward      Router   
        
           [  1]      DeepSpeed ZeRO-3        
        - LLaVA backward           
        - Router forward         
        """
        router_is_warmup = self._router_info.get('router_is_warmup', True)
        
        if router_is_warmup or router_candidate_indices is None:
            return None, {}
        
        #     ID    list
        if isinstance(router_candidate_indices, torch.Tensor):
            indices_list = router_candidate_indices.tolist()
        else:
            indices_list = list(router_candidate_indices)
        
        #    final_batch_size          
        final_batch_size = self.args.per_device_train_batch_size
        
        try:
            router_model = self.valve_trainer.router.module if hasattr(self.valve_trainer.router, 'module') else self.valve_trainer.router

            #    CLIP    CPU         DeepSpeed 
            img_feats, txt_feats, cluster_ids = self.valve_trainer.get_batch_features(indices_list)

            # [Defensive Device Move]       Router           
            router_device = next(router_model.parameters()).device
            router_dtype = next(router_model.parameters()).dtype

            #                                 detach
            img_feats = img_feats.detach().to(device=router_device, dtype=router_dtype)
            txt_feats = txt_feats.detach().to(device=router_device, dtype=router_dtype)

            #    A                          RL         
            router_mask_for_training = self._router_info.get('router_mask_for_training', None)
            candidate_indices_for_training = self._router_info.get('router_candidate_indices_for_training', None)

            action_valid = True
            action_skip_reason = None

            if router_mask_for_training is None:
                action_valid = False
                action_skip_reason = 'action_mask_none'
                self.mask_none_skip_count += 1
                action_mask_A = None
            else:
                action_mask_A = router_mask_for_training.detach().to(device=router_device)

            if action_valid:
                if candidate_indices_for_training is None:
                    action_valid = False
                    action_skip_reason = 'candidate_indices_none'
                    self.candidate_mismatch_skip_count += 1
                else:
                    if len(candidate_indices_for_training) != len(indices_list):
                        action_valid = False
                        action_skip_reason = 'candidate_len_mismatch'
                        self.candidate_mismatch_skip_count += 1
                    else:
                        for a, b in zip(candidate_indices_for_training, indices_list):
                            if int(a) != int(b):
                                action_valid = False
                                action_skip_reason = 'candidate_order_mismatch'
                                self.candidate_mismatch_skip_count += 1
                                break

            #              B       
            mask_match_rate = float('nan')
            if action_valid and action_mask_A is not None:
                with torch.no_grad():
                    logits_no_grad = self.valve_trainer.router(img_feats, txt_feats)
                    router_soft_scores_no_grad = torch.sigmoid(logits_no_grad)
                    if not torch.isfinite(router_soft_scores_no_grad).all():
                        print(f"[Warning] step={self.state.global_step}        router_soft_scores monitor      0.5")
                        router_soft_scores_no_grad = torch.nan_to_num(router_soft_scores_no_grad, nan=0.5, posinf=1.0, neginf=0.0)
                k = min(final_batch_size, len(indices_list))
                _, topk_indices = torch.topk(router_soft_scores_no_grad, k, dim=0)
                router_mask_monitor = torch.zeros_like(router_soft_scores_no_grad)
                router_mask_monitor.scatter_(0, topk_indices, 1.0)
                mask_match_rate = (action_mask_A > 0.5).eq(router_mask_monitor > 0.5).float().mean().item()

            #               A    
            if action_valid and action_mask_A is not None and candidate_indices_for_training is not None:
                try:
                    mask_list = action_mask_A.detach().cpu().tolist()
                    selected_indices = [idx for idx, m in zip(candidate_indices_for_training, mask_list) if m > 0.5]
                    self.valve_trainer.current_selected_indices = selected_indices
                    self.valve_trainer.log_batch_info(len(candidate_indices_for_training), len(selected_indices))
                    self.valve_trainer.step()
                except Exception as e:
                    print(f"[Warning]         : {e}")
            
            with torch.no_grad():
                router_clip_distance = router_model.compute_clip_distance(img_feats, txt_feats)
            
            if router_cluster_ids is None:
                router_cluster_ids = cluster_ids.to(device=router_device)
            
            router_cluster_ids = router_cluster_ids.detach()
            
            action_record = {
                'img_feats': img_feats.detach().cpu(),
                'txt_feats': txt_feats.detach().cpu(),
                'mask': action_mask_A.detach().cpu() if action_mask_A is not None else None,
                'candidate_indices': [int(x) for x in candidate_indices_for_training] if candidate_indices_for_training is not None else None,
                'clip_distance': router_clip_distance.detach().cpu() if router_clip_distance is not None else None,
                'cluster_ids': router_cluster_ids.detach().cpu() if router_cluster_ids is not None else None,
                'selected_action_stats': dict(selected_action_stats) if isinstance(selected_action_stats, dict) else None,
                'action_step': int(self.state.global_step),
                'action_valid': bool(action_valid),
                'skip_reason': action_skip_reason,
                'mask_match_rate': mask_match_rate,
            }

            #           Router             
            self.router_micro_step_count += 1
            should_update = (self.router_micro_step_count % self.router_update_interval == 0)

            is_main_rank = True
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    rank = dist.get_rank()
                    is_main_rank = (rank == 0)

                    #     micro_step                       
                    ms = torch.tensor([self.router_micro_step_count], device=router_device, dtype=torch.int64)
                    ms_min = ms.clone()
                    ms_max = ms.clone()
                    dist.all_reduce(ms_min, op=dist.ReduceOp.MIN)
                    dist.all_reduce(ms_max, op=dist.ReduceOp.MAX)
                    if ms_min.item() != ms_max.item():
                        self.router_micro_step_count = int(ms_min.item())
                        should_update = (self.router_micro_step_count % self.router_update_interval == 0)
                        if is_main_rank and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                            print(
                                f"[Router sync] step={self.state.global_step} micro_step   : "
                                f"min={int(ms_min.item())}, max={int(ms_max.item())}      min"
                            )

                    #     should_update         rank     
                    up_flag = torch.tensor([1 if should_update else 0], device=router_device, dtype=torch.int32)
                    up_min = up_flag.clone()
                    up_max = up_flag.clone()
                    dist.all_reduce(up_min, op=dist.ReduceOp.MIN)
                    dist.all_reduce(up_max, op=dist.ReduceOp.MAX)
                    if up_min.item() != up_max.item():
                        should_update = False
                        if is_main_rank and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                            print(
                                f"[Router sync] step={self.state.global_step} should_update    "
                                f"       skip      "
                            )
            except Exception as e:
                print(f"[Warning] step={self.state.global_step} Router micro-step     : {e}")

            if is_main_rank and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                print(f"[Router forward] step={self.state.global_step}, "
                        f"micro_step={self.router_micro_step_count}, "
                      f"update_interval={self.router_update_interval}, "
                      f"should_update={should_update}, "
                      f"router_step_count={self.router_step_count}")
            
            if should_update:
                #   4   LLaVA backward     Golden loss Router            
                if is_main_rank and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                    print(f"[Router update] step={self.state.global_step}    compute_golden_loss()")
                t0_golden = time.time()
                active_golden_dataloader = None
                golden_shard_id = -1
                if self.golden_shard_dataloaders:
                    golden_shard_id = self.golden_shard_cursor % len(self.golden_shard_dataloaders)
                    active_golden_dataloader = self.golden_shard_dataloaders[golden_shard_id]
                    self.golden_shard_cursor = (golden_shard_id + 1) % len(self.golden_shard_dataloaders)
                val_metrics_for_log = self.compute_golden_loss(golden_dataloader=active_golden_dataloader, shard_id=golden_shard_id)  # reward(t)
                golden_eval_time = time.time() - t0_golden
                val_loss_for_log = val_metrics_for_log['val_loss']
                group_loss_dict = val_metrics_for_log.get('group_loss_dict', None)
                group_count_dict = val_metrics_for_log.get('group_count_dict', None)
                reward_shard_id = int(val_metrics_for_log.get('shard_id', -1))
                if is_main_rank and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                    print(f"[Router update] step={self.state.global_step}    compute_golden_loss(), "
                          f"elapsed={time.time() - t0_golden:.2f}s")

                delayed_action = self.router_action_queue.popleft() if len(self.router_action_queue) > 0 else None

                if delayed_action is None:
                    #               Router        reward    
                    if action_record.get('action_valid', False):
                        self.router_action_queue.append(action_record)
                    loss_dict = {
                        'loss_val': val_loss_for_log.detach() if isinstance(val_loss_for_log, torch.Tensor) else torch.tensor(val_loss_for_log),
                        'golden_eval_time': torch.tensor(float(golden_eval_time), device=router_device),
                        'reward_step': torch.tensor(float(self.state.global_step), device=router_device),
                        'action_step': torch.tensor(float(-1), device=router_device),
                        'mask_none_skip_count': torch.tensor(float(self.mask_none_skip_count), device=router_device),
                        'candidate_mismatch_skip_count': torch.tensor(float(self.candidate_mismatch_skip_count), device=router_device),
                    }
                    if self.state.global_step % 10 == 0 or self.state.global_step <= 5:
                        print(f"[Router loss] step={self.state.global_step}, delayed_action=None (cold start), skip update")
                    return None, loss_dict

                #                    
                if (not delayed_action.get('action_valid', False)) or delayed_action.get('mask') is None:
                    if action_record.get('action_valid', False):
                        self.router_action_queue.append(action_record)
                    loss_dict = {
                        'loss_val': val_loss_for_log.detach() if isinstance(val_loss_for_log, torch.Tensor) else torch.tensor(val_loss_for_log),
                        'golden_eval_time': torch.tensor(float(golden_eval_time), device=router_device),
                        'reward_step': torch.tensor(float(self.state.global_step), device=router_device),
                        'action_step': torch.tensor(float(delayed_action.get('action_step', -1)), device=router_device),
                        'mask_none_skip_count': torch.tensor(float(self.mask_none_skip_count), device=router_device),
                        'candidate_mismatch_skip_count': torch.tensor(float(self.candidate_mismatch_skip_count), device=router_device),
                    }
                    if self.state.global_step % 10 == 0 or self.state.global_step <= 5:
                        print(f"[Router loss] step={self.state.global_step}, delayed_action invalid, skip update, reason={delayed_action.get('skip_reason')}")
                    return None, loss_dict

                #                      
                prev_img_feats = delayed_action['img_feats'].to(device=router_device, dtype=router_dtype)
                prev_txt_feats = delayed_action['txt_feats'].to(device=router_device, dtype=router_dtype)
                prev_mask = delayed_action['mask'].to(device=router_device)
                prev_clip_dist = delayed_action['clip_distance'].to(device=router_device) if delayed_action['clip_distance'] is not None else None
                prev_cluster_ids = delayed_action['cluster_ids'].to(device=router_device) if delayed_action['cluster_ids'] is not None else None
                prev_selected_action_stats = delayed_action.get('selected_action_stats') if delayed_action.get('selected_action_stats') is not None else None

                #      forward          
                logits_prev = self.valve_trainer.router(prev_img_feats, prev_txt_feats)
                router_soft_scores_prev = torch.sigmoid(logits_prev)
                if not torch.isfinite(router_soft_scores_prev).all():
                    print(f"[Warning] step={self.state.global_step}        router_soft_scores replay      0.5")
                    router_soft_scores_prev = torch.nan_to_num(router_soft_scores_prev, nan=0.5, posinf=1.0, neginf=0.0)

                #    Router    reward(t)    action(t-k)
                router_loss, loss_dict = self.valve_trainer.compute_router_loss_from_batch(
                    val_loss=val_loss_for_log,
                    mask=prev_mask,
                    soft_scores=router_soft_scores_prev,
                    clip_distance=prev_clip_dist,
                    cluster_ids=prev_cluster_ids,
                    img_features=prev_img_feats,
                    txt_features=prev_txt_feats,
                    group_loss_dict=group_loss_dict,
                    group_count_dict=group_count_dict,
                    shard_id=reward_shard_id,
                    router_progress=(
                        self.router_step_count
                        / max(
                            1,
                            self.router_total_update_steps
                            if self.router_total_update_steps is not None
                            else self.state.max_steps,
                        )
                    ),
                    selected_action_stats=prev_selected_action_stats,
                )
                self.router_step_count += 1

                #                   
                with torch.no_grad():
                    loss_dict['router_scores_mean'] = router_soft_scores_prev.mean()
                    loss_dict['router_scores_std'] = router_soft_scores_prev.std()
                    loss_dict['router_scores_min'] = router_soft_scores_prev.min()
                    loss_dict['router_scores_max'] = router_soft_scores_prev.max()
                    scores_flat = router_soft_scores_prev.detach().float().reshape(-1)
                    if scores_flat.numel() > 0:
                        loss_dict['router_prob_lt_005_ratio'] = (scores_flat < 0.05).float().mean()
                        loss_dict['router_prob_gt_095_ratio'] = (scores_flat > 0.95).float().mean()
                    logits_prev_float = logits_prev.detach().float().reshape(-1)
                    if logits_prev_float.numel() > 0:
                        loss_dict['router_logits_mean'] = logits_prev_float.mean()
                        loss_dict['router_logits_std'] = logits_prev_float.std(unbiased=False)
                        loss_dict['router_logits_min'] = logits_prev_float.min()
                        loss_dict['router_logits_max'] = logits_prev_float.max()
                        loss_dict['router_logits_p90'] = torch.quantile(logits_prev_float, 0.90)
                        loss_dict['router_logits_p95'] = torch.quantile(logits_prev_float, 0.95)
                    selection_prob_lt_005 = self._router_info.get('selection_prob_lt_005', None)
                    selection_prob_gt_095 = self._router_info.get('selection_prob_gt_095', None)
                    selection_logits_mean = self._router_info.get('selection_logits_mean', None)
                    selection_logits_std = self._router_info.get('selection_logits_std', None)
                    if selection_prob_lt_005 is not None:
                        loss_dict['router_selection_prob_lt_005_ratio'] = torch.tensor(float(selection_prob_lt_005), device=router_device)
                    if selection_prob_gt_095 is not None:
                        loss_dict['router_selection_prob_gt_095_ratio'] = torch.tensor(float(selection_prob_gt_095), device=router_device)
                    if selection_logits_mean is not None:
                        loss_dict['router_selection_logits_mean'] = torch.tensor(float(selection_logits_mean), device=router_device)
                    if selection_logits_std is not None:
                        loss_dict['router_selection_logits_std'] = torch.tensor(float(selection_logits_std), device=router_device)
                    #                     1.0 
                    if scores_flat.numel() > 0:
                        loss_dict['router_scores_p90'] = torch.quantile(scores_flat, 0.90)
                        loss_dict['router_scores_p95'] = torch.quantile(scores_flat, 0.95)

                    if prev_clip_dist is not None:
                        loss_dict['clip_distance_mean'] = prev_clip_dist.mean()
                        selected_mask = prev_mask > 0.5
                        if selected_mask.any():
                            loss_dict['selected_clip_dist'] = prev_clip_dist[selected_mask].mean()

                    loss_dict['reward_step'] = torch.tensor(float(self.state.global_step), device=router_device)
                    loss_dict['action_step'] = torch.tensor(float(delayed_action.get('action_step', -1)), device=router_device)
                    loss_dict['golden_eval_time'] = torch.tensor(float(golden_eval_time), device=router_device)
                    mmr = delayed_action.get('mask_match_rate', float('nan'))
                    loss_dict['mask_match_rate'] = torch.tensor(float(mmr), device=router_device)
                    loss_dict['mask_none_skip_count'] = torch.tensor(float(self.mask_none_skip_count), device=router_device)
                    loss_dict['candidate_mismatch_skip_count'] = torch.tensor(float(self.candidate_mismatch_skip_count), device=router_device)
                    loss_dict['golden_shard_id'] = torch.tensor(float(reward_shard_id), device=router_device)

                #              
                if action_record.get('action_valid', False):
                    self.router_action_queue.append(action_record)

                if is_main_rank and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                    print(f"[Router loss] step={self.state.global_step}, "
                          f"router_loss={router_loss.item() if router_loss is not None else None}, "
                          f"router_step_count={self.router_step_count}, "
                          f"reward_step={self.state.global_step}, action_step={delayed_action.get('action_step', -1)}, "
                          f"mask_match_rate={delayed_action.get('mask_match_rate', float('nan')):.4f}")

                return router_loss, loss_dict
            else:
                #         
                #    should_update=True        reward(t)           update-step     
                return None, {}
                
        except Exception as e:
            print(f"[Warning] Router forward   : {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    def training_step(self, model, inputs):
        """
           training_step   backward      Router forward

           [  1]      DeepSpeed ZeRO-3          
        - LLaVA forward + backward    
        -      Router forward + backward
        -             
        """
        #    Warmup        LLaVA batch        prepare 
        is_lightweight_router_batch = bool(inputs.get('router_has_all_candidates', False))
        if not is_lightweight_router_batch:
            inputs = self._prepare_inputs(inputs)

        # 1. LLaVA forward + backward
        llava_loss = self.compute_loss(model, inputs)
        
        # 2. LLaVA backward            
        #       HuggingFace Trainer           
        #        backward         gradient_accumulation_steps
        router_info_pre = getattr(self, '_router_info', {})
        pre_backward_is_warmup = router_info_pre.get('router_is_warmup', True)
        pre_backward_has_candidates = router_info_pre.get('router_candidate_indices', None) is not None
        predicted_router_micro_step = getattr(self, 'router_micro_step_count', 0) + 1
        should_capture_influence = (
            (not pre_backward_is_warmup)
            and pre_backward_has_candidates
            and (predicted_router_micro_step % self.router_update_interval == 0)
        )
        self.valve_trainer._install_online_influence_hooks(
            model,
            optimizer=self.optimizer,
            default_lora_lr=float(getattr(self.args, "learning_rate", 1.0)),
        )
        self.valve_trainer.online_influence_capture_enabled = bool(should_capture_influence)
        self.valve_trainer._clear_online_influence_hook_buffer()
        self.accelerator.backward(llava_loss)
        self.valve_trainer.online_influence_capture_enabled = False
        
        if self.debug_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        # 3.   backward      Router forward    LLaVA       
        router_info = getattr(self, '_router_info', {})
        router_is_warmup = router_info.get('router_is_warmup', True)
        router_candidate_indices = router_info.get('router_candidate_indices', None)
        router_clip_distance = router_info.get('router_clip_distance', None)
        router_cluster_ids = router_info.get('router_cluster_ids', None)
        selected_action_stats = None
        
        should_log_train_step = True
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                should_log_train_step = (dist.get_rank() == 0)
        except Exception:
            pass

        if should_log_train_step and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
            print(f"[training_step] step={self.state.global_step}, "
                  f"router_is_warmup={router_is_warmup}, "
                  f"has_candidate_indices={router_candidate_indices is not None}, "
                  f"will_train_router={not router_is_warmup and router_candidate_indices is not None}")
        
        router_loss = None
        router_loss_dict = {}

        #        rank   router_candidate_indices     None   warmup        
        #        rank    Router       all_reduce/backward    NCCL    
        #           rank      Router      Router             
        global_can_train_router = (not router_is_warmup and router_candidate_indices is not None)
        local_can_train_router = global_can_train_router
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                can_flag = torch.tensor(
                    [1 if local_can_train_router else 0],
                    device=llava_loss.device,
                    dtype=torch.int32,
                )
                can_flag_min = can_flag.clone()
                can_flag_max = can_flag.clone()
                dist.all_reduce(can_flag_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(can_flag_max, op=dist.ReduceOp.MAX)
                all_can = (can_flag_min.item() == 1)
                any_can = (can_flag_max.item() == 1)
                global_can_train_router = all_can
                #    rank           
                if any_can and not all_can and dist.get_rank() == 0 and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                    print(
                        f"[Router gate] step={self.state.global_step}     rank           "
                        f"local_can_train_router={local_can_train_router}        Router      "
                    )
        except Exception as e:
            print(f"[Warning] step={self.state.global_step} Router       : {e}")

        if global_can_train_router:
            selected_action_stats = self.valve_trainer._collect_online_selected_action_stats(
                model,
                optimizer=self.optimizer,
                default_lora_lr=float(getattr(self.args, "learning_rate", 1.0)),
            )
        else:
            self.valve_trainer._reset_online_influence_state()

        if global_can_train_router:
            t0_router_update = time.time()
            router_loss, router_loss_dict = self.compute_router_loss_after_backward(
                router_candidate_indices=router_candidate_indices,
                router_clip_distance=router_clip_distance,
                router_cluster_ids=router_cluster_ids,
                selected_action_stats=selected_action_stats,
            )
            router_loss_dict['router_update_time'] = float(time.time() - t0_router_update)

            #             rank   router_loss    rank   None    DDP/NCCL collective   
            local_has_router_loss = bool(router_loss is not None and getattr(router_loss, "requires_grad", False))
            global_has_router_loss = local_has_router_loss
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    has_flag = torch.tensor(
                        [1 if local_has_router_loss else 0],
                        device=llava_loss.device,
                        dtype=torch.int32,
                    )
                    has_min = has_flag.clone()
                    has_max = has_flag.clone()
                    dist.all_reduce(has_min, op=dist.ReduceOp.MIN)
                    dist.all_reduce(has_max, op=dist.ReduceOp.MAX)
                    all_has = (has_min.item() == 1)
                    any_has = (has_max.item() == 1)
                    global_has_router_loss = all_has
                    if any_has and not all_has:
                        router_loss = None
                        router_loss_dict["router_loss_rank_mismatch_skip"] = 1.0
                        if dist.get_rank() == 0 and (self.state.global_step % 10 == 0 or self.state.global_step <= 5):
                            print(
                                f"[Router sync] step={self.state.global_step}     rank   router_loss        "
                                f"       Router backward/step      "
                            )
            except Exception as e:
                print(f"[Warning] step={self.state.global_step} router_loss        : {e}")
            if not global_has_router_loss:
                router_loss = None

            #   4 Router    backward +    optimizer.step DDP         
            if router_loss is not None and router_loss.requires_grad:
                local_nonfinite = (not torch.isfinite(router_loss).item())

                #          rank backward    rank skip     DDP/NCCL collective    
                global_nonfinite = local_nonfinite
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        flag = torch.tensor(
                            [1 if local_nonfinite else 0],
                            device=router_loss.device,
                            dtype=torch.int32,
                        )
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                        global_nonfinite = (flag.item() > 0)
                except Exception as e:
                    print(f"[Warning] step={self.state.global_step}          : {e}")

                if global_nonfinite:
                    if local_nonfinite:
                        print(f"[Warning] step={self.state.global_step} router_loss is non-finite, skipping Router backward/step")
                    else:
                        print(f"[Warning] step={self.state.global_step} another rank has non-finite router_loss, skipping Router backward/step")
                    router_loss = None

            if router_loss is not None and router_loss.requires_grad:
                if self.router_optimizer is not None and self.router_accum_step_count == 0:
                    self.router_optimizer.zero_grad()

                router_backward_loss = router_loss
                if self.router_optimizer is not None and self.router_optimizer_accum_steps > 1:
                    router_backward_loss = router_loss / float(self.router_optimizer_accum_steps)
                router_loss_dict['router_backward_loss_scale'] = float(
                    1.0 / float(self.router_optimizer_accum_steps)
                    if self.router_optimizer is not None and self.router_optimizer_accum_steps > 1
                    else 1.0
                )
                router_backward_loss.backward()
                self.router_accum_step_count += 1

                is_final_training_step = False
                try:
                    is_final_training_step = (int(self.state.global_step) + 1 >= int(self.state.max_steps))
                except Exception:
                    is_final_training_step = False
                #               Router     slot  
                if self.router_total_update_steps is None:
                    total_router_updates, _ = self._estimate_router_update_counts()
                    self.router_total_update_steps = total_router_updates
                is_last_router_update_slot = (
                    self.router_total_update_steps is not None
                    and self.router_step_count >= self.router_total_update_steps
                )
                should_step_router_optimizer = (
                    self.router_accum_step_count >= self.router_optimizer_accum_steps
                ) or is_final_training_step or is_last_router_update_slot
                router_loss_dict['router_optimizer_step_applied'] = 0.0
                router_loss_dict['router_accum_progress'] = float(self.router_accum_step_count)
                router_loss_dict['router_optimizer_accum_steps'] = float(self.router_optimizer_accum_steps)
                router_loss_dict['router_last_update_slot_flush'] = float(1.0 if is_last_router_update_slot else 0.0)

                if self.router_optimizer is not None and should_step_router_optimizer:
                    router_model = self.valve_trainer.router.module if hasattr(self.valve_trainer.router, 'module') else self.valve_trainer.router

                    with torch.no_grad():
                        grad_sq_sum = 0.0
                        param_snapshots = {}
                        adam_mean_abs_m = []
                        adam_mean_sqrt_v = []
                        adam_mean_effective_step = []
                        for name, param in router_model.named_parameters():
                            if not param.requires_grad:
                                continue
                            if param.grad is not None:
                                grad_norm = param.grad.detach().float().norm(2)
                                grad_sq_sum += float(grad_norm.item()) ** 2
                            param_snapshots[name] = param.detach().float().clone()
                        for group in self.router_optimizer.param_groups:
                            for param in group.get('params', []):
                                if param is None or not param.requires_grad:
                                    continue
                                state = self.router_optimizer.state.get(param, None)
                                if not state:
                                    continue
                                exp_avg = state.get('exp_avg', None)
                                exp_avg_sq = state.get('exp_avg_sq', None)
                                if exp_avg is None or exp_avg_sq is None:
                                    continue
                                exp_avg_abs = exp_avg.detach().float().abs().mean().item()
                                exp_avg_sq_sqrt = exp_avg_sq.detach().float().sqrt().mean().item()
                                effective_step = (
                                    exp_avg.detach().float().abs()
                                    / (exp_avg_sq.detach().float().sqrt() + group.get('eps', 1e-8))
                                ).mean().item()
                                adam_mean_abs_m.append(exp_avg_abs)
                                adam_mean_sqrt_v.append(exp_avg_sq_sqrt)
                                adam_mean_effective_step.append(effective_step)
                        router_loss_dict['router_grad_norm'] = grad_sq_sum ** 0.5
                        if adam_mean_abs_m:
                            router_loss_dict['router_adam_mean_abs_m'] = float(sum(adam_mean_abs_m) / len(adam_mean_abs_m))
                        if adam_mean_sqrt_v:
                            router_loss_dict['router_adam_mean_sqrt_v'] = float(sum(adam_mean_sqrt_v) / len(adam_mean_sqrt_v))
                        if adam_mean_effective_step:
                            router_loss_dict['router_adam_mean_effective_step'] = float(sum(adam_mean_effective_step) / len(adam_mean_effective_step))

                    self.router_optimizer.step()
                    self.router_optimizer_step_count += 1
                    self.router_accum_step_count = 0

                    with torch.no_grad():
                        update_sq_sum = 0.0
                        for name, param in router_model.named_parameters():
                            if not param.requires_grad:
                                continue
                            prev_param = param_snapshots.get(name, None)
                            if prev_param is None:
                                continue
                            delta_norm = (param.detach().float() - prev_param).norm(2)
                            update_sq_sum += float(delta_norm.item()) ** 2
                        router_loss_dict['router_update_norm'] = update_sq_sum ** 0.5
                        selected_dataset_names = self._router_info.get('selected_dataset_names', []) or []
                        unselected_dataset_names = self._router_info.get('unselected_dataset_names', []) or []
                        selected_counter = Counter(selected_dataset_names)
                        unselected_counter = Counter(unselected_dataset_names)
                        all_dataset_names = sorted(set(selected_counter.keys()) | set(unselected_counter.keys()))
                        if all_dataset_names:
                            sign_vector = []
                            conflict_mass = 0.0
                            total_mass = 0.0
                            for dataset_name in all_dataset_names:
                                selected_count = float(selected_counter.get(dataset_name, 0))
                                unselected_count = float(unselected_counter.get(dataset_name, 0))
                                diff = selected_count - unselected_count
                                sign_vector.append(diff)
                                conflict_mass += min(selected_count, unselected_count)
                                total_mass += selected_count + unselected_count
                            nonzero_count = sum(1 for diff in sign_vector if abs(diff) > 0)
                            sign_sum = sum(1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0) for diff in sign_vector)
                            router_loss_dict['router_group_sign_consistency'] = abs(sign_sum) / max(nonzero_count, 1)
                            router_loss_dict['router_group_conflict_ratio'] = conflict_mass / max(total_mass, 1e-12)
                            selected_total = float(sum(selected_counter.values()))
                            if selected_total > 0:
                                selected_entropy = 0.0
                                for count in selected_counter.values():
                                    prob = float(count) / selected_total
                                    if prob > 0:
                                        selected_entropy -= prob * np.log(prob)
                                router_loss_dict['router_selected_dataset_entropy'] = selected_entropy
                        router_grad_norm = float(router_loss_dict.get('router_grad_norm', 0.0))
                        router_update_norm = float(router_loss_dict.get('router_update_norm', 0.0))
                        ratio = router_update_norm / max(router_grad_norm, 1e-12)
                        router_loss_dict['router_update_to_grad_ratio'] = ratio
                        router_loss_dict['router_update_norm_ema'] = self.router_update_norm_ema
                        router_loss_dict['router_update_lag'] = float(self.router_optimizer_step_count) / max(float(self.router_step_count), 1.0)
                        update_limited_flag = 1.0 if (router_grad_norm > 0 and ratio < self.update_limited_threshold) else 0.0
                        if update_limited_flag > 0:
                            self.update_limited_streak += 1
                        else:
                            self.update_limited_streak = 0
                        router_loss_dict['update_limited_flag'] = update_limited_flag
                        router_loss_dict['update_limited_streak'] = float(self.update_limited_streak)
                        router_loss_dict['router_optimizer_step_applied'] = 1.0
                        router_loss_dict['router_accum_progress'] = 0.0

                    self.router_optimizer.zero_grad()

                if self.router_optimizer is not None and should_step_router_optimizer:
                    if self.router_scheduler is None:
                        total_router_updates, total_router_optimizer_steps = self._estimate_router_update_counts()
                        self.router_total_update_steps = total_router_updates
                        self.router_scheduler_total_steps = total_router_optimizer_steps
                        self.router_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.router_optimizer,
                            T_max=total_router_optimizer_steps,
                            eta_min=self.router_lr_min,
                        )
                        print(f"[Scheduler] Router CosineAnnealingLR    : reward_updates={total_router_updates}, "
                              f"optimizer_steps={total_router_optimizer_steps}, "
                              f"accum={self.router_optimizer_accum_steps}, "
                              f"lr {self.router_lr:.2e}   {self.router_lr_min:.2e}")
                    if self.router_scheduler_total_steps is None or self.router_optimizer_step_count <= self.router_scheduler_total_steps:
                        self.router_scheduler.step()

                if self.router_optimizer is not None:
                    current_router_lr = self.router_optimizer.param_groups[0]['lr']
                    if not hasattr(self, '_current_router_loss_dict'):
                        self._current_router_loss_dict = {}
                    self._current_router_loss_dict['datavalve/router_lr'] = current_router_lr
                    self._current_router_loss_dict['datavalve/router_lr_min'] = self.router_lr_min

                #       Router backward   
                if self.debug_cuda_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # 4.    loss_dict   callback        training_step     wandb.log 
        #    HuggingFace Trainer + DeepSpeed          callback        
        if not hasattr(self, '_current_router_loss_dict'):
            self._current_router_loss_dict = {}

        selected_dataset_names = router_info.get('selected_dataset_names', []) if isinstance(router_info, dict) else []
        unselected_dataset_names = router_info.get('unselected_dataset_names', []) if isinstance(router_info, dict) else []
        selected_counter = Counter(selected_dataset_names or [])
        unselected_counter = Counter(unselected_dataset_names or [])
        all_datasets = sorted(set(selected_counter.keys()) | set(unselected_counter.keys()))
        if all_datasets:
            selected_total = float(sum(selected_counter.values()))
            unselected_total = float(sum(unselected_counter.values()))
            candidate_total = selected_total + unselected_total
            learnability_gap = 0.0
            dataset_enrich_values = []
            tracked_datasets = (
                "vg",
                "refcoco",
                "vqav2",
                "aokvqa",
                "gqa",
                "ocrvqa",
                "textcaps",
                "llava_instruct",
                "sharegpt",
            )
            for dataset_name in all_datasets:
                selected_count = float(selected_counter.get(dataset_name, 0))
                unselected_count = float(unselected_counter.get(dataset_name, 0))
                candidate_count = selected_count + unselected_count
                selected_ratio = selected_count / max(selected_total, 1.0)
                unselected_ratio = unselected_count / max(unselected_total, 1.0)
                candidate_ratio = candidate_count / max(candidate_total, 1.0)
                dataset_enrich = selected_ratio / max(candidate_ratio, 1e-12)
                learnability_gap += abs(selected_ratio - unselected_ratio)
                dataset_enrich_values.append(dataset_enrich)

                if dataset_name in tracked_datasets:
                    prefix = f"datavalve/selected_dataset_{dataset_name}"
                    self._current_router_loss_dict[prefix] = selected_ratio
                    self._current_router_loss_dict[f"datavalve/candidate_dataset_{dataset_name}"] = candidate_ratio
                    self._current_router_loss_dict[f"datavalve/dataset_enrich_{dataset_name}"] = dataset_enrich
            self.learnability_gap_ema = (
                self.learnability_ema_decay * self.learnability_gap_ema
                + (1.0 - self.learnability_ema_decay) * float(learnability_gap)
            )
            selected_entropy = 0.0
            for selected_count in selected_counter.values():
                prob = float(selected_count) / max(selected_total, 1.0)
                if prob > 0:
                    selected_entropy -= prob * np.log(prob)
            enrich_min = min(dataset_enrich_values) if dataset_enrich_values else 0.0
            enrich_max = max(dataset_enrich_values) if dataset_enrich_values else 0.0
            enrich_mean = float(sum(dataset_enrich_values) / max(len(dataset_enrich_values), 1))
            self._current_router_loss_dict['datavalve/selected_unselected_dataset_gap'] = float(learnability_gap)
            self._current_router_loss_dict['datavalve/selected_unselected_dataset_gap_ema'] = float(self.learnability_gap_ema)
        self._current_router_loss_dict.update({
            "train/llava_loss": llava_loss.item() if isinstance(llava_loss, torch.Tensor) else llava_loss,
        })
        if isinstance(router_info, dict):
            candidate_indices = router_info.get('candidate_indices', []) or []
            selected_indices = router_info.get('selected_indices', []) or []
            if len(candidate_indices) > 0:
                for metric_name in (
                    'selection_topk_margin',
                    'selected_score_mean',
                    'unselected_score_mean',
                    'selected_unselected_score_gap',
                    'router_selection_time',
                ):
                    metric_value = router_info.get(metric_name, None)
                    if metric_value is not None and np.isfinite(float(metric_value)):
                        self._current_router_loss_dict[f'datavalve/{metric_name}'] = float(metric_value)
            topk_margin = router_info.get('selection_topk_margin', None)
            if topk_margin is not None and np.isfinite(float(topk_margin)):
                if not self.valve_trainer.selection_topk_margin_ema_initialized:
                    self.valve_trainer.selection_topk_margin_ema = float(topk_margin)
                    self.valve_trainer.selection_topk_margin_ema_initialized = True
                else:
                    self.valve_trainer.selection_topk_margin_ema = (
                        0.9 * self.valve_trainer.selection_topk_margin_ema
                        + 0.1 * float(topk_margin)
                    )
        selected_seen = getattr(self.valve_trainer, 'selected_indices', [])
        selected_seen_total = float(len(selected_seen))
        if selected_seen_total > 0:
            selected_seen_unique = float(len(set(selected_seen)))
            dataset_len = 0
            try:
                dataset_len = len(self.train_dataset) if self.train_dataset is not None else 0
            except Exception:
                dataset_len = 0
            if dataset_len > 0:
                pass

        if torch.cuda.is_available():
            try:
                device = llava_loss.device if isinstance(llava_loss, torch.Tensor) else torch.device('cuda')
                if getattr(device, 'type', None) == 'cuda':
                    device_idx = device.index if device.index is not None else torch.cuda.current_device()
                else:
                    device_idx = torch.cuda.current_device()
            except Exception:
                pass
        
        #    Router              
        #    router_update_interval=4   logging_steps=10
        #   step % 4 != 0   router_loss_dict            0
        #       warmup      0   warmup          
        if router_loss_dict:
            for key, value in router_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    self._current_router_loss_dict[f"datavalve/{key}"] = value.item()
                else:
                    self._current_router_loss_dict[f"datavalve/{key}"] = value
        elif router_is_warmup:
            #    Warmup      0
            self._current_router_loss_dict.update({
                "datavalve/loss_reinforce": 0.0,
                "datavalve/advantage": 0.0,
            })
        # else:   warmup   router_loss_dict                 
        
        #    loss        HuggingFace Trainer            
        #     optimizer.step()    Trainer             accumulation        
        #      update                
        try:
            grad_accum_steps = max(1, int(getattr(self.args, 'gradient_accumulation_steps', 1)))
            self.valve_trainer.llava_micro_step_count += 1
            reached_update_boundary = False
            accelerator_sync_gradients = getattr(self.accelerator, 'sync_gradients', None)
            if isinstance(accelerator_sync_gradients, bool):
                reached_update_boundary = accelerator_sync_gradients
            elif grad_accum_steps <= 1:
                reached_update_boundary = True
            else:
                reached_update_boundary = (
                    self.valve_trainer.llava_micro_step_count % grad_accum_steps == 0
                )
            if reached_update_boundary:
                self.valve_trainer.accum_lora_grad_snapshot = {}
        except Exception:
            pass

        #           loss     detach      backward     
        total_loss = llava_loss.detach()
        if router_loss is not None:
            total_loss = total_loss + router_loss.detach()
        
        return total_loss


# =============================================================================
# Callback for logging DataValve metrics to wandb
# =============================================================================

class DataValveWandbCallback(TrainerCallback):
    """
        Callback      DataValve        wandb
    
       [CRITICAL FIX 2026-01-14] 
       HuggingFace   WandbCallback   on_log       wandb.log(logs)
          logs        WandbCallback     
    
      
              callback     wandb.log()
    """
    
    def __init__(self, trainer=None):
        """
        Args:
            trainer: LLaVATrainer_DataValve        callback     
        """
        self.trainer = trainer
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
          Trainer         
             wandb.log()        
        """
        if not HAS_WANDB:
            return
            
        #        
        if not state.is_world_process_zero:
            return
        
        #           trainer   
        trainer = self.trainer
        
        #       trainer     loss_dict       wandb
        if trainer is not None and hasattr(trainer, "_current_router_loss_dict"):
            loss_dict = trainer._current_router_loss_dict
            if loss_dict:
                try:
                    wandb.log(loss_dict)
                except Exception as e:
                    print(f"[DataValveWandbCallback] wandb.log   : {e}")
