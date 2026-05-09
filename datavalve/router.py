"""
 DataValve - Router  ()

:
1.  CLIP Image + Text 
2.  MLP (concat )
3. 
4.  Top-K + STE 

[2026-03-18 ] CrossAttention -> MLP
:,,CrossAttention 
(seq_len=1  gate).
MLP (~400K vs ~1.18M), REINFORCE .
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DataValveRouter(nn.Module):
    """
     Router  () - MLP 

    :
    concat([image_feat, text_feat]) -> MLP -> logit

    [2026-03-18]  CrossAttention  MLP:
    - :~1.18M -> ~400K
    - , REINFORCE 
    - (forward/get_selection_mask/compute_clip_distance)
    """

    def __init__(
        self,
        clip_image_dim: int = 768,
        clip_text_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,       # ,
        dropout: float = 0.1,
        bias_init: float = 0.5,
    ):
        super().__init__()

        self.clip_image_dim = clip_image_dim
        self.clip_text_dim = clip_text_dim
        self.hidden_dim = hidden_dim

        # concat 
        input_dim = clip_image_dim + clip_text_dim  # 1536

        # 4-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),   # 1536 -> 512
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 -> 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), # 256 -> 128
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Linear(hidden_dim // 2, 1)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.gate.bias, bias_init)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[DataValveRouter]  (MLP )")
        print(f"  CLIP Image Dim: {clip_image_dim}")
        print(f"  CLIP Text Dim: {clip_text_dim}")
        print(f"  Hidden Dim: {hidden_dim}")
        print(f"  Bias Init: {bias_init} (   {torch.sigmoid(torch.tensor(bias_init)).item():.1%})")
        print(f"  : {total_params:,}")
        # [V9 REMOVED] DEBUG CHECK  ZeRO-3 __init__ ,data[0] 

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: CLIP Image  [B, 768]
            text_features: CLIP Text  [B, 768]

        Returns:
            logits:  logits [B] ( sigmoid)
        """
        x = torch.cat([image_features, text_features], dim=-1)  # [B, 1536]
        x = self.mlp(x)                                          # [B, hidden_dim]
        return self.gate(x).squeeze(-1)                          # [B]

    def get_selection_mask(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         Top-K  mask ( STE)

        Args:
            image_features: CLIP Image  [B, 768]
            text_features: CLIP Text  [B, 768]
            k: 

        Returns:
            mask:  mask [B], 1.0 
            topk_indices:  [k]
            logits:  logits [B]
        """
        logits = self.forward(image_features, text_features)  # [B]
        mask, topk_indices = self.topk_ste(logits, k)
        return mask, topk_indices, logits

    @staticmethod
    def topk_ste(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-K + Straight-Through Estimator

        :  (Top-K)
        :  logits

        Args:
            logits:  logits [B]
            k: 

        Returns:
            mask:  mask [B]
            topk_indices:  [k]
        """
        B = logits.shape[0]
        k = min(k, B)

        _, topk_indices = torch.topk(logits, k, dim=0)

        mask_hard = torch.zeros_like(logits)
        mask_hard.scatter_(0, topk_indices, 1.0)

        # STE:  mask,
        soft_probs = torch.sigmoid(logits)
        mask = mask_hard.detach() + soft_probs - soft_probs.detach()

        return mask, topk_indices

    def compute_clip_distance(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
         CLIP  (Cosine Distance),

        Args:
            image_features: CLIP Image  [B, 768]
            text_features: CLIP Text  [B, 768]

        Returns:
            distance: Cosine Distance [B],  [0, 2]
        """
        image_norm = F.normalize(image_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)
        cos_sim = (image_norm * text_norm).sum(dim=-1)  # [B]
        return 1.0 - cos_sim  # [B]
