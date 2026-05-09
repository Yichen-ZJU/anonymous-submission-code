"""
     DataValve -     

            
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DataValveConfig:
    """     DataValve   """
    
    # ====================      ====================
    train_data_path: str = "./data/remaining_664298.json"
    golden_data_path: str = "./data/selected_top1000.json"
    clip_feature_path: str = "./data/scores/llava_clip_feature.pt"
    image_folder: str = "./data"
    
    # LLaVA     
    model_name_or_path: str = "./llava-ckpt/vicuna-7b-v1.5/"
    vision_tower: str = "./llava-ckpt/clip-vit-large-patch14/"
    
    #     
    output_dir: str = "./checkpoints/datavalve"
    
    # ====================         ====================
    #     
    target_ratio: float = 0.208           #        (~20.8%)
    warmup_ratio: float = 0.05            #   5%      (Warm-up) [2026-01-22      ]
    final_batch_size: int = 8             #        LLaVA     
    super_batch_multiplier: int = 6       # Super-Batch = 6   Final = 48 [   16.67% Router    ]
    
    # Router     
    router_update_freq: int = 10          #   10   LLaVA step      Router
    
    # ====================        ====================
    #        lambda_sparsity   Top-K       
    lambda_reinforce: float = 0.1         #           
    
    # [ICML Enhancement]       
    
    # [ICML Enhancement] CVaR/DPP   
    
    
    # ==================== Router      ====================
    clip_image_dim: int = 768             # CLIP Image     
    clip_text_dim: int = 768              # CLIP Text     
    router_hidden_dim: int = 256          # Router      
    router_num_heads: int = 4             # CrossAttention   
    router_dropout: float = 0.1           # Dropout
    router_bias_init: float = 0.5         #    [2026-01-21 FIX]     bias     (sigmoid(0.5) 62%)
    
    # ====================      ====================
    num_epochs: int = 1
    learning_rate_llava: float = 2e-4     # LLaVA LoRA    
    learning_rate_router: float = 1e-4    # Router    
    weight_decay: float = 0.0
    warmup_steps: int = 100               # LR warmup
    
    # Batch   
    golden_batch_size: int = 32           #     batch
    gradient_accumulation_steps: int = 1
    
    # ==================== LoRA    ====================
    lora_enable: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # ====================      ====================
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4  #    spawn         > 0
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    
    # DeepSpeed
    deepspeed_config: Optional[str] = None
    local_rank: int = -1
    
    @property
    def super_batch_size(self) -> int:
        """   Super-Batch   """
        return self.final_batch_size * self.super_batch_multiplier
    
    @property
    def router_select_ratio(self) -> float:
        """Router        """
        return self.final_batch_size / self.super_batch_size
    
    @property
    def actual_total_ratio(self) -> float:
        """       """
        return self.warmup_ratio + (1 - self.warmup_ratio) * self.router_select_ratio
    
    def validate(self):
        """       """
        assert os.path.exists(self.train_data_path), f"       : {self.train_data_path}"
        assert os.path.exists(self.golden_data_path), f"        : {self.golden_data_path}"
        assert os.path.exists(self.clip_feature_path), f"CLIP      : {self.clip_feature_path}"
        assert 0 < self.target_ratio < 1, f"target_ratio     (0, 1)   "
        assert 0 < self.warmup_ratio < self.target_ratio, f"warmup_ratio      target_ratio"
        assert self.final_batch_size > 0, f"final_batch_size       "
        assert self.super_batch_multiplier > 1, f"super_batch_multiplier      1"
        
        print(f"[Config]       ")
        print(f"  Super-Batch   : {self.super_batch_size}")
        print(f"  Router     : {self.router_select_ratio:.2%}")
        print(f"         : {self.actual_total_ratio:.2%}")
    
    def __post_init__(self):
        """      """
        #       
        os.makedirs(self.output_dir, exist_ok=True)


#     
DEFAULT_CONFIG = DataValveConfig()
