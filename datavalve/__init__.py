"""
 DataValve - 

:
-  (Router): ,
-  ():   
-  (): ,

:
1. DataValveRouter:  CrossAttention 
2. DataValveLoss: ( +  + )
3. DataValveValveTrainer:  GRPO  Valve Trainer
4. LLaVATrainer_DataValve:  Router  LLaVA Trainer
5. DataValveConfig: 

:
1. Warmup  (3%):  LLaVA,
2. Router  (95%): 
   - : LLaVA  Router 
   - : Router  Golden Set 

:
- Super-Batch: 48  (8   6)
- Final-Batch: 8  (Top-K )
- : ~16.7% (8/48)
"""

from .config import DataValveConfig
from .router import DataValveRouter
from .losses import DataValveLoss, RouterLossWithGradientDetach, compute_selection_statistics
from .trainer import (
    DataValveValveTrainer,
    DataValveSuperBatchSampler,
    DataValveValveCollator,
    LLaVATrainer_DataValve,
)
from .dataset import (
    DataValveDataset,
    GoldenDataset,
    SuperBatchSampler,
    collate_fn,
    create_dataloaders,
)
from .utils import (
    load_clip_features,
    load_cluster_assignments,
    validate_data_consistency,
    TrainingLogger,
)

__all__ = [
    # Config
    "DataValveConfig",
    # Router
    "DataValveRouter",
    # Losses
    "DataValveLoss",
    "RouterLossWithGradientDetach",
    "compute_selection_statistics",
    # Trainer ( GRPO )
    "DataValveValveTrainer",
    "DataValveSuperBatchSampler",
    "DataValveValveCollator",
    "LLaVATrainer_DataValve",
    # Dataset
    "DataValveDataset",
    "GoldenDataset",
    "SuperBatchSampler",
    "collate_fn",
    "create_dataloaders",
    # Utils
    "load_clip_features",
    "load_cluster_assignments",
    "validate_data_consistency",
    "TrainingLogger",
]

__version__ = "1.0.0"
