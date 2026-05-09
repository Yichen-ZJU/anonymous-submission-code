#           :
# - https://github.com/haotian-liu/LLaVA
# - DataValve     

#      DataValve                 

"""
     DataValve     


1.    CrossAttention Router    MLP Scorer
2.    Top-K + STE               
3.                   
4.               20     
5.    Golden Set        Router   
"""

import os
import copy
import math
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Any, Dict, Optional, Sequence, List

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
import tokenizers
import sys

# =====                =====
#       CUDA         
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  #       

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# wandb            wandb      
#      offline         WANDB_MODE   online         
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["WANDB_DISABLE_CODE"] = "true"

#          Python         datavalve   
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("./LLaVA")

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset, DataLoader, Subset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


from datavalve.config import DataValveConfig
from datavalve.router import DataValveRouter
from datavalve.llava_datavalve_model import LlavaLlamaForCausalLM_DataValve
from datavalve.trainer import (
    DataValveValveTrainer,
    DataValveSuperBatchSampler,
    DataValveValveCollator,
    LLaVATrainer_DataValve,
    DataValveWandbCallback,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

local_rank = None


def rank0_print(*args):
    """     (rank0)     """
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    """         """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    """          """
    data_path: str = field(default=None, metadata={"help": "      "})
    golden_data_path: Optional[str] = field(default=None, metadata={"help": "Golden Set   "})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """         """
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


    target_ratio: float = field(
        default=0.2,
        metadata={"help": "          0.2    20%"}
    )
    warmup_ratio_datavalve: float = field(
        default=0.03,
        metadata={"help": "Warm-up     "}
    )
    # [2026-01-23   ]    Sand Loss    (1.0 -> 0.5)    "    "   
    # CVaR      CLIP                   
    lambda_sand: float = field(
        default=0.5,
        metadata={"help": "      "}
    )
    lambda_diversity: float = field(
        default=0.5,
        metadata={"help": "       "}
    )
    # [2026-01-23   ]    RL    (0.1 -> 0.5)   Validation Loss       
    #                   CLIP/DPP    
    lambda_reinforce: float = field(
        default=0.5,
        metadata={"help": "REINFORCE / Policy Gradient     "}
    )

    # [ICML Enhancement]       
    loss_type_sand: str = field(
        default="cvar",
        metadata={"help": "      : 'mean' or 'cvar'"}
    )
    loss_type_div: str = field(
        default="dpp",
        metadata={"help": "       : 'entropy' or 'dpp'"}
    )

    # [ICML Enhancement] CVaR/DPP   
    cvar_alpha: float = field(
        default=0.5,
        metadata={"help": "CVaR       alpha%   "}
    )
    dpp_epsilon: float = field(
        default=1e-4,
        metadata={"help": "DPP        "}
    )

    # [ICML Enhancement] Zero-Shock Handover     
    norm_sand: float = field(
        default=1.0,  # [  ] 1.0
        metadata={"help": "Sand loss      "}
    )
    norm_div: float = field(
        default=0.1,  # [  ] 0.1 (   DPP Loss        ~15)
        metadata={"help": "Diversity loss       (DPP     )"}
    )
    norm_rl: float = field(
        default=1.0,  # [2026-01-23   ]   0.1     1.0   RL       
        metadata={"help": "RL loss      "}
    )
    router_lr: float = field(
        default=1e-4,
        metadata={"help": "Router    "}
    )
    router_lr_min: float = field(
        default=5e-5,
        metadata={"help": "Router cosine scheduler                    "}
    )
    router_hidden_dim: int = field(
        default=256,
        metadata={"help": "Router      "}
    )
    router_num_heads: int = field(
        default=4,
        metadata={"help": "Router      "}
    )
    router_update_interval: int = field(
        default=4,
        metadata={"help": "Router         "}
    )
    router_optimizer_accum_steps: int = field(
        default=1,
        metadata={"help": "Router                optimizer.step()    -shard     "}
    )

    ema_decay: float = field(
        default=0.99,
        metadata={"help": "REINFORCE baseline   EMA           "}
    )
    advantage_mode: str = field(
        default="delta",
        metadata={"help": "Advantage     : 'ema'(L_val-EMA         )   'delta'(L_val(t)-L_val(t-1)          )"}
    )
    gamma_ratio_penalty: float = field(
        default=0.5,
        metadata={"help": "Top-K RL        gamma1"}
    )
    gamma_entropy: float = field(
        default=0.01,
        metadata={"help": "Top-K RL       gamma2 loss    -gamma2*H "}
    )
    gamma_logit_anchor: float = field(
        default=0.0,
        metadata={"help": "V17.1-Lite: mean-logit anchor    z-loss style "}
    )
    logit_anchor_target: Optional[float] = field(
        default=None,
        metadata={"help": "V17.1-Lite: mean-logit anchor           log(target_ratio/(1-target_ratio))"}
    )
    reward_group_enable: bool = field(
        default=False,
        metadata={"help": "V16:      4       task-aware reward"}
    )
    group_norm: int = field(
        default=0,
        metadata={"help": "     group-aware reward normalization 1=   0=          reward "}
    )
    reward_group_n_min: int = field(
        default=2,
        metadata={"help": "V16:    golden batch                    reward"}
    )
    reward_group_clip: float = field(
        default=0.3,
        metadata={"help": "V16:    reward   clip   "}
    )
    reward_group_ema_decay: float = field(
        default=0.9,
        metadata={"help": "V16:    reward EMA    "}
    )
    reward_group_fallback_min_active: int = field(
        default=2,
        metadata={"help": "V16.1: active groups           advantage"}
    )
    advantage_norm_floor: float = field(
        default=0.05,
        metadata={"help": "V17.2 Full-min: advantage            "}
    )
    reward_activity_ema_decay: float = field(
        default=0.9,
        metadata={"help": "V17.2 Full-min: reward activity EMA    "}
    )
    reward_baseline_recent_k: int = field(
        default=4,
        metadata={"help": "V17.3 Run B: shard-aware short baseline        "}
    )
    reward_alpha_early: float = field(
        default=0.6,
        metadata={"help": "V17.3 Run B:    mixed baseline   EMA   "}
    )
    reward_alpha_late: float = field(
        default=0.3,
        metadata={"help": "V17.3 Run B:    mixed baseline   EMA   "}
    )
    reward_alpha_switch_ratio: float = field(
        default=0.2,
        metadata={"help": "V17.3 Run B: baseline alpha   early    late      "}
    )
    reward_shard_aware: bool = field(
        default=True,
        metadata={"help": "V17.3 Run B:      shard-aware reward baseline"}
    )
    diversity_min_scale: float = field(
        default=0.2,
        metadata={"help": "V17.2 Full-min: diversity          "}
    )
    diversity_reward_activity_target: float = field(
        default=0.05,
        metadata={"help": "V17.2 Full-min: reward activity       diversity gate     "}
    )
    diversity_reward_activity_power: float = field(
        default=1.0,
        metadata={"help": "V17.2 Full-min: reward activity gate    "}
    )


    use_clustering: bool = field(default=True)
    clustering_results_path: Optional[str] = field(default=None)
    cluster_assignments_path: Optional[str] = field(
        default=None,
        metadata={"help": "         "}
    )
    num_clusters: int = field(default=20)

    # CLIP     
    clip_features_path: str = field(
        default="./data/scores/llava_clip_feature.pt",
        metadata={"help": "CLIP       "}
    )
    grad_norm_path: Optional[str] = field(
        default=None,
        metadata={"help": "[deprecated]    sample-level grad norm JSON       online influence gate      "}
    )
    ig_z_clip: float = field(
        default=3.0,
        metadata={"help": "V20: delta/gate z-score clip"}
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    """
       DeepSpeed ZeRO-3        


    1.    modifier_rank=0       DeepSpeed rank 0       
    2.      active_sub_modules           
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        #    [CRITICAL FIX 1]    modifier_rank=0
        #    DeepSpeed rank 0           active_sub_modules   
        #    DeepSpeed          

        #    [CRITICAL FIX 2]         active_sub_modules    modifier_rank    
        if hasattr(param, "ds_active_sub_modules"):
            param.ds_active_sub_modules.clear()

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )

        #    [KEY FIX]    modifier_rank=0   
        with zero.GatheredParameters([param], modifier_rank=0):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    """   PEFT      LoRA     """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True, use_safe_get=False):
    """
        LoRA          MM Projector            Router 

       CoIDO     +           
    -       LoRA   requires_grad=True    
    -     MM Projector Router             
    -                    

    Args:
        named_params:        
        require_grad_only:             
        use_safe_get:      DeepSpeed     safe_get_full_fp32_param    2 
    Returns:
          LoRA           MM Projector + Router          
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

    #    2    DeepSpeed              
    if use_safe_get:
        try:
            from deepspeed.utils import safe_get_full_fp32_param
            to_return = {
                k: safe_get_full_fp32_param(v).cpu() 
                for k, v in to_return.items()
            }
            return to_return
        except ImportError:
            #             1
            pass

    #    1        maybe_zero_3     modifier_rank=0 
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
       MM Adapter   mm_projector      

       [CRITICAL]    LLaVA        MM Projector    
      : ./LLaVA/llava/train/train.py:164-167

      get_peft_state_non_lora_maybe_zero_3     
    -           keys_to_match    
    -       requires_grad
    -          MM Projector
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """   tokenizer     embedding    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """        """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    """V1       """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ],
        dim=0,
    )

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """        """
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    raise ValueError(f"Unknown conversation style: {conversation_lib.default_conversation.sep_style}")


class LazySupervisedDataset(Dataset):
    """            """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.lazy_datavalve_fetch = False

    def __len__(self):
        return len(self.list_data_dict)

    def get_datavalve_lazy_ref(self, i: int) -> Dict[str, Any]:
        sample_meta = self.list_data_dict[int(i)]
        source_value = sample_meta.get("dataset", sample_meta.get("source", sample_meta.get("dataset_name", "unknown")))
        return {
            "__datavalve_lazy_ref__": True,
            "__dataset_index__": int(i),
            "unique_idx": sample_meta.get("unique_idx", int(i)),
            "source": str(source_value),
        }

    def __getitems__(self, indices):
        if getattr(self, "lazy_datavalve_fetch", False):
            return [self.get_datavalve_lazy_ref(int(i)) for i in indices]
        return [self.__getitem__(int(i)) for i in indices]

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor


            try:
                from PIL import ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True  #          
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            except Exception as e:

                print(f"    Warning: Failed to load image {image_file}: {e}")
                print(f"   Skipping to next sample...")

                return self.__getitem__((i + 1) % len(self.list_data_dict))
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

        #    unique_idx
        data_dict["unique_idx"] = self.list_data_dict[i].get("unique_idx", i)

        # [V16]          Golden grouped reward     
        sample_meta = self.list_data_dict[i]
        source_value = sample_meta.get("dataset", sample_meta.get("source", sample_meta.get("dataset_name", "unknown")))
        data_dict["source"] = str(source_value)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """     """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        #    unique_indices
        batch["unique_indices"] = [instance["unique_idx"] for instance in instances]

        # [V16]        Golden grouped reward    
        if "source" in instances[0]:
            batch["sources"] = [instance["source"] for instance in instances]

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """           """
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train(attn_implementation=None):
    """     """
    global local_rank


    if "--group-norm" in sys.argv:
        sys.argv = ["--group_norm" if arg == "--group-norm" else arg for arg in sys.argv]


    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    #     wandb   WANDB_MODE        
    if local_rank == 0 or local_rank == -1:
        if HAS_WANDB and training_args.report_to and "wandb" in training_args.report_to:
            try:
                wandb_mode = str(os.environ.get("WANDB_MODE", "online")).strip().lower()
                if wandb_mode not in {"online", "offline", "disabled", "dryrun"}:
                    wandb_mode = "online"

                print(f"[Rank 0]       wandb (mode={wandb_mode})...")

                wandb.init(
                    project="DataValve",
                    name=training_args.run_name,
                    mode=wandb_mode,
                    config={
                        "target_ratio": training_args.target_ratio,
                        "warmup_ratio": training_args.warmup_ratio_datavalve,
                        "lambda_sand": training_args.lambda_sand,
                        "lambda_diversity": training_args.lambda_diversity,
                        "lambda_reinforce": training_args.lambda_reinforce,
                        "ema_decay": training_args.ema_decay,
                        "gamma_ratio_penalty": training_args.gamma_ratio_penalty,
                        "gamma_entropy": training_args.gamma_entropy,
                        "gamma_logit_anchor": training_args.gamma_logit_anchor,
                        "logit_anchor_target": training_args.logit_anchor_target,
                        "advantage_norm_floor": training_args.advantage_norm_floor,
                        "reward_activity_ema_decay": training_args.reward_activity_ema_decay,
                        "diversity_min_scale": training_args.diversity_min_scale,
                        "diversity_reward_activity_target": training_args.diversity_reward_activity_target,
                        "diversity_reward_activity_power": training_args.diversity_reward_activity_power,
                        "reward_group_enable": training_args.reward_group_enable,
                        "reward_group_n_min": training_args.reward_group_n_min,
                        "reward_group_clip": training_args.reward_group_clip,
                        "reward_group_ema_decay": training_args.reward_group_ema_decay,
                        "reward_group_fallback_min_active": training_args.reward_group_fallback_min_active,
                        # [ICML Enhancement]        
                        "loss_type_sand": training_args.loss_type_sand,
                        "loss_type_div": training_args.loss_type_div,
                        "cvar_alpha": training_args.cvar_alpha,
                        "dpp_epsilon": training_args.dpp_epsilon,
                        "norm_sand": training_args.norm_sand,
                        "norm_div": training_args.norm_div,
                        "norm_rl": training_args.norm_rl,

                        "router_lr": training_args.router_lr,
                        "router_lr_min": training_args.router_lr_min,
                        "router_optimizer_accum_steps": training_args.router_optimizer_accum_steps,
                        "learning_rate": training_args.learning_rate,
                        "batch_size": training_args.per_device_train_batch_size,
                    },
                    settings=wandb.Settings(
                        _disable_stats=True,  #               
                        _disable_meta=True   #        
                    )
                )
                print(f"[Rank 0]   wandb       (mode={wandb_mode})")
                if wandb_mode == "offline":
                    print("[Rank 0]     :           'wandb sync'          ")
            except Exception as e:
                print(f"[Rank 0]   wandb      : {e}")
                print("[Rank 0]           wandb")


    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # BNB   
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )


    rank0_print("=" * 60)
    rank0_print("       DataValve   ")
    rank0_print("=" * 60)

    model = LlavaLlamaForCausalLM_DataValve.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )

    rank0_print(f"        : {model_args.model_name_or_path}")

    model.config.use_cache = False

    # ===== LoRA    =====
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        rank0_print("\n" + "=" * 60)
        rank0_print("   LoRA")
        rank0_print("=" * 60)

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        rank0_print(f"LoRA r: {training_args.lora_r}")
        rank0_print(f"LoRA alpha: {training_args.lora_alpha}")
        rank0_print(f"LoRA dropout: {training_args.lora_dropout}")

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        rank0_print("  LoRA     ")
        rank0_print("=" * 60 + "\n")

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    #    tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        #    [CRITICAL FIX]    MM Projector              
        #    MM Projector     requires_grad=False          
        #      LLaVA           LoRA   
        model.config._mm_projector_trained = model_args.tune_mm_mlp_adapter
        rank0_print(f"[  ] MM Projector     : {model_args.tune_mm_mlp_adapter}")

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)


    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # =====    :   4 Router      + DDP  =====
    # LLaVA     DeepSpeed ZeRO-3 Router      DDP +       
    rank0_print("\n" + "=" * 60)
    rank0_print("        Router DDP       ")
    rank0_print("=" * 60)

    def _find_internal_router(m):
        if hasattr(m, 'datavalve_router'):
            return m.datavalve_router
        if hasattr(m, 'base_model'):
            return _find_internal_router(m.base_model)
        if hasattr(m, 'model'):
            return _find_internal_router(m.model)
        return None

    internal_router = _find_internal_router(model)
    if internal_router is not None:
        for p in internal_router.parameters():
            p.requires_grad = False
        rank0_print("          Router   ")

    router_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    # gate.bias             b0 = log(r / (1-r))
    #    sigmoid(b0)   target_ratio               
    target_ratio_for_bias = float(training_args.target_ratio)
    target_ratio_for_bias = min(max(target_ratio_for_bias, 1e-6), 1.0 - 1e-6)
    bias_init = math.log(target_ratio_for_bias / (1.0 - target_ratio_for_bias))

    router = DataValveRouter(
        clip_image_dim=768,
        clip_text_dim=768,
        hidden_dim=training_args.router_hidden_dim,
        num_heads=training_args.router_num_heads,
        dropout=0.1,
        bias_init=bias_init,
    ).to(device=training_args.device, dtype=router_dtype)

    #    Router      
    for param in router.parameters():
        param.requires_grad = True

    #         DDP    Router     rank      +        
    if torch.distributed.is_initialized():
        local_rank_for_ddp = int(os.environ.get("LOCAL_RANK", 0))
        router = DDP(
            router,
            device_ids=[local_rank_for_ddp],
            output_device=local_rank_for_ddp,
            find_unused_parameters=False,
        )
        rank0_print("  Router     DDP   ")

    rank0_print(f"  Router    : {sum(p.numel() for p in router.parameters()):,}")
    rank0_print(f"    :    Router DDP    ")
    rank0_print(f"  gate.bias    : {bias_init:.6f} (sigmoid {1/(1+math.exp(-bias_init)):.4f}, target_ratio={training_args.target_ratio:.4f})")

    # =====      =====
    dataset_size = len(data_module['train_dataset'])
    final_batch_size = training_args.per_device_train_batch_size
    target_ratio = training_args.target_ratio
    super_batch_size = int(final_batch_size / target_ratio)

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = super_batch_size * num_devices
    estimated_total_batches = (dataset_size // effective_batch_size) * int(training_args.num_train_epochs)

    rank0_print(f"[   ]      : {dataset_size}")
    rank0_print(f"[   ] Final Batch Size: {final_batch_size}")
    rank0_print(f"[   ] Target Ratio: {target_ratio:.1%}")
    rank0_print(f"[   ] Super-Batch Size: {super_batch_size}")
    rank0_print(f"[   ] GPU   : {num_devices}")
    rank0_print(f"[   ]     Batch  : {estimated_total_batches}")


    cluster_ids_path = None
    if training_args.use_clustering:
        #      cluster_assignments_path      clustering_results_path
        candidate_paths = [
            training_args.cluster_assignments_path,
            training_args.clustering_results_path,
        ]
        for path in candidate_paths:
            if path and os.path.exists(path):
                cluster_ids_path = path
                rank0_print(f"[   ]       : {cluster_ids_path}")
                break

        if cluster_ids_path is None:

            if training_args.loss_type_div == "entropy":
                rank0_print("[   ]   :           entropy diversity loss      ")
            else:
                rank0_print(f"[   ]   :             {training_args.loss_type_div} diversity loss        ")

    # =====    DataValveValveTrainer =====
    valve_trainer = DataValveValveTrainer(
        router=router,
        clip_features_path=training_args.clip_features_path,
        cluster_ids_path=cluster_ids_path,
        grad_norm_path=training_args.grad_norm_path,
        total_batches=estimated_total_batches,
        target_ratio=target_ratio,
        warmup_ratio=training_args.warmup_ratio_datavalve,
        lambda_sand=training_args.lambda_sand,
        lambda_diversity=training_args.lambda_diversity,
        lambda_reinforce=training_args.lambda_reinforce,
        num_clusters=training_args.num_clusters,
        ema_decay=training_args.ema_decay,
        # [ICML Enhancement]          
        loss_type_sand=training_args.loss_type_sand,
        loss_type_div=training_args.loss_type_div,
        cvar_alpha=training_args.cvar_alpha,
        dpp_epsilon=training_args.dpp_epsilon,
        # [ICML Enhancement] Zero-Shock Handover     
        norm_sand=training_args.norm_sand,
        norm_div=training_args.norm_div,
        norm_rl=training_args.norm_rl,
        advantage_mode=training_args.advantage_mode,
        gamma_ratio_penalty=0.0,
        gamma_entropy=training_args.gamma_entropy,
        gamma_logit_anchor=training_args.gamma_logit_anchor,
        logit_anchor_target=training_args.logit_anchor_target,
        reward_group_enable=False,
        group_norm=False,
        reward_group_n_min=training_args.reward_group_n_min,
        reward_group_clip=training_args.reward_group_clip,
        reward_group_ema_decay=training_args.reward_group_ema_decay,
        reward_group_fallback_min_active=training_args.reward_group_fallback_min_active,
        advantage_norm_floor=training_args.advantage_norm_floor,
        reward_activity_ema_decay=training_args.reward_activity_ema_decay,
        diversity_min_scale=training_args.diversity_min_scale,
        diversity_reward_activity_target=training_args.diversity_reward_activity_target,
        diversity_reward_activity_power=training_args.diversity_reward_activity_power,
        ig_z_clip=training_args.ig_z_clip,
        device='cuda',
        checkpoint_dir=training_args.output_dir,  #       
        save_interval=1000,  #   1000   batch     
    )
    rank0_print("  DataValveValveTrainer     ")

    # =====    Super-Batch Sampler =====
    rank0_print("\n" + "=" * 60)
    rank0_print("   Super-Batch Sampler         ")
    rank0_print("=" * 60)

    # [FIX]        
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        rank0_print(f"       : rank={rank}, world_size={world_size}")
    else:
        rank = 0
        world_size = 1
        rank0_print(f"        ")

    # [2026-01-27 NEW]           modality_lengths
    train_dataset = data_module['train_dataset']
    has_modality_lengths = hasattr(train_dataset, 'modality_lengths')
    rank0_print(f"        modality_lengths: {has_modality_lengths}")
    if has_modality_lengths:
        rank0_print(f"        group_by_modality_length      LLaVA      ")

    super_batch_sampler = DataValveSuperBatchSampler(
        data_source=train_dataset,
        final_batch_size=final_batch_size,
        target_ratio=target_ratio,
        warmup_ratio=training_args.warmup_ratio_datavalve,
        shuffle=True,
        drop_last=True,
        rank=rank,           # [FIX]       rank
        world_size=world_size,  # [FIX]       world_size
        seed=42,             # [FIX]           
        group_by_modality_length=training_args.group_by_modality_length,  # [2026-01-27 NEW]       
    )
    valve_trainer.grpo_sampler = super_batch_sampler
    # [     ]    Sampler            
    valve_trainer.warmup_batches = super_batch_sampler.warmup_batches
    valve_trainer.total_batches = super_batch_sampler.total_batches
    rank0_print("  Super-Batch Sampler             ")

    # =====    Valve Collator =====
    rank0_print("\n" + "=" * 60)
    rank0_print("      Valve Collator")
    rank0_print("=" * 60)

    original_collate_fn = data_module['data_collator']

    valve_collator = DataValveValveCollator(
        valve_trainer=valve_trainer,
        dataset=data_module['train_dataset'],
        final_batch_size=final_batch_size,
        target_ratio=target_ratio,
        warmup_ratio=training_args.warmup_ratio_datavalve,
        original_collate_fn=original_collate_fn,
    )

    if hasattr(data_module['train_dataset'], 'lazy_datavalve_fetch'):
        data_module['train_dataset'].lazy_datavalve_fetch = True
        rank0_print("  Router      lazy candidate fetch    index/meta      __getitem__ ")

    data_module['data_collator'] = valve_collator
    rank0_print("      Valve Collator    ")
    rank0_print("=" * 60 + "\n")

    # =====    Golden Set DataLoader =====
    golden_dataloader = None
    golden_shard_dataloaders = []
    if data_args.golden_data_path and os.path.exists(data_args.golden_data_path):
        rank0_print("\n" + "=" * 60)
        rank0_print("   Golden Set DataLoader")
        rank0_print("=" * 60)

        #    Golden Set   
        golden_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.golden_data_path,
            data_args=data_args,
        )

        golden_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        golden_batch_size = training_args.per_device_eval_batch_size
        golden_dataloader = DataLoader(
            golden_dataset,
            batch_size=golden_batch_size,
            shuffle=True,
            collate_fn=golden_collator,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

        rank0_print(f"  Golden Set   : {len(golden_dataset)}")

        golden_total = len(golden_dataset)
        num_golden_shards = 4
        shard_ranges = []
        for shard_id in range(num_golden_shards):
            start = (golden_total * shard_id) // num_golden_shards
            end = (golden_total * (shard_id + 1)) // num_golden_shards
            if start < end:
                shard_ranges.append((start, end))
                shard_dataset = Subset(golden_dataset, list(range(start, end)))
                golden_shard_dataloaders.append(
                    DataLoader(
                        shard_dataset,
                        batch_size=golden_batch_size,
                        shuffle=False,
                        collate_fn=golden_collator,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True,
                    )
                )
        rank0_print(f"  Golden Shards(range-based): {len(golden_shard_dataloaders)}")
        rank0_print(f"  Ranges: {shard_ranges}")
        rank0_print("=" * 60 + "\n")
    else:
        rank0_print("[   ]   : Golden Set     REINFORCE       ")

    #    generation_config
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.do_sample = True
        rank0_print("      generation_config (   do_sample=True)")

    #    Trainer
    trainer = LLaVATrainer_DataValve(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        valve_trainer=valve_trainer,
        golden_dataloader=golden_dataloader,
        golden_shard_dataloaders=golden_shard_dataloaders,
        router_lr=training_args.router_lr,
        router_lr_min=training_args.router_lr_min,
        router_update_interval=training_args.router_update_interval,
        router_optimizer_accum_steps=training_args.router_optimizer_accum_steps,
        original_collate_fn=original_collate_fn,
        **data_module,
    )

    #       Callback      DataValve     wandb
    #    [CRITICAL FIX]      trainer       callback      _current_router_loss_dict
    datavalve_callback = DataValveWandbCallback(trainer=trainer)
    trainer.add_callback(datavalve_callback)
    rank0_print("      DataValveWandbCallback")


    rank0_print("\n" + "=" * 60)
    rank0_print("    ")
    rank0_print("=" * 60)

    #       DataLoader   
    rank0_print(f"   valve_trainer.grpo_sampler: {hasattr(valve_trainer, 'grpo_sampler')}")
    if hasattr(valve_trainer, 'grpo_sampler'):
        rank0_print(f"  grpo_sampler  : {valve_trainer.grpo_sampler}")
    rank0_print(f"   trainer.valve_trainer: {hasattr(trainer, 'valve_trainer')}")
    if hasattr(trainer, 'valve_trainer'):
        rank0_print(f"  trainer.valve_trainer.grpo_sampler: {hasattr(trainer.valve_trainer, 'grpo_sampler')}")
        if hasattr(trainer.valve_trainer, 'grpo_sampler'):
            rank0_print(f"     : {trainer.valve_trainer.grpo_sampler}")

    train_dataloader = trainer.get_train_dataloader()
    rank0_print(f"DataLoader   : {type(train_dataloader)}")
    rank0_print(f"Batch Sampler: {train_dataloader.batch_sampler}")
    rank0_print(f"Collate Fn: {type(train_dataloader.collate_fn)}")
    rank0_print("=" * 60 + "\n")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ===== [2026-01-19 FIX]    DataLoader   barrier    =====
    #          DataLoader worker       
    #      barrier/synchronize            

    try:
        #    golden_dataloader       
        if 'golden_dataloader' in dir() and golden_dataloader is not None:
            del golden_dataloader

        #    Python     
        import gc
        gc.collect()

        #    CUDA        synchronize 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        pass  #             

    # =====        =====
    rank0_print("\n" + "=" * 60)
    rank0_print("     DataValve     ")
    rank0_print("=" * 60)

    valve_stats = valve_trainer.get_stats()
    collator_stats = valve_collator.get_stats()

    #    rank    
    rank0_print(f"Valve Trainer (   Rank {local_rank}):")
    rank0_print(f"      : {valve_stats['current_batch']}/{valve_stats['total_batches']}")
    rank0_print(f"      : {valve_stats['target_ratio']:.2%}")
    rank0_print(f"      : {valve_stats['actual_ratio']:.2%}")
    rank0_print(f"     : {valve_stats['total_sampled']:,}")
    rank0_print(f"      : {valve_stats['total_candidates']:,}")
    rank0_print(f"       : {valve_stats['unique_selected']:,}")

    # [2025-01-14 FIX]      GPU    
    #    [2026-01-19 CRITICAL FIX]    all_reduce     
    #       NCCL          all_reduce       
    #    realtime                  
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()

        #      all_reduce       
        # sampled_tensor = torch.tensor([valve_stats['total_sampled']], dtype=torch.long, device=local_rank)
        # torch.distributed.all_reduce(sampled_tensor, op=torch.distributed.ReduceOp.SUM)
        # total_sampled_all = sampled_tensor.item()

        #        world_size        
        total_sampled_all = valve_stats['total_sampled'] * world_size
        total_candidates_all = valve_stats['total_candidates'] * world_size

        actual_ratio_all = total_sampled_all / total_candidates_all if total_candidates_all > 0 else 0

        rank0_print(f"\n        {world_size}   GPU :")
        rank0_print(f"          : {total_sampled_all:,}")
        rank0_print(f"          : {total_candidates_all:,}")
        rank0_print(f"      : {actual_ratio_all:.2%}")
        rank0_print(f"      : {valve_stats['target_ratio']:.2%}")

    rank0_print(f"\nCollator:")
    rank0_print(f"     : {collator_stats['total_batches']}")
    rank0_print(f"  Warmup   : {collator_stats['warmup_batches']}")
    rank0_print(f"  Router   : {collator_stats['router_batches']}")
    rank0_print("=" * 60 + "\n")

    trainer.save_state()
    model.config.use_cache = True

    # =====    DataValve =====
    #    CoIDO:                  
    rank0_print("\n[  ]    DataValve       ...")

    # 1.                      DeepSpeed 
    if local_rank == 0 or local_rank == -1:
        try:
            indices_path = os.path.join(training_args.output_dir, "selected_indices.json")
            valve_trainer.save_selected_indices(indices_path)

            selected_dataset_path = os.path.join(training_args.output_dir, "selected_dataset.json")
            valve_trainer.export_selected_dataset(
                original_data_path=data_args.data_path,
                output_path=selected_dataset_path
            )
            rank0_print(f"            ")
        except Exception as e:
            rank0_print(f"            : {e}")

    rank0_print("[  ]           CoIDO    ...")

    if training_args.lora_enable:
        rank0_print("\n" + "=" * 60)
        rank0_print("   LoRA       CoIDO    ")
        rank0_print("=" * 60)

        # =====    LoRA       CoIDO/LLaVA    =====
        rank0_print("[  ]    LoRA   ...")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        rank0_print(f"[  ]     {len(state_dict)}   LoRA   ")

        # =====    Non-LoRA       MM Projector   Router =====
        #    [ULTIMATE FIX]    model.state_dict()               DeepSpeed GatheredParameters
        #        active_sub_modules   
        #    [CRITICAL FIX]     Router               
        #        Router                      
        rank0_print("[  ]     Router                ...")

        # [CRITICAL FIX 2026-03-18] ZeRO-3      deepspeed.zero.GatheredParameters
        #   4 Router       DDP     ZeRO   
        router_dict = {}
        try:
            save_router = router.module if hasattr(router, 'module') else router
            router_params = list(save_router.named_parameters())
            if local_rank == 0 or local_rank == -1:
                for key, param in router_params:
                    prefix = 'base_model.datavalve_router.' if hasattr(model, 'base_model') else 'datavalve_router.'
                    router_dict[prefix + key] = param.data.cpu().clone()
                rank0_print(f"[  ]     Router      : {len(router_dict)}  ")
        except Exception as e:
            rank0_print(f"[  ]         Router          state_dict: {e}")
            save_router = router.module if hasattr(router, 'module') else router
            router_state_dict = save_router.state_dict()
            for key, value in router_state_dict.items():
                prefix = 'base_model.datavalve_router.' if hasattr(model, 'base_model') else 'datavalve_router.'
                router_dict[prefix + key] = value.cpu().clone()

        rank0_print(f"[  ]   Router       {len(router_dict)}    ")

        # =====    non-LoRA         A      maybe_zero_3 =====
        #    [CRITICAL FIX 2026-01-09]     require_grad_only=True 
        # require_grad_only=False     7B        ~14GB                 OOM
        # 1%                      
        rank0_print("[  ]    MM Projector   ...")
        non_lora_state_dict = {}

        try:
            #    [KEY FIX]     MM Projector        
            #    LLaVA     get_mm_adapter_state_maybe_zero_3   
            mm_projector_state_dict = get_mm_adapter_state_maybe_zero_3(
                model.named_parameters(),
                keys_to_match=['mm_projector']  #     mm_projector
            )

            if len(mm_projector_state_dict) > 0:
                non_lora_state_dict = mm_projector_state_dict
                rank0_print(f"[  ]         {len(non_lora_state_dict)}   MM Projector   ")

                #    MM Projector     
                expected_mm_params = [
                    'base_model.model.model.mm_projector.0.weight',
                    'base_model.model.model.mm_projector.0.bias',
                    'base_model.model.model.mm_projector.2.weight',
                    'base_model.model.model.mm_projector.2.bias',
                ]
                missing_mm_params = [k for k in expected_mm_params if k not in non_lora_state_dict]

                if len(non_lora_state_dict) < 4:
                    rank0_print(f"       : MM Projector       {len(non_lora_state_dict)}/{len(expected_mm_params)} ")
                    if missing_mm_params:
                        rank0_print(f"       : {missing_mm_params}")
                        rank0_print("            ...")


                        if hasattr(model_args, 'pretrain_mm_mlp_adapter') and model_args.pretrain_mm_mlp_adapter:
                            try:
                                pretrain_path = model_args.pretrain_mm_mlp_adapter
                                if os.path.exists(pretrain_path):
                                    pretrain_params = torch.load(pretrain_path, map_location='cpu')
                                    for pretrain_key, pretrain_value in pretrain_params.items():
                                        if pretrain_key.startswith('model.mm_projector.'):
                                            target_key = 'base_model.model.' + pretrain_key
                                            if target_key in missing_mm_params:
                                                non_lora_state_dict[target_key] = pretrain_value.cpu().clone()
                                                rank0_print(f"        : {target_key}")
                            except Exception as e:
                                rank0_print(f"           : {e}")
                else:
                    rank0_print("    MM Projector     ")

                #    [CRITICAL FIX] Router              non_lora_trainables.bin  
                if len(router_dict) > 0:
                    rank0_print(f"    Router      : {len(router_dict)}         ")
                else:
                    rank0_print("       : Router       ")
            else:
                rank0_print("[  ]          non_lora_trainables.bin      ")
                rank0_print("        DeepSpeed        ")
        except Exception as e:
            rank0_print(f"[  ]         : {e}")
            import traceback
            rank0_print(traceback.format_exc())
            non_lora_state_dict = {}

        # 4.    rank 0         LLaVA     CoIDO 
        if local_rank == 0 or local_rank == -1:
            #    [CRITICAL FIX]         LLaVA          Llama   
            #      model.base_model.model.config     model.config (LoRA wrapper)
            rank0_print("[  ]            LLaVA   ...")

            #       LLaVA         LoRA wrapper 
            if hasattr(model, 'base_model'):
                if hasattr(model.base_model, 'model'):
                    # PeftModel -> base_model -> LlavaLlamaForCausalLM -> config
                    base_config = model.base_model.model.config
                else:
                    base_config = model.base_model.config
            else:
                base_config = model.config

            #    [FIX]      model_type   llava_llama          
            base_config.model_type = 'llava_llama'
            base_config.architectures = ['LlavaLlamaForCausalLM']
            rank0_print(f"       model_type = llava_llama")
            rank0_print(f"       architectures = ['LlavaLlamaForCausalLM']")

            #    [FIX]    _name_or_path       
            if hasattr(model_args, 'model_name_or_path') and model_args.model_name_or_path:
                base_config._name_or_path = model_args.model_name_or_path
                rank0_print(f"       _name_or_path = {model_args.model_name_or_path}")

            #    [FIX]        LLaVA             
            llava_config_fields = {
                # Vision Tower   
                'mm_vision_tower': getattr(model_args, 'vision_tower', './checkpoints/clip-vit-large-patch14'),
                'mm_vision_select_layer': getattr(model_args, 'mm_vision_select_layer', -2),
                'mm_vision_select_feature': 'patch',
                'mm_hidden_size': 1024,

                # Projector   
                'mm_projector_type': getattr(model_args, 'mm_projector_type', 'mlp2x_gelu'),
                'mm_projector_lr': getattr(training_args, 'mm_projector_lr', 2e-5),
                'use_mm_proj': True,
                'tune_mm_mlp_adapter': getattr(model_args, 'tune_mm_mlp_adapter', False),
                'freeze_mm_mlp_adapter': not getattr(model_args, 'tune_mm_mlp_adapter', False),

                # Image     
                'image_aspect_ratio': getattr(data_args, 'image_aspect_ratio', 'pad'),
                'mm_patch_merge_type': 'flat',
                'mm_use_im_start_end': getattr(model_args, 'mm_use_im_start_end', False),
                'mm_use_im_patch_token': getattr(model_args, 'mm_use_im_patch_token', False),

                # Tokenizer   
                'tokenizer_model_max_length': getattr(training_args, 'model_max_length', 2048),
                'tokenizer_padding_side': 'right',

                # Generation   
                'do_sample': True,

                #    LLaMA         
                'attention_bias': False,
                'attention_dropout': 0.0,
            }

            #    [FIX]                
            for key, value in llava_config_fields.items():
                if value is not None:
                    setattr(base_config, key, value)
                    rank0_print(f"    {key} = {value}")

            #    [CRITICAL]    base_config        model.config
            base_config.save_pretrained(training_args.output_dir)
            rank0_print(f"\n         : {training_args.output_dir}/config.json")


            saved_config_path = os.path.join(training_args.output_dir, 'config.json')
            if os.path.exists(saved_config_path):
                import json
                with open(saved_config_path, 'r') as f:
                    saved_config = json.load(f)


                required_fields = ['model_type', 'mm_projector_type', 'mm_vision_tower', 'image_aspect_ratio']
                missing_fields = [f for f in required_fields if f not in saved_config]

                if saved_config.get('model_type') == 'llava_llama' and not missing_fields:
                    rank0_print("          :          ")
                    rank0_print(f"     model_type = {saved_config.get('model_type')}")
                    rank0_print(f"     mm_projector_type = {saved_config.get('mm_projector_type')}")
                    rank0_print(f"     mm_vision_tower = {saved_config.get('mm_vision_tower')}")
                else:
                    rank0_print(f"       :        ")
                    rank0_print(f"     model_type = {saved_config.get('model_type')}")
                    if missing_fields:
                        rank0_print(f"         : {missing_fields}")

            #    LoRA   
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            rank0_print(f"  LoRA      : {training_args.output_dir}")

            #    Tokenizer
            tokenizer.save_pretrained(training_args.output_dir)
            rank0_print(f"  Tokenizer    : {training_args.output_dir}")

            #    [CRITICAL FIX]    Non-LoRA        MM Projector     Router 
            # Router           non_lora_trainables.bin      
            # 1. Router    LLaVA       
            # 2. LLaVA   builder.py     non_lora_trainables.bin       
            # 3. Router              LLaVA           
            if non_lora_state_dict:
                non_lora_path = os.path.join(training_args.output_dir, 'non_lora_trainables.bin')
                torch.save(non_lora_state_dict, non_lora_path)
                file_size_mb = os.path.getsize(non_lora_path) / 1024 / 1024
                rank0_print(f"  Non-LoRA      : {non_lora_path}")
                rank0_print(f"      : {file_size_mb:.2f} MB")
                rank0_print(f"      : {len(non_lora_state_dict)}")
                rank0_print(f"       :     MM Projector        Router   ")


                mm_projector_params = [k for k in non_lora_state_dict.keys() if 'mm_projector' in k]
                other_params = [k for k in non_lora_state_dict.keys() if k not in mm_projector_params]

                rank0_print("\n         :")
                if mm_projector_params:
                    rank0_print(f"\n  MM Projector ({len(mm_projector_params)}  ):")
                    for key in sorted(mm_projector_params):
                        shape = list(non_lora_state_dict[key].shape)
                        rank0_print(f"    - {key}: {shape}")

                if other_params:
                    rank0_print(f"\n       ({len(other_params)}  ):")
                    for key in sorted(other_params):
                        shape = list(non_lora_state_dict[key].shape)
                        rank0_print(f"    - {key}: {shape}")

            #    [CRITICAL FIX]      Router         
            if router_dict:
                router_path = os.path.join(training_args.output_dir, 'router_weights.bin')
                torch.save(router_dict, router_path)
                router_size_mb = os.path.getsize(router_path) / 1024 / 1024
                rank0_print(f"\n  Router        : {router_path}")
                rank0_print(f"      : {router_size_mb:.2f} MB")
                rank0_print(f"      : {len(router_dict)}")
                rank0_print(f"       : Router        LLaVA      ")
            else:
                rank0_print("     : non_lora_state_dict   ")
                rank0_print("  non_lora_trainables.bin    ")

            rank0_print("=" * 60 + "\n")

    # =====        =====
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        rank0_print("[  ] LoRA            ")

    # =====         =====
    if local_rank == 0 or local_rank == -1:
        rank0_print("\n" + "=" * 60)
        rank0_print("    DataValve     ")
        rank0_print("=" * 60)


        non_lora_path = os.path.join(training_args.output_dir, "non_lora_trainables.bin")
        indices_path = os.path.join(training_args.output_dir, "selected_indices.json")
        dataset_path = os.path.join(training_args.output_dir, "selected_dataset.json")

        #    non_lora_trainables.bin      Router
        if os.path.exists(non_lora_path):
            saved_params = torch.load(non_lora_path, map_location='cpu')
            router_params_saved = [k for k in saved_params.keys() if 'router' in k.lower() or 'datavalve' in k.lower()]
            rank0_print(f"  non_lora_trainables.bin    ")
            rank0_print(f"  -     : {len(saved_params)}")
            rank0_print(f"  - Router    : {len(router_params_saved)}")
            if len(router_params_saved) > 0:
                rank0_print(f"    Router        non_lora_trainables.bin  !")
            else:
                rank0_print(f"       : Router      non_lora_trainables.bin  ")
        else:
            rank0_print(f"   non_lora_trainables.bin    : {non_lora_path}")

        if os.path.exists(indices_path):
            rank0_print(f"         : {indices_path}")
        else:
            rank0_print(f"          : {indices_path}")

        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                selected_data = json.load(f)
            rank0_print(f"          : {dataset_path}")
            rank0_print(f"       : {len(selected_data)}")
        else:
            rank0_print(f"           : {dataset_path}")


        if hasattr(data_module['train_dataset'], 'truncated_samples') and data_module['train_dataset'].truncated_samples:
            truncated_path = os.path.join(training_args.output_dir, "truncated.json")
            with open(truncated_path, 'w') as f:
                json.dump(data_module['train_dataset'].truncated_samples, f, indent=2)
            rank0_print(f"         : {truncated_path} ({len(data_module['train_dataset'].truncated_samples)}  )")

        #    [CRITICAL FIX 2026-01-13]           config.json 
        #    vicuna-7b-v1.5   config.json    LLaMA         LLaVA   
        #                 LLaVA config              LLaVA
        #       generation_config.json            
        import shutil
        base_model_path = model_args.model_name_or_path
        #    [FIX]     generation_config.json     config.json 
        config_files_to_copy = ['generation_config.json']  #     config.json
        for config_file in config_files_to_copy:
            src = os.path.join(base_model_path, config_file)
            dst = os.path.join(training_args.output_dir, config_file)

            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                rank0_print(f"         : {config_file}")
            elif os.path.exists(dst):
                rank0_print(f"     {config_file}     ")

        rank0_print("=" * 60 + "\n")

        if HAS_WANDB and training_args.report_to and "wandb" in training_args.report_to:
            wandb.finish()

    # =====          =====
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        rank0_print("[  ]            ")


if __name__ == "__main__":
    disable_fa = os.environ.get("DISABLE_FLASH_ATTENTION", "0") == "1"
    attn_impl = None if disable_fa else "flash_attention_2"
    train(attn_implementation=attn_impl)
