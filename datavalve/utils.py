"""
 DataValve - 

:
1. CLIP 
2. 
3. 
4. 
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# ====================  ====================

def load_clip_features(feature_path: str) -> Dict[str, torch.Tensor]:
    """
     CLIP 

    Args:
        feature_path:  (.pt)

    Returns:
        features: {unique_idx: feature_tensor}
    """
    print(f"[load_clip_features] : {feature_path}")
    features = torch.load(feature_path, map_location="cpu")
    print(f"  : {len(features)}")
    sample_key = list(features.keys())[0]
    sample_feature = features[sample_key]
    if isinstance(sample_feature, np.ndarray):
        sample_feature = torch.from_numpy(sample_feature)
    print(f"  : {sample_feature.shape}")

    return features

def split_clip_features(
    features: Dict[str, torch.Tensor],
    image_dim: int = 768,
    text_dim: int = 768,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
     CLIP 

    Args:
        features:  {unique_idx: [image || text]}
        image_dim: 
        text_dim: 

    Returns:
        image_features: {unique_idx: image_feature}
        text_features: {unique_idx: text_feature}
    """
    image_features = {}
    text_features = {}

    for uid, feat in features.items():
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)

        image_features[uid] = feat[:image_dim]
        text_features[uid] = feat[image_dim:image_dim + text_dim]

    return image_features, text_features

def normalize_features(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    L2 
    """
    normalized = {}
    for uid, feat in features.items():
        normalized[uid] = F.normalize(feat.unsqueeze(0), p=2, dim=1).squeeze(0)
    return normalized

# ====================  ====================

def load_cluster_assignments(cluster_path: str) -> Dict[str, int]:
    """


    Args:
        cluster_path:  (.json  .pt)

    Returns:
        cluster_ids: {unique_idx: cluster_id}
    """
    print(f"[load_cluster_assignments] : {cluster_path}")

    if cluster_path.endswith(".json"):
        with open(cluster_path, "r") as f:
            cluster_ids = json.load(f)
    elif cluster_path.endswith(".pt"):
        cluster_ids = torch.load(cluster_path)
    else:
        raise ValueError(f": {cluster_path}")

    print(f"  : {len(cluster_ids)}")
    cluster_counts = defaultdict(int)
    for cid in cluster_ids.values():
        cluster_counts[cid] += 1

    print(f"  : {len(cluster_counts)}")
    print(f"  : {dict(sorted(cluster_counts.items()))}")

    return cluster_ids

def create_cluster_mapping(
    data_path: str,
    spectral_labels_path: str,
    output_path: str,
):
    """
     ID 

    Args:
        data_path:  JSON 
        spectral_labels_path: 
        output_path: 
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    labels = torch.load(spectral_labels_path)

    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    cluster_ids = {}
    for idx, item in enumerate(data):
        unique_idx = str(item.get("unique_idx", idx))
        if idx < len(labels):
            cluster_ids[unique_idx] = int(labels[idx])
        else:
            cluster_ids[unique_idx] = 0  # 
    with open(output_path, "w") as f:
        json.dump(cluster_ids, f)

    print(f"[create_cluster_mapping] : {output_path}")
    print(f"  : {len(cluster_ids)}")

def get_cluster_distribution(
    cluster_ids: Dict[str, int],
    selected_indices: List[str],
    num_clusters: int = 20,
) -> Dict[str, Any]:
    """


    Args:
        cluster_ids: 
        selected_indices:  ID
        num_clusters: 

    Returns:
        distribution: 
    """
    selected_per_cluster = [0] * num_clusters
    for uid in selected_indices:
        cid = cluster_ids.get(str(uid), 0)
        selected_per_cluster[cid] += 1
    total_selected = len(selected_indices)
    distribution = {
        "total_selected": total_selected,
        "per_cluster": selected_per_cluster,
        "percentages": [c / total_selected * 100 if total_selected > 0 else 0 
                        for c in selected_per_cluster],
    }
    expected = total_selected / num_clusters
    variance = sum((c - expected) ** 2 for c in selected_per_cluster) / num_clusters
    distribution["std"] = variance ** 0.5
    distribution["cv"] = distribution["std"] / expected if expected > 0 else 0

    return distribution

# ====================  ====================

def compute_selection_metrics(
    router,
    dataloader,
    k: int,
    device: torch.device,
) -> Dict[str, float]:
    """


    Args:
        router: DataValveRouter
        dataloader: 
        k: 
        device: 

    Returns:
        metrics: 
    """
    router.eval()

    all_logits = []
    all_selected = []
    all_distances = []

    with torch.no_grad():
        for batch in dataloader:
            clip_image = batch["clip_image_features"].to(device)
            clip_text = batch["clip_text_features"].to(device)

            #  mask
            mask, topk_idx, logits = router.get_selection_mask(
                clip_image, clip_text, k
            )
            distance = router.compute_clip_distance(clip_image, clip_text)

            all_logits.extend(logits.cpu().tolist())
            all_selected.extend(mask.cpu().tolist())
            all_distances.extend(distance.cpu().tolist())

    router.train()
    logits_arr = np.array(all_logits)
    selected_arr = np.array(all_selected)
    distances_arr = np.array(all_distances)

    metrics = {
        "mean_logit": float(logits_arr.mean()),
        "std_logit": float(logits_arr.std()),
        "selection_rate": float(selected_arr.mean()),
        "mean_distance_selected": float((distances_arr * selected_arr).sum() / (selected_arr.sum() + 1e-8)),
        "mean_distance_all": float(distances_arr.mean()),
    }

    return metrics

# ====================  ====================

class TrainingLogger:
    """"""

    def __init__(self, log_dir: str, use_wandb: bool = False):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.logs = []

        os.makedirs(log_dir, exist_ok=True)

        if use_wandb:
            import wandb
            self.wandb = wandb

    def log(self, metrics: Dict[str, float], step: int):
        """"""
        metrics["step"] = step
        self.logs.append(metrics)

        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def save(self):
        """"""
        log_path = os.path.join(self.log_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"[TrainingLogger] : {log_path}")

# ====================  ====================

def plot_selection_distribution(
    cluster_distribution: Dict,
    save_path: str,
):
    """


    Args:
        cluster_distribution: 
        save_path: 
    """
    try:
        import matplotlib.pyplot as plt

        clusters = list(range(len(cluster_distribution["per_cluster"])))
        counts = cluster_distribution["per_cluster"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(clusters, counts, color="steelblue")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Selected Samples")
        ax.set_title("Selection Distribution Across Clusters")
        ax.set_xticks(clusters)
        mean_count = sum(counts) / len(counts)
        ax.axhline(y=mean_count, color="red", linestyle="--", label=f"Mean: {mean_count:.1f}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"[plot_selection_distribution] : {save_path}")
    except ImportError:
        print("[Warning] matplotlib ,")

# ====================  ====================

def validate_data_consistency(
    data_path: str,
    clip_features_path: str,
    cluster_ids_path: Optional[str] = None,
) -> Dict[str, Any]:
    """


    Args:
        data_path:  JSON 
        clip_features_path: CLIP 
        cluster_ids_path:  ID 

    Returns:
        report: 
    """
    report = {"valid": True, "issues": []}
    with open(data_path, "r") as f:
        data = json.load(f)
    data_ids = set(str(item.get("unique_idx", i)) for i, item in enumerate(data))
    clip_features = torch.load(clip_features_path, map_location="cpu")
    feature_ids = set(str(k) for k in clip_features.keys())
    missing_features = data_ids - feature_ids
    if missing_features:
        report["valid"] = False
        report["issues"].append(f" {len(missing_features)}  CLIP ")

    report["data_count"] = len(data_ids)
    report["feature_count"] = len(feature_ids)
    report["coverage"] = len(data_ids & feature_ids) / len(data_ids) * 100
    if cluster_ids_path and os.path.exists(cluster_ids_path):
        with open(cluster_ids_path, "r") as f:
            cluster_ids = json.load(f)
        cluster_id_set = set(str(k) for k in cluster_ids.keys())

        missing_clusters = data_ids - cluster_id_set
        if missing_clusters:
            report["issues"].append(f" {len(missing_clusters)}  ID")

        report["cluster_coverage"] = len(data_ids & cluster_id_set) / len(data_ids) * 100

    print(f"\n[validate_data_consistency] :")
    print(f"  : {report['data_count']}")
    print(f"  : {report['feature_count']}")
    print(f"  : {report['coverage']:.1f}%")
    if "cluster_coverage" in report:
        print(f"  : {report['cluster_coverage']:.1f}%")
    if report["issues"]:
        print(f"  : {report['issues']}")

    return report

# ====================  ====================

def analyze_router_weights(router, save_path: Optional[str] = None):
    """
     Router 

    Args:
        router: DataValveRouter
        save_path: ()
    """
    analysis = {}

    for name, param in router.named_parameters():
        data = param.data.cpu().numpy()
        analysis[name] = {
            "shape": list(data.shape),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }

    print("\n[analyze_router_weights] Router :")
    for name, stats in analysis.items():
        print(f"  {name}:")
        print(f"    Shape: {stats['shape']}")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"  : {save_path}")

    return analysis
