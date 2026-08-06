import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import anndata as ad  # Using pure anndata to avoid scanpy HPC issues during training

import copy
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, classification_report

from sklearn.model_selection import GroupKFold

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ============================================================
# Utilities
# ============================================================
def build_sparse_knn_graph(coords: torch.Tensor, k: int, include_self: bool = True) -> torch.Tensor:
    """
    Uses KD-Tree to find k-nearest neighbors in O(N log N) time and O(Nk) memory.
    Bypasses dense N x N matrix creation completely.
    Returns edge indices of shape [M, 2] where col 0 is member (i) and col 1 is anchor (j).
    """
    coords_np = coords.cpu().numpy()
    n = len(coords_np)
    
    # Ensure k doesn't exceed total nodes
    n_neighbors = min(k + 1, n) 
    
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
    knn.fit(coords_np)
    
    indices = knn.kneighbors(coords_np, return_distance=False)
    
    # <--- CRITICAL FIX: Correct member (i) and anchor (j) orientation --->
    # For a query cell q (anchor), its neighbors are r (members)
    anchor_nodes = np.repeat(np.arange(n), n_neighbors) # j_idx (Anchor / Query cell)
    member_nodes = indices.reshape(-1)                  # i_idx (Member / Neighbor cells)
    
    # Column 0 becomes member (i), Column 1 becomes anchor (j)
    edge_index = np.column_stack((member_nodes, anchor_nodes))
    
    if not include_self:
        # Filter out self-loops if needed
        mask = edge_index[:, 0] != edge_index[:, 1]
        edge_index = edge_index[mask]
        
    return torch.tensor(edge_index, dtype=torch.long, device=coords.device)

def masked_sigmoid(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return probs * mask.float()


def accuracy(logits, y, mask=None):
    """
    Protects against empty splits for accuracy computation.
    """
    if mask is not None:
        if mask.sum().item() == 0:
            return 0.0  # Return 0 or raise ValueError based on preference for evaluation
        logits = logits[mask]
        y = y[mask]
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())

# ============================================================
# Dataset & Config
# ============================================================
class SpatialSample:
    def __init__(self, x: torch.Tensor, coords: torch.Tensor, y: torch.Tensor,
                 raw_coords: np.ndarray, obs_names: np.ndarray, # <--- NEW
                 train_mask: Optional[torch.Tensor] = None, val_mask: Optional[torch.Tensor] = None,
                 test_mask: Optional[torch.Tensor] = None, sample_id: str = "sample",
                 spatial_edge_index: Optional[torch.Tensor] = None,
                 patient_id: str = "unknown"): # <--- NEW: Added patient_id
        self.x = x
        self.coords = coords
        self.raw_coords = raw_coords # <--- NEW
        self.obs_names = obs_names   # <--- NEW
        self.y = y
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.sample_id = sample_id
        self.spatial_edge_index = spatial_edge_index #<======= updated
        self.patient_id = patient_id # <--- NEW: Store patient ID

class SpatialGraphDataset(Dataset):
    def __init__(self, samples: List[SpatialSample]):
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> SpatialSample:
        return self.samples[idx]

def simple_collate(batch: List[SpatialSample]) -> List[SpatialSample]:
    return batch

@dataclass
class Config:
    # ==========================================================
    # Model Architecture
    # ==========================================================
    in_dim: int
    hidden_dim: int = 128
    out_dim: int = 64
    num_classes: int = 5
    class_names: Optional[List[str]] = None

    spatial_k: int = 20
    num_layers: int = 2
    dropout: float = 0.2

    role_dim: int = 64
    spatial_hidden_dim: int = 32
    symmetric_ablation: bool = False

    # ==========================================================
    # Training
    # ==========================================================
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================================
    # Hyperedge Constraints
    # ==========================================================
    min_hyperedge_size: float = 3.0     #4.0 #2.0

    # max_hyperedge_size will be derived automatically
    # from spatial_k + 1

    # ==========================================================
    # Loss Weights
    # ==========================================================
    lambda_sparse: float = 1e-3
    lambda_entropy: float = 1e-3
    lambda_degeneracy: float = 0.05     #0.5 #5e-3
    lambda_overlap: float = 1e-3

    # Optional losses
    lambda_spatial: float = 0.0
    lambda_smooth: float = 0.0

    # ==========================================================
    # Evaluation
    # ==========================================================
    eval_protocol: str = "sample"

# ============================================================
# Model Architecture
# ============================================================
# ============================================================
# Sparse Hypergraph Convolution & Normalization
# ============================================================
def normalize_incidence_sparse(
    i_idx: torch.Tensor, 
    j_idx: torch.Tensor, 
    H_val: torch.Tensor, 
    W_e: torch.Tensor, 
    num_nodes: int, 
    num_edges: int, 
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes node and edge degrees using sparse scatter addition.
    """
    # de = sum_i H[i, e] -> sum over members (i_idx) for each anchor (j_idx)
    de = torch.zeros(num_edges, device=H_val.device).scatter_add_(0, j_idx, H_val).clamp_min(eps)
    
    # dv = sum_e H[i, e] * W_e[e] -> sum over anchors (j_idx) for each member (i_idx)
    weighted_H_val = H_val * W_e[j_idx]
    dv = torch.zeros(num_nodes, device=H_val.device).scatter_add_(0, i_idx, weighted_H_val).clamp_min(eps)
    
    dv_inv_sqrt = dv.pow(-0.5)
    de_inv = de.pow(-1.0)
    
    return dv_inv_sqrt, de_inv

class HypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, H_val: torch.Tensor, W_e: torch.Tensor):
        x = self.dropout(x)
        x = self.lin(x)
        num_nodes = x.size(0)
        num_edges = num_nodes # In Option A, every node acts as an anchor (E = N)
        
        W_e = W_e.clamp_min(1e-8)
        dv_inv_sqrt, de_inv = normalize_incidence_sparse(i_idx, j_idx, H_val, W_e, num_nodes, num_edges)
        
        # 1. Node to Hyperedge Aggregation: Z_e = sum_i (H_ie * W_e * dv_inv_sqrt_i * X_i) * de_inv_e
        x_norm = x * dv_inv_sqrt.unsqueeze(-1)
        msg_to_edge = x_norm[i_idx] * (H_val * W_e[j_idx]).unsqueeze(-1)
        
        edge_feat = torch.zeros(num_edges, x.size(1), device=x.device)
        edge_feat.scatter_add_(0, j_idx.unsqueeze(-1).expand(-1, x.size(1)), msg_to_edge)
        edge_feat = edge_feat * de_inv.unsqueeze(-1)
        
        # 2. Hyperedge to Node Aggregation: X'_i = sum_e (H_ie * dv_inv_sqrt_i * Z_e)
        msg_to_node = edge_feat[j_idx] * H_val.unsqueeze(-1)
        
        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.scatter_add_(0, i_idx.unsqueeze(-1).expand(-1, x.size(1)), msg_to_node)
        out = out * dv_inv_sqrt.unsqueeze(-1)
        
        return out + self.bias

# ============================================================
# Sparse Constructor
# ============================================================
class SpatialHyperedgeConstructor(nn.Module):
    def __init__(self, dim: int, role_dim: int = 64, spatial_hidden_dim: int = 32, symmetric_ablation: bool = False):
        super().__init__()
        self.symmetric_ablation = symmetric_ablation
        
        if self.symmetric_ablation:
            self.shared_proj = nn.Linear(dim, role_dim)
            self.shared_bias = nn.Linear(role_dim, 1)
            
            # <--- CRITICAL FIX 8: Symmetric spatial MLP takes ONLY 1D distance --->
            self.symmetric_spatial_mlp = nn.Sequential(
                nn.Linear(1, spatial_hidden_dim),
                nn.ReLU(),
                nn.Linear(spatial_hidden_dim, 1)
            )
        else:
            self.member_proj = nn.Linear(dim, role_dim)
            self.anchor_proj = nn.Linear(dim, role_dim)
            self.compat = nn.Bilinear(role_dim, role_dim, 1, bias=False)
            self.member_bias = nn.Linear(role_dim, 1)
            self.anchor_bias = nn.Linear(role_dim, 1)
            
            # Asymmetric spatial MLP takes 3D features (rel_x, rel_y, d2)
            self.spatial_mlp = nn.Sequential(
                nn.Linear(3, spatial_hidden_dim),
                nn.ReLU(),
                nn.Linear(spatial_hidden_dim, 1)
            )
        
        self.weight_mlp = nn.Sequential(
            nn.Linear(role_dim, role_dim),
            nn.ReLU(),
            nn.Linear(role_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, spatial_edge_index: torch.Tensor, return_logits: bool = False):
        i_idx = spatial_edge_index[:, 0]
        j_idx = spatial_edge_index[:, 1]
    
        # Compute distance metrics
        rel = coords[i_idx] - coords[j_idx]
        d2 = (rel ** 2).sum(dim=-1, keepdim=True)
    
        if self.symmetric_ablation:
            proj_h = self.shared_proj(h)
            member_sel = proj_h[i_idx]
            anchor_sel = proj_h[j_idx]
            
            compat_logits = (member_sel * anchor_sel).sum(dim=-1)
            member_logits = self.shared_bias(member_sel).squeeze(-1)
            anchor_logits = self.shared_bias(anchor_sel).squeeze(-1)
            
            anchor = proj_h 
            
            # <--- CRITICAL FIX 8: Use absolute distance for symmetric spatial features --->
            distance = torch.sqrt(d2 + 1e-8)
            spatial_logits = self.symmetric_spatial_mlp(distance).squeeze(-1)
            
        else:
            member = self.member_proj(h)
            anchor = self.anchor_proj(h)
            
            member_sel = member[i_idx]
            anchor_sel = anchor[j_idx]
            
            compat_logits = self.compat(member_sel, anchor_sel).squeeze(-1)
            member_logits = self.member_bias(member_sel).squeeze(-1)
            anchor_logits = self.anchor_bias(anchor_sel).squeeze(-1)
            
            # Asymmetric: Keep directional relative coordinates
            spatial_feat = torch.cat([rel, d2], dim=-1)
            spatial_logits = self.spatial_mlp(spatial_feat).squeeze(-1)
    
        sparse_logits = compat_logits + member_logits + anchor_logits + spatial_logits
        raw_sparse_probs = torch.sigmoid(sparse_logits)
        
        self_loop_mask = i_idx == j_idx
        sparse_probs = torch.where(
            self_loop_mask,
            torch.ones_like(raw_sparse_probs),
            raw_sparse_probs
        )
    
        W_e = self.weight_mlp(anchor).squeeze(-1)
    
        return i_idx, j_idx, sparse_probs, W_e

class SpatialNicheHypergraphNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
        )
        
        # <--- MODIFIED: Pass the ablation flag to the constructor --->
        self.constructor = SpatialHyperedgeConstructor(
            dim=cfg.hidden_dim,
            role_dim=cfg.role_dim,
            spatial_hidden_dim=cfg.spatial_hidden_dim,
            symmetric_ablation=cfg.symmetric_ablation
        )
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(cfg.num_layers):
            in_dim = cfg.hidden_dim if i == 0 else cfg.out_dim
            self.layers.append(HypergraphConv(in_dim, cfg.out_dim, cfg.dropout))
            self.norms.append(nn.LayerNorm(cfg.out_dim))
        self.classifier = nn.Linear(cfg.out_dim, cfg.num_classes)

    # Inside SpatialNicheHypergraphNet
    def forward(self, x, coords, spatial_edge_index=None, return_aux=True):
        if spatial_edge_index is None:
            spatial_edge_index = build_sparse_knn_graph(coords, self.cfg.spatial_k, include_self=True)
        h = self.encoder(x)
        
        i_idx, j_idx, H_val, W_e = self.constructor(h, coords, spatial_edge_index, return_logits=return_aux)
        
        # 2. Pass sparse elements directly to the convolution layers
        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, i_idx, j_idx, H_val, W_e)
            h_new = norm(F.relu(h_new))
            h = h + h_new if h.shape[-1] == h_new.shape[-1] else h_new
            
        logits = self.classifier(h)
        out = {"logits": logits, "embeddings": h}
        
        if return_aux:
            out["i_idx"] = i_idx
            out["j_idx"] = j_idx
            out["H_val"] = H_val
            out["hyperedge_weights"] = W_e
            out["num_nodes"] = x.size(0)
            
        return out

# ============================================================
# Sparse Regularizers (Replaces the Dense ones)
# ============================================================
def incidence_sparsity_loss_sparse(i_idx: torch.Tensor, j_idx: torch.Tensor, H_val: torch.Tensor) -> torch.Tensor:
    offdiag = i_idx != j_idx
    if not offdiag.any():
        return torch.tensor(0.0, device=H_val.device)
    return H_val[offdiag].mean()

def incidence_entropy_loss_sparse(i_idx: torch.Tensor, j_idx: torch.Tensor, H_val: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    offdiag = i_idx != j_idx
    if not offdiag.any():
        return torch.tensor(0.0, device=H_val.device)
    p = H_val[offdiag].clamp(min=eps, max=1.0 - eps)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return ent.mean()

# def hyperedge_degeneracy_loss_sparse(j_idx: torch.Tensor, H_val: torch.Tensor, num_anchors: int, min_size: float = 2.0, max_size: float = 50.0) -> torch.Tensor:
#     sizes = torch.zeros(num_anchors, device=H_val.device).scatter_add_(0, j_idx, H_val)
#     penalty_small = torch.relu(min_size - sizes)
#     penalty_large = torch.relu(sizes - max_size)
#     return (penalty_small + penalty_large).mean()

def hyperedge_degeneracy_loss_sparse(
    j_idx: torch.Tensor,
    H_val: torch.Tensor,
    num_anchors: int,
    min_size: float = 2.0,
    max_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Prevents hyperedges from collapsing to extremely small
    or unrealistically large sizes.

    The upper bound should be derived from the spatial
    candidate neighborhood (e.g., spatial_k + 1),
    rather than an arbitrary constant.
    """

    if max_size is None:
        raise ValueError("max_size must be provided.")

    sizes = torch.zeros(
        num_anchors,
        device=H_val.device
    ).scatter_add_(0, j_idx, H_val)

    # <--- CRITICAL FIX 9: Squared hinge penalties for smoother pressure --->
    penalty_small = torch.relu(min_size - sizes).pow(2)
    penalty_large = torch.relu(sizes - max_size).pow(2)

    return (penalty_small + penalty_large).mean()

def hyperedge_overlap_loss_sparse(i_idx: torch.Tensor, j_idx: torch.Tensor, H_val: torch.Tensor, num_nodes: int, num_edges: int) -> torch.Tensor:
    """
    Computes overlap penalty using a sparse, memory-efficient O(N) implementation.
    Subsamples anchors to avoid O(E^2) matrix operations.
    """
    if num_edges <= 1:
        return torch.tensor(0.0, device=H_val.device)
    
    num_samples = min(num_edges, 256)
    sampled_anchors = torch.randperm(num_edges, device=H_val.device)[:num_samples]
    
    # Create a small dense subset (N x 256) which is strictly O(N) memory
    H_sub = torch.zeros(num_nodes, num_samples, device=H_val.device)
    
    for local_j, orig_j in enumerate(sampled_anchors):
        mask = (j_idx == orig_j)
        H_sub[i_idx[mask], local_j] = H_val[mask]
        
    Hn = H_sub / H_sub.sum(dim=0, keepdim=True).clamp_min(1e-8)
    overlap = Hn.t() @ Hn
    eye = torch.eye(num_samples, device=H_val.device, dtype=torch.bool)
    return overlap[~eye].mean()

# ============================================================
# Regularizers & Loss
# ============================================================

def supervised_loss(logits, y, mask, class_weights=None):
    """
    Computes class-balanced cross entropy and protects against empty splits.
    class_weights should be a tensor of shape (num_classes,) moved to the correct device.
    """
    if mask is not None:
        if mask.sum().item() == 0:
            raise ValueError("Supervised loss received an empty mask.")
        logits = logits[mask]
        y = y[mask]
    return F.cross_entropy(logits, y, weight=class_weights)




def hyperedge_size_var_loss(H: torch.Tensor) -> torch.Tensor:
    sizes = H.sum(dim=0)
    if sizes.numel() <= 1:
        return torch.tensor(0.0, device=H.device)
    return sizes.var(unbiased=False)



def local_smoothness_loss_sparse(emb: torch.Tensor, edge_index: torch.Tensor, enabled: bool = False):
    if not enabled:
        return torch.tensor(0.0, device=emb.device)
    
    i_idx, j_idx = edge_index[:, 0], edge_index[:, 1]
    # Filter self-loops for smoothness penalty
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    
    if len(i_idx) == 0:
        return torch.tensor(0.0, device=emb.device)
        
    sq_dists = ((emb[i_idx] - emb[j_idx]) ** 2).sum(dim=-1)
    return sq_dists.mean()


# ============================================================
# Core Objective and Forward Pass
# ============================================================
def training_objective(out, sample, cfg, class_weights=None):
    """
    Computes objective strictly on training data using SPARSE tensors.
    """
    train_mask = sample.train_mask
    sup = supervised_loss(out["logits"], sample.y, train_mask, class_weights=class_weights)
    
    # Retrieve sparse elements
    i_idx, j_idx = out["i_idx"], out["j_idx"]
    H_val = out["H_val"]
    num_nodes = out["num_nodes"]
    num_edges = num_nodes # Since every node is an anchor in Option A
    
    # Compute sparse regularizers
    sparse_loss = incidence_sparsity_loss_sparse(i_idx, j_idx, H_val)
    entropy = incidence_entropy_loss_sparse(i_idx, j_idx, H_val)

    # <--- CRITICAL FIX 9: Use configured min_size --->
    degeneracy = hyperedge_degeneracy_loss_sparse(
        j_idx=j_idx,
        H_val=H_val,
        num_anchors=num_edges,
        min_size=cfg.min_hyperedge_size, 
        max_size=cfg.spatial_k + 1
    )

    overlap = hyperedge_overlap_loss_sparse(i_idx, j_idx, H_val, num_nodes, num_edges)
    
    loss = (
    sup
    + cfg.lambda_sparse * sparse_loss
    + cfg.lambda_entropy * entropy
    + cfg.lambda_degeneracy * degeneracy
    + cfg.lambda_overlap * overlap
)
    
    stats = {
        "loss": float(loss.item()),
        "sup": float(sup.item()),
        "degeneracy": float(degeneracy.item())
    }
    return loss, stats


# ============================================================
# Train / evaluate Helpers
# ============================================================
def run_epoch(model, loader, optimizer, cfg, split: str = "train", class_weights=None, 
              save_niche_dir: Optional[str] = None, fold: int = 0, seed: int = 0,
              feature_mean: Optional[torch.Tensor] = None, feature_std: Optional[torch.Tensor] = None): # <--- NEW Arguments
    is_train = (split == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_sup = 0.0
    total_degeneracy = 0.0
    
    all_preds = []
    all_labels = []
    
    count = 0

    for batch in loader:
        for sample in batch:
            
            # <--- CRITICAL FIX 1: Apply Inductive Feature Normalization --->
            x_val = sample.x.to(cfg.device)
            if feature_mean is not None and feature_std is not None:
                x_val = (x_val - feature_mean) / feature_std
                
            moved = SpatialSample(
                x=x_val, # Use the normalized features
                coords=sample.coords.to(cfg.device),
                y=sample.y.to(cfg.device),
                raw_coords=sample.raw_coords, 
                obs_names=sample.obs_names,   
                train_mask=sample.train_mask.to(cfg.device) if sample.train_mask is not None else None,
                val_mask=sample.val_mask.to(cfg.device) if sample.val_mask is not None else None,
                test_mask=sample.test_mask.to(cfg.device) if sample.test_mask is not None else None,
                sample_id=sample.sample_id,
                spatial_edge_index=sample.spatial_edge_index.to(cfg.device) if sample.spatial_edge_index is not None else None,
                patient_id=sample.patient_id 
            )

            with torch.set_grad_enabled(is_train):
                out = model(moved.x, moved.coords, spatial_edge_index=moved.spatial_edge_index, return_aux=is_train or (save_niche_dir is not None))
                mask = getattr(moved, f"{split}_mask")

                if is_train:
                    loss, stats = training_objective(out, moved, cfg, class_weights)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += stats["loss"]
                    total_sup += stats["sup"]
                    total_degeneracy += stats.get("degeneracy", 0.0)
                else:
                    sup_loss = supervised_loss(out["logits"], moved.y, mask, class_weights)
                    total_loss += float(sup_loss.item())
                    total_sup += float(sup_loss.item())
                    
                    # <--- CRITICAL FIX 2: Simplified Niche Export (No Masking) --->
                    if save_niche_dir is not None and mask is not None and mask.sum().item() > 0:
                        
                        # Directly save the complete core output. No masks needed since test cores are fully held-out.
                        save_dict = {
                            "core_id": sample.sample_id,
                            "patient_id": sample.patient_id,
                            "fold": fold,
                            "seed": seed,
                            
                            # Node-level features (Full core)
                            "cell_ids": sample.obs_names,
                            "coordinates": moved.coords.detach().cpu().numpy(),
                            "raw_coordinates": sample.raw_coords,
                            "true_cell_types": moved.y.detach().cpu().numpy(),
                            "predicted_labels": out["logits"].argmax(dim=-1).detach().cpu().numpy(),
                            "embeddings": out["embeddings"].detach().cpu().numpy(),
                            
                            # Graph/Edge-level features (Full core)
                            "member_indices": out["i_idx"].detach().cpu().numpy(),
                            "anchor_indices": out["j_idx"].detach().cpu().numpy(),
                            "membership_weights": out["H_val"].detach().cpu().numpy(),
                            "hyperedge_weights": out["hyperedge_weights"].detach().cpu().numpy(),
                        }
                        
                        file_name = f"niche_data_{sample.sample_id}_fold_{fold}_seed_{seed}.pt"
                        torch.save(save_dict, os.path.join(save_niche_dir, file_name))

                if mask is not None and mask.sum().item() > 0:
                    logits_masked = out["logits"][mask]
                    y_masked = moved.y[mask]
                    all_preds.append(logits_masked.argmax(dim=-1).cpu())
                    all_labels.append(y_masked.cpu())

                count += 1

    if len(all_preds) > 0:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        
        acc = float((y_pred == y_true).mean())
        
        # <--- FIX: Supply fixed labels and target names as per advisor feedback --->
        labels_array = np.arange(cfg.num_classes)
        
        macro_f1 = f1_score(y_true, y_pred, labels=labels_array, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_prec = precision_score(y_true, y_pred, labels=labels_array, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, labels=labels_array, average='macro', zero_division=0)
        
        class_report = None
        if split == "test":
            class_report = classification_report(
                y_true, 
                y_pred, 
                labels=labels_array,
                target_names=cfg.class_names, # Uses the real names (e.g., 'B-cell', 'Tumor')
                output_dict=True, 
                zero_division=0
            )
            
    
    else:
        acc, macro_f1, bal_acc, macro_prec, macro_rec = 0.0, 0.0, 0.0, 0.0, 0.0
        class_report = None

    avg_stats = {
        f"{split}_loss": total_loss / max(count, 1),
        f"{split}_sup": total_sup / max(count, 1),
        f"{split}_acc": acc,
        f"{split}_macro_f1": macro_f1,
        f"{split}_bal_acc": bal_acc,
        f"{split}_precision_macro": macro_prec, # <--- FIX: Changed to match main loop
        f"{split}_recall_macro": macro_rec,     # <--- FIX: Changed to match main loop
        f"{split}_class_report": class_report   
    }
    
    if is_train:
        avg_stats[f"{split}_degeneracy"] = total_degeneracy / max(count, 1)

    return avg_stats

# ============================================================
# Data Loader (From Cleaned .h5ad)
# ============================================================
def load_prepared_data(h5ad_path: str, cfg: Config) -> List[SpatialSample]:
    print(f"Loading preprocessed data from: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    
    cfg.in_dim = adata.shape[1]
    
    # <--- FIX: Extract and save exact class names from the dataset --->
    if 'class_names' in adata.uns:
        cfg.class_names = list(adata.uns['class_names'])
        cfg.num_classes = len(cfg.class_names)
    else:
        cfg.num_classes = len(np.unique(adata.obs['numeric_labels']))
        cfg.class_names = [f"Class_{i}" for i in range(cfg.num_classes)]
        
    print(f"Dataset Details -> Features: {cfg.in_dim}, Classes: {cfg.num_classes}")
    print(f"Class Names: {cfg.class_names}")

    # <--- FEEDBACK v7 (Point 3): Validate label-to-name mapping --->
    observed_labels = np.unique(adata.obs["numeric_labels"])
    expected_labels = np.arange(cfg.num_classes)
    if not np.all(np.isin(observed_labels, expected_labels)):
        raise ValueError("numeric_labels are incompatible with class_names.")
    
    samples = []
    for core_id in adata.obs['core'].unique():
        core_adata = adata[adata.obs['core'] == core_id].copy()
        n_nodes = core_adata.shape[0]
        
        if n_nodes < cfg.spatial_k + 1: continue

        # 1. Load Feature Matrix (Without per-core Z-scoring - Inductive approach)
        x_tensor = torch.tensor(core_adata.X.toarray() if hasattr(core_adata.X, "toarray") else core_adata.X, dtype=torch.float32)
        
        # <--- REMOVED: per-core normalization --->
        # x_tensor = (x_tensor - x_tensor.mean(dim=0, keepdim=True)) / x_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
            
        # # 1. Feature Normalization (Z-score per core)
        # x_tensor = torch.tensor(core_adata.X.toarray() if hasattr(core_adata.X, "toarray") else core_adata.X, dtype=torch.float32)
        # x_tensor = (x_tensor - x_tensor.mean(dim=0, keepdim=True)) / x_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)

        # <--- CRITICAL FIX 2: Construct kNN using raw physical coordinates --->
        coords_val = core_adata.obs[['x_coordinate', 'y_coordinate']].values if 'x_coordinate' in core_adata.obs.columns else core_adata.obs[['x', 'y']].values
        raw_coords = torch.tensor(coords_val, dtype=torch.float32)

        # Generate sparse KD-Tree graph directly using raw physical coordinates
        spatial_edge_index = build_sparse_knn_graph(raw_coords, cfg.spatial_k, include_self=True).cpu()

        # Isotropic normalization for the neural network
        coords_centered = raw_coords - raw_coords.mean(dim=0, keepdim=True)
        coord_scale = coords_centered.std().clamp_min(1e-6)
        coords_tensor = coords_centered / coord_scale

        # 3. Labels and Masks
        y_tensor = torch.tensor(core_adata.obs['numeric_labels'].values, dtype=torch.long)
        
        all_true = torch.ones(n_nodes, dtype=torch.bool)
        train_mask = all_true.clone()
        val_mask = all_true.clone()
        test_mask = all_true.clone()
        
        # <--- CRITICAL FIX 3: Verify patient consistency within each core --->
        patient_ids = core_adata.obs["patient_id"].dropna().astype(str).unique()
        if len(patient_ids) != 1:
            raise ValueError(f"Core {core_id} has {len(patient_ids)} patient IDs: {patient_ids}")
        p_id = patient_ids[0]

        samples.append(
            SpatialSample(
                x=x_tensor,
                coords=coords_tensor,
                y=y_tensor,
                raw_coords=raw_coords.numpy(),            # <--- NEW: Raw physical coordinates
                obs_names=core_adata.obs_names.to_numpy(),# <--- NEW: Original cell barcodes
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                sample_id=f"core_{core_id}",
                spatial_edge_index=spatial_edge_index,
                patient_id=p_id, 
            )
        )
        
    return samples

def assign_split_masks(samples: List[SpatialSample], split_name: str) -> None:
    """
    Held-out-core evaluation:
    use all nodes inside each core for that split.
    """
    for sample in samples:
        n = sample.y.size(0)
        false_mask = torch.zeros(n, dtype=torch.bool)
        true_mask = torch.ones(n, dtype=torch.bool)

        if split_name == "train":
            sample.train_mask = true_mask
            sample.val_mask = false_mask
            sample.test_mask = false_mask
        elif split_name == "val":
            sample.train_mask = false_mask
            sample.val_mask = true_mask
            sample.test_mask = false_mask
        elif split_name == "test":
            sample.train_mask = false_mask
            sample.val_mask = false_mask
            sample.test_mask = true_mask
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

# ============================================================
# Main
# ============================================================

def main():
    # Initial setup
    torch.autograd.set_detect_anomaly(False)
    
    output_dir = "/project/output"
    os.makedirs(output_dir, exist_ok=True)

    cfg = Config(in_dim=0, num_classes=0)

    # 1. Load data ONLY ONCE to save time
    CLEANED_H5AD_FILE = "/project/data.h5ad"
    all_samples = load_prepared_data(CLEANED_H5AD_FILE, cfg)
    print(f"Successfully prepared {len(all_samples)} tissue graphs (cores).")

    # 2. Define multiple independent seeds for initialization robustness
    SEEDS = [0, 7, 42, 123, 999]
    
    # Extract Patient Groups
    from collections import Counter
    groups = np.asarray([sample.patient_id for sample in all_samples])
    unique_patient_ids = sorted(np.unique(groups).tolist())

    # <--- FEEDBACK v7 (Point 6): Require at least three patients --->
    if len(unique_patient_ids) < 3:
        raise ValueError("Rotating LOPO evaluation requires at least three patients.")

    # <--- FIX: Explicitly restrict to Leave-One-Patient-Out (LOPO) evaluation --->
    # Removed the min(5, ...) limit so it scales correctly for datasets with > 5 patients
    n_splits = len(unique_patient_ids) 
    print(f"Total Unique Patients: {len(unique_patient_ids)} | Setting K-Fold (LOPO) to: {n_splits}")
    
    splitter = GroupKFold(n_splits=n_splits)

    # Materialize the folds once so they remain identical across all seeds.
    outer_folds = list(
        splitter.split(
            np.arange(len(all_samples)),
            groups=groups
        )
    )

    # <--- FIX: Unconditional rotation mapping (Works perfectly for LOPO) --->
    validation_patient_for_test = {
        test_patient: unique_patient_ids[
            (position + 1) % len(unique_patient_ids)
        ]
        for position, test_patient in enumerate(unique_patient_ids)
    }

    # <--- NEW: Expanded Dictionaries for Patient Metrics --->
    results_f1 = {seed: [] for seed in SEEDS}
    results_bal_acc = {seed: [] for seed in SEEDS}
    results_prec = {seed: [] for seed in SEEDS} # <--- NEW: Seed-level Precision
    results_rec = {seed: [] for seed in SEEDS}  # <--- NEW: Seed-level Recall
    
    fold_patient_f1 = {f: [] for f in range(n_splits)} 
    fold_patient_bal_acc = {f: [] for f in range(n_splits)}
    fold_patient_precision = {f: [] for f in range(n_splits)}
    fold_patient_recall = {f: [] for f in range(n_splits)}
    fold_class_reports = {f: [] for f in range(n_splits)}
    test_patients_record = {}

    # <--- FEEDBACK v7 (Point 5): Lists to save publication-ready tables --->
    csv_patient_overall = []
    csv_patient_class = []
    csv_seed_results = []       # <--- NEW: For seed-level results
    csv_fold_assignments = []   # <--- NEW: For fold assignments


    # Counters to verify perfect rotation at the end
    test_patient_counts = Counter()
    val_patient_counts = Counter()

    # 3. Outer Loop: Fixed Data Partitions Grouped by Patient
    for fold, (train_val_idx, test_idx) in enumerate(outer_folds):
        print(f"\n{'='*60}")
        print(f"### STARTING FIXED FOLD {fold + 1}/{n_splits} (Grouped by Patient) ###")
        print(f"{'='*60}")

        test_patients = sorted(np.unique(groups[test_idx]).tolist())

        if len(test_patients) != 1:
            raise ValueError(
                "The rotating validation procedure requires exactly "
                f"one test patient per fold, but fold {fold + 1} has "
                f"{test_patients}."
            )

        test_patient = test_patients[0]
        val_patient = validation_patient_for_test[test_patient]

        # Assign designated validation patient and training patients
        val_idx = np.asarray([
            idx for idx in train_val_idx
            if groups[idx] == val_patient
        ])

        train_idx = np.asarray([
            idx for idx in train_val_idx
            if groups[idx] != val_patient
        ])

        train_samples = [all_samples[idx] for idx in train_idx]
        val_samples = [all_samples[idx] for idx in val_idx]
        test_samples = [all_samples[idx] for idx in test_idx]

        

        # <--- FIX: Convert NumPy strings to standard Python strings --->
        train_patient_ids = set(str(p) for p in groups[train_idx])
        val_patient_ids = set(str(p) for p in groups[val_idx])
        test_patient_ids = set(str(p) for p in groups[test_idx])

        # <--- FIX (Point 7): Replace assert with explicit exceptions for critical leakage validation --->
        if not train_patient_ids.isdisjoint(val_patient_ids):
            raise ValueError("Leakage between Train and Val patients.")
        
        if not train_patient_ids.isdisjoint(test_patient_ids):
            raise ValueError("Leakage between Train and Test patients.")
            
        if not val_patient_ids.isdisjoint(test_patient_ids):
            raise ValueError("Leakage between Val and Test patients.")

        if len(val_patient_ids) != 1:
            raise ValueError("Validation split must contain exactly 1 patient.")
            
        if len(test_patient_ids) != 1:
            raise ValueError("Test split must contain exactly 1 patient.")
            
        if len(train_idx) == 0:
            raise ValueError("Train split is empty.")
            
        if len(val_idx) == 0:
            raise ValueError("Val split is empty.")
            
        if len(test_idx) == 0:
            raise ValueError("Test split is empty.")

        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"Train patients: {sorted(train_patient_ids)}")
        print(f"Val patients:   {sorted(val_patient_ids)}")
        print(f"Test patients:  {sorted(test_patient_ids)}")
        print(f"-------------------------------\n")

        # <--- NEW: Save Fold Assignments for CSV --->
        csv_fold_assignments.append({
            "Fold": fold + 1,
            "Train_Patients": ", ".join(sorted(train_patient_ids)),
            "Val_Patient": ", ".join(sorted(val_patient_ids)),
            "Test_Patient": ", ".join(sorted(test_patient_ids))
        })

        # Record patient distribution for tracking
        test_patients_record[fold] = sorted(list(test_patient_ids))
        test_patient_counts.update(test_patient_ids)
        val_patient_counts.update(val_patient_ids)

        # Calculate Inductive Statistics from Training Cores ONLY
        train_features = torch.cat([s.x for s in train_samples], dim=0)
        fold_feature_mean = train_features.mean(dim=0, keepdim=True).to(cfg.device)
        fold_feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6).to(cfg.device)
        del train_features 

        if cfg.eval_protocol == "sample":
            assign_split_masks(train_samples, "train")
            assign_split_masks(val_samples, "val")
            assign_split_masks(test_samples, "test")

        train_loader = DataLoader(SpatialGraphDataset(train_samples), batch_size=1, shuffle=True, collate_fn=simple_collate)
        val_loader = DataLoader(SpatialGraphDataset(val_samples), batch_size=1, shuffle=False, collate_fn=simple_collate)
        test_loader = DataLoader(SpatialGraphDataset(test_samples), batch_size=1, shuffle=False, collate_fn=simple_collate)

        # Calculate Class Weights
        train_labels = torch.cat([s.y[s.train_mask] for s in train_samples])
        class_counts = torch.bincount(train_labels, minlength=cfg.num_classes).float().clamp(min=1.0)
        class_weights = (class_counts.sum() / (cfg.num_classes * class_counts)).to(cfg.device)

        # 4. Inner Loop: Iterate over Model Initialization Seeds
        for seed in SEEDS:
            print(f"\n{'-'*40}")
            print(f" Fold {fold + 1} | Training with Seed {seed} ")
            print(f"{'-'*40}")
            
            set_seed(seed)
            
            model = SpatialNicheHypergraphNet(cfg).to(cfg.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            
            best_val = -1.0
            best_state = None
            best_epoch = 0

            # Training Loop
            for epoch in range(1, cfg.epochs + 1):
                train_stats = run_epoch(model, train_loader, optimizer, cfg, split="train", class_weights=class_weights, 
                                        feature_mean=fold_feature_mean, feature_std=fold_feature_std)
                val_stats = run_epoch(model, val_loader, optimizer, cfg, split="val", class_weights=class_weights,
                                      feature_mean=fold_feature_mean, feature_std=fold_feature_std)

                if val_stats["val_macro_f1"] > best_val:
                    best_val = val_stats["val_macro_f1"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train_loss={train_stats['train_loss']:.4f} | "
                        f"val_loss={val_stats['val_loss']:.4f} | "
                        f"val_macro_f1={val_stats['val_macro_f1']:.4f} | val_bal_acc={val_stats['val_bal_acc']:.4f}"
                    )

            if best_state is not None:
                model.load_state_dict(best_state)
                model_save_path = os.path.join(output_dir, f"best_model_brcr_fold_{fold + 1}_seed_{seed}.pth")
                torch.save(best_state, model_save_path)
                
                test_stats = run_epoch(
                    model, test_loader, optimizer, cfg, split="test", 
                    class_weights=class_weights, 
                    save_niche_dir=output_dir, 
                    fold=fold + 1,             
                    seed=seed,
                    feature_mean=fold_feature_mean,
                    feature_std=fold_feature_std
                )
               
                print(
                    f"Fold {fold + 1} | Seed {seed} | Best Epoch: {best_epoch} | "
                    f"Macro-F1={test_stats['test_macro_f1']:.4f} | "
                    f"Balanced Acc={test_stats['test_bal_acc']:.4f} | "
                    f"Macro-Precision={test_stats['test_precision_macro']:.4f} | "
                    f"Fixed-Class Macro-Recall={test_stats['test_recall_macro']:.4f}"
                )
                # <--- FIX: Direct key access to avoid silent zero substitution --->
                results_f1[seed].append(test_stats['test_macro_f1'])
                results_bal_acc[seed].append(test_stats['test_bal_acc'])
                results_prec[seed].append(test_stats['test_precision_macro']) # <--- NEW: Seed-level Precision
                results_rec[seed].append(test_stats['test_recall_macro'])     # <--- NEW: Seed-level Recall
                
                # <--- UPDATED: Append to patient-level tracking (Direct Access) --->
                fold_patient_f1[fold].append(test_stats['test_macro_f1'])
                fold_patient_bal_acc[fold].append(test_stats['test_bal_acc'])
                fold_patient_precision[fold].append(test_stats['test_precision_macro'])
                fold_patient_recall[fold].append(test_stats['test_recall_macro'])
                fold_class_reports[fold].append(test_stats['test_class_report'])

                # <--- NEW: Save Seed-Level Results for CSV --->
                csv_seed_results.append({
                    "Fold": fold + 1,
                    "Test_Patient": ", ".join(sorted(test_patient_ids)),
                    "Seed": seed,
                    "Macro-F1": test_stats['test_macro_f1'],
                    "Bal-Acc": test_stats['test_bal_acc'],
                    "Macro-Precision": test_stats['test_precision_macro'],
                    "Fixed-Class-Recall": test_stats['test_recall_macro']
                })

# <--- MISSING PART: Aggregate and Print Per-Class Metrics across 5 seeds for this fold --->
        # <--- FEEDBACK v7 (Point 4 & 5): Add support column and save to CSV list --->
        print(f"\n{'='*90}")
        print(f" Aggregated Per-Class Report | Fold {fold + 1} | Patient(s) {test_patients_record[fold]}")
        print(f"{'='*90}")
        print(f"{'Class Name':<25} | {'Precision':<15} | {'Recall':<15} | {'F1-Score':<15} | {'Support'}")
        print("-" * 90)
        
        c_names = cfg.class_names if cfg.class_names else [str(i) for i in range(cfg.num_classes)]
        
        for cls_name in c_names:
            p_list = [rep[cls_name]['precision'] for rep in fold_class_reports[fold] if cls_name in rep]
            r_list = [rep[cls_name]['recall'] for rep in fold_class_reports[fold] if cls_name in rep]
            f_list = [rep[cls_name]['f1-score'] for rep in fold_class_reports[fold] if cls_name in rep]
            
            # Support consistency check
            support_values = [rep[cls_name]["support"] for rep in fold_class_reports[fold] if cls_name in rep]
            if len(support_values) > 0:
                if len(set(support_values)) != 1:
                    raise ValueError(f"Per-class support changed across seeds for {cls_name}.")
                support = int(support_values[0])
            else:
                support = 0
            
            p_m, p_s = np.mean(p_list), np.std(p_list, ddof=1) if len(p_list) > 1 else 0
            r_m, r_s = np.mean(r_list), np.std(r_list, ddof=1) if len(r_list) > 1 else 0
            f_m, f_s = np.mean(f_list), np.std(f_list, ddof=1) if len(f_list) > 1 else 0
            
            print(f"{cls_name:<25} | {p_m:.4f} ± {p_s:.4f} | {r_m:.4f} ± {r_s:.4f} | {f_m:.4f} ± {f_s:.4f} | {support}")
            
            # Save for CSV
            csv_patient_class.append({
                "Fold": fold + 1,
                "Test_Patient": str(test_patients_record[fold]),
                "Class": cls_name,
                "Precision_Mean": p_m, "Precision_SD": p_s,
                "Recall_Mean": r_m, "Recall_SD": r_s,
                "F1_Mean": f_m, "F1_SD": f_s,
                "Support": support
            })
        print("-" * 90 + "\n")

    # Evaluate Metrics across Seeds
    seed_cv_f1 = [np.mean(results_f1[seed]) for seed in SEEDS]
    seed_cv_bal_acc = [np.mean(results_bal_acc[seed]) for seed in SEEDS]
    
    # <--- NEW: Calculate seed-level tracking for precision and recall --->
    seed_cv_prec = [np.mean(results_prec[seed]) for seed in SEEDS] 
    seed_cv_rec = [np.mean(results_rec[seed]) for seed in SEEDS]   

    mean_f1 = np.mean(seed_cv_f1)
    std_f1 = np.std(seed_cv_f1, ddof=1)

    mean_bal_acc = np.mean(seed_cv_bal_acc)
    std_bal_acc = np.std(seed_cv_bal_acc, ddof=1)
    
    # <--- NEW: Calculate overall precision and recall means & stds --->
    mean_prec = np.mean(seed_cv_prec)
    std_prec = np.std(seed_cv_prec, ddof=1)
    
    mean_rec = np.mean(seed_cv_rec)
    std_rec = np.std(seed_cv_rec, ddof=1)

    total_runs = n_splits * len(SEEDS)
    print(f"\n{'='*70}")
    print(f" 🚀 FINAL ROBUSTNESS REPORT ({n_splits} Folds × {len(SEEDS)} Seeds = {total_runs} Runs) 🚀")
    print(f"{'='*70}")
    
    # Patient-level detailed metrics
    print("--- PATIENT-LEVEL VARIABILITY ---")
    for f in range(n_splits):
        f1_mean, f1_std = np.mean(fold_patient_f1[f]), np.std(fold_patient_f1[f], ddof=1)
        acc_mean, acc_std = np.mean(fold_patient_bal_acc[f]), np.std(fold_patient_bal_acc[f], ddof=1)
        prec_mean, prec_std = np.mean(fold_patient_precision[f]), np.std(fold_patient_precision[f], ddof=1)
        rec_mean, rec_std = np.mean(fold_patient_recall[f]), np.std(fold_patient_recall[f], ddof=1)
        
        # Save for CSV
        csv_patient_overall.append({
            "Fold": f + 1, "Test_Patient": str(test_patients_record[f]),
            "Macro-F1_Mean": f1_mean, "Macro-F1_SD": f1_std,
            "Bal-Acc_Mean": acc_mean, "Bal-Acc_SD": acc_std,
            "Macro-Precision_Mean": prec_mean, "Macro-Precision_SD": prec_std,
            "Fixed-Class-Recall_Mean": rec_mean, "Fixed-Class-Recall_SD": rec_std
        })
        
        print(f"Patient(s) {test_patients_record[f]} (Fold {f+1}):")
        print(f"  Macro-F1:                 {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  Bal-Acc:                  {acc_mean:.4f} ± {acc_std:.4f}")
        # <--- FEEDBACK v7 (Point 2): Rename labels --->
        print(f"  Macro-Precision:          {prec_mean:.4f} ± {prec_std:.4f}")
        print(f"  Fixed-Class Macro-Recall: {rec_mean:.4f} ± {rec_std:.4f}\n")
    print("---------------------------------")
    
    # <--- FEEDBACK v7 (Point 2): Rename overall labels --->
    print(f"Overall Test Macro-Precision:          {mean_prec:.4f} ± {std_prec:.4f} (Seed-based SD)")
    print(f"Overall Test Fixed-Class Macro-Recall: {mean_rec:.4f} ± {std_rec:.4f} (Seed-based SD)")
    print(f"Overall Test Macro-F1:                 {mean_f1:.4f} ± {std_f1:.4f} (Seed-based SD)")
    print(f"Overall Test Balanced Acc:             {mean_bal_acc:.4f} ± {std_bal_acc:.4f} (Seed-based SD)")
    print(f"{'='*70}\n")

    # Verification of exact rotation logic
    print("Test-patient counts:", dict(test_patient_counts))
    print("Validation-patient counts:", dict(val_patient_counts))

    if not all(test_patient_counts[p] == 1 for p in unique_patient_ids):
        raise ValueError("Rotation Failed: Every patient must be tested exactly once.")

    if not all(val_patient_counts[p] == 1 for p in unique_patient_ids):
        raise ValueError("Rotation Failed: Every patient must be used for validation exactly once.")

    print("✅ PERFECT ROTATION CONFIRMED: Each patient tested & validated exactly once.")

    # <--- FEEDBACK v7 (Point 5): Export DataFrames to CSV --->
    df_overall = pd.DataFrame(csv_patient_overall)
    df_class = pd.DataFrame(csv_patient_class)
    df_seed = pd.DataFrame(csv_seed_results)           # <--- NEW
    df_fold = pd.DataFrame(csv_fold_assignments)       # <--- NEW
    
    df_overall.to_csv(os.path.join(output_dir, "patient_overall_metrics.csv"), index=False)
    df_class.to_csv(os.path.join(output_dir, "patient_by_class_metrics.csv"), index=False)
    df_seed.to_csv(os.path.join(output_dir, "seed_level_results.csv"), index=False)      # <--- NEW
    df_fold.to_csv(os.path.join(output_dir, "fold_assignments.csv"), index=False)        # <--- NEW
    
    print(f"✅ Saved all 4 publication-ready CSV reports to: {output_dir}")
if __name__ == "__main__":
    main()