import os
import glob
import torch
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.stats import entropy, spearmanr
from statsmodels.stats.multitest import fdrcorrection
from scipy.spatial import ConvexHull, cKDTree, QhullError
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import re
from pathlib import Path

def export_niche_celltype_counts(
    hard_anchor,
    hard_member,
    valid_anchor_indices,
    adata,
    output_csv="anchored_niche_celltype_counts.csv",
    patient_id=None,
    core_id=None,
    seed=None,
    celltype_key="cell_type",
):
    """
    Create one row per anchored niche using 1D sparse arrays.
    """
    if "spatial" not in adata.obsm:
        raise KeyError('Coordinates were not found in adata.obsm["spatial"].')

    if celltype_key not in adata.obs.columns:
        raise KeyError(f'Cell-type column "{celltype_key}" was not found.')

    coordinates = np.asarray(adata.obsm["spatial"])
    barcodes = adata.obs_names.astype(str).to_numpy()
    celltypes = (
        adata.obs[celltype_key]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
        .to_numpy()
    )

    all_celltypes = sorted(pd.unique(celltypes))
    celltype_columns = {}
    used_columns = set()

    for celltype in all_celltypes:
        import re
        base = re.sub(r"[^A-Za-z0-9]+", "_", celltype).strip("_")
        base = base or "Unknown"
        column = f"count_{base}"
        suffix = 2
        while column in used_columns:
            column = f"count_{base}_{suffix}"
            suffix += 1
        celltype_columns[celltype] = column
        used_columns.add(column)

    records = []

    for anchor_idx in valid_anchor_indices:
        member_indices = hard_member[hard_anchor == anchor_idx]
        member_celltypes = celltypes[member_indices]
        counts = pd.Series(member_celltypes).value_counts()
        
        record = {
            "patient_id": patient_id,
            "core_id": core_id,
            "seed": seed,
            "anchor_index": int(anchor_idx),
            "anchor_barcode": barcodes[anchor_idx],
            "anchor_celltype": celltypes[anchor_idx],
            "anchor_x": float(coordinates[anchor_idx, 0]),
            "anchor_y": float(coordinates[anchor_idx, 1]),
            "niche_size": int(len(member_indices)),
        }
        
        for celltype in all_celltypes:
            record[celltype_columns[celltype]] = int(counts.get(celltype, 0))
            
        records.append(record)

    result = pd.DataFrame(records)

    if len(result) > 0:
        count_columns = list(celltype_columns.values())
        count_sum = result[count_columns].sum(axis=1)
        if not np.array_equal(count_sum.to_numpy(), result["niche_size"].to_numpy()):
            raise RuntimeError("Cell-type counts do not sum to niche_size for every niche.")

    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result):,} anchored niches to {output_csv}")
    
    return result, celltype_columns

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class HyperNicheEvaluator:
    def __init__(self, input_dir: str, output_dir: str, thresholds: list = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.thresholds = thresholds if thresholds is not None else [0.01, 0.02, 0.05, 0.10]  #[0.05, 0.10, 0.20]
        
        self.MIN_HYPEREDGE_WEIGHT = 0.05
        self.PRIMARY_MEMBERSHIP_THRESHOLD = 0.10
        self.EXPECTED_SEEDS = 5
        # self.EXPECTED_SEED_SET = {0, 1, 2, 3, 4}
        self.EXPECTED_SEED_SET = {0, 7, 42, 123, 999}
        self.MAX_REDUNDANCY_RADIUS = 200.0
        self.dataset = []
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def load_and_validate_data(self):
        print(f"Scanning directory: {self.input_dir}")
        search_pattern = os.path.join(self.input_dir, "niche_data_*.pt")
        file_paths = sorted(glob.glob(search_pattern))
        
        if not file_paths:
            raise FileNotFoundError("No .pt files found!")

        required_fields = [
            "patient_id", "core_id", "fold", "seed",
            "cell_ids", "raw_coordinates", "true_cell_types",
            "hyperedge_weights", "membership_weights",
            "member_indices", "anchor_indices"
        ]
        
        raw_data = []
        for p in file_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            for field in required_fields:
                if field not in d:
                    raise ValueError(f"Missing required field: '{field}' in file {p}")
            raw_data.append(d)
        
        summary_records = [{
            "patient_id": str(d["patient_id"]),
            "core_id": str(d["core_id"]),
            "fold": int(d["fold"]),
            "seed": int(d["seed"])
        } for d in raw_data]
        summary_df = pd.DataFrame(summary_records)
        
        if summary_df.duplicated(["core_id", "seed"]).any():
            raise ValueError("A (core_id, seed) combination occurs more than once in the dataset.")
            
        for core_id, group in summary_df.groupby("core_id"):
            observed = set(group["seed"])
            if observed != self.EXPECTED_SEED_SET:
                raise ValueError(f"Core {core_id}: expected seeds {self.EXPECTED_SEED_SET}, found {observed}")

        if (summary_df.groupby("core_id")["patient_id"].nunique() != 1).any():
            raise ValueError("Multiple patients found for a single core.")

        if (summary_df.groupby("core_id")["fold"].nunique() != 1).any():
            raise ValueError("A core appears in more than one fold.")
            
        patients_per_fold = summary_df.groupby("fold")["patient_id"].nunique()
        if (patients_per_fold != 1).any():
            raise ValueError("Each held-out fold must map to exactly one patient.")

        cores = summary_df['core_id'].unique()
        for core in cores:
            core_runs = sorted([d for d in raw_data if d["core_id"] == core], key=lambda x: x["seed"])
            self._validate_core_alignment(core_runs)
            self.dataset.append(core_runs)
            
        print(f"Successfully loaded and strictly validated {len(cores)} unique cores.")

    def _validate_core_alignment(self, core_runs):
        num_nodes = len(to_numpy(core_runs[0]["cell_ids"]))
        
        for run in core_runs:
            H_val = to_numpy(run['membership_weights']).reshape(-1)
            i_idx = to_numpy(run['member_indices']).astype(np.int64)
            j_idx = to_numpy(run['anchor_indices']).astype(np.int64)
            W_e = to_numpy(run['hyperedge_weights']).reshape(-1)
            coords = to_numpy(run['raw_coordinates'])
            true_labels = to_numpy(run['true_cell_types']).reshape(-1)
            
            if len(coords) != num_nodes or len(true_labels) != num_nodes or coords.ndim != 2:
                raise ValueError(f"Structural shape mismatch in {run['core_id']}.")
            if not (len(H_val) == len(i_idx) == len(j_idx)):
                raise ValueError(f"Mismatch in sparse tensor lengths for {run['core_id']}.")
                
            if len(i_idx) > 0 and len(j_idx) > 0:
                if not (i_idx.min() >= 0 and i_idx.max() < num_nodes):
                    raise ValueError(f"Out of bounds member index for {run['core_id']}.")
                if not (j_idx.min() >= 0 and j_idx.max() < num_nodes):
                    raise ValueError(f"Out of bounds anchor index for {run['core_id']}.")

            if len(W_e) != num_nodes:
                raise ValueError(f"Expected one hyperedge weight per node in {run['core_id']}.")
            if not np.isfinite(coords).all():
                raise ValueError(f"Non-finite coordinates detected in {run['core_id']}.")
            if not np.isfinite(W_e).all():
                raise ValueError(f"Non-finite hyperedge weights detected in {run['core_id']}.")
            if not np.isfinite(H_val).all():
                raise ValueError(f"Non-finite membership weights detected in {run['core_id']}.")

        ref = core_runs[0]
        for run in core_runs[1:]:
            if not np.array_equal(to_numpy(ref["cell_ids"]), to_numpy(run["cell_ids"])):
                raise ValueError(f"Cell ordering differs across seeds for {run['core_id']}.")
            if not np.allclose(to_numpy(ref["raw_coordinates"]), to_numpy(run["raw_coordinates"])):
                raise ValueError(f"Coordinates differ across seeds for {run['core_id']}.")
            if not np.array_equal(to_numpy(ref["true_cell_types"]), to_numpy(run["true_cell_types"])):
                raise ValueError(f"True labels differ across seeds for {run['core_id']}.")

    def calculate_niche_metrics(self, run_data, threshold: float):
        W_e = to_numpy(run_data['hyperedge_weights']).reshape(-1)
        coords = to_numpy(run_data['raw_coordinates'])
        i_idx = to_numpy(run_data['member_indices']).astype(np.int64)
        j_idx = to_numpy(run_data['anchor_indices']).astype(np.int64)
        H_val = to_numpy(run_data['membership_weights']).reshape(-1)
        true_labels = to_numpy(run_data['true_cell_types']).reshape(-1)
        
        metrics = []
        eligible_anchors = np.where(W_e > self.MIN_HYPEREDGE_WEIGHT)[0]
        spatial_tree = cKDTree(coords)
        unique_cell_types = np.unique(true_labels)
        global_ct_log = np.log(len(unique_cell_types)) if len(unique_cell_types) > 1 else 1.0
        
        for anchor in eligible_anchors:
            mask = (j_idx == anchor)
            anchor_H_val = H_val[mask]
            anchor_members = i_idx[mask]
            
            soft_mass = np.sum(anchor_H_val)
            hard_mask = anchor_H_val > threshold
            hard_size = np.sum(hard_mask)
            
            # if hard_size < 3: 
            #     continue

            if hard_size < 3: 
                continue
                
            valid_members = anchor_members[hard_mask]
            valid_weights = anchor_H_val[hard_mask]
            member_coords = coords[valid_members]
            anchor_coord = coords[anchor].reshape(1, -1)
            
            distances = cdist(member_coords, anchor_coord).flatten()
            weighted_radius = np.sum(valid_weights * distances) / np.sum(valid_weights)
            max_radius = np.max(distances)
            radius_90th = np.percentile(distances, 90)
            
            center_of_mass = np.average(member_coords, axis=0, weights=valid_weights).reshape(1, -1)
            distances_to_com = cdist(member_coords, center_of_mass).flatten()
            radius_of_gyration = np.sqrt(np.sum(valid_weights * (distances_to_com ** 2)) / np.sum(valid_weights))
            
            try:
                hull_area = ConvexHull(member_coords).volume
            except QhullError:
                hull_area = 0.0 
                
            _, knn_members = spatial_tree.query(anchor_coord.flatten(), k=hard_size)
            knn_members = [knn_members] if hard_size == 1 else knn_members
            knn_coords = coords[knn_members]
            knn_distances = cdist(knn_coords, anchor_coord).flatten()
            knn_max_radius = np.max(knn_distances)
            knn_radius_90th = np.percentile(knn_distances, 90)

            labels = true_labels[valid_members]
            counts = Counter(labels)
            probs = np.array(list(counts.values())) / hard_size
            comp_entropy = entropy(probs) / global_ct_log
            
            metrics.append({
                "patient_id": run_data["patient_id"],
                "core_id": run_data["core_id"],
                "fold": run_data["fold"],
                "seed": run_data["seed"],
                "threshold": threshold,
                "anchor_idx": anchor,
                "anchor_type": true_labels[anchor],
                "hyperedge_weight": W_e[anchor],
                "soft_mass": soft_mass,
                "hard_size": hard_size,
                "weighted_radius": weighted_radius,
                "max_radius": max_radius,
                "radius_90th": radius_90th,
                "radius_of_gyration": radius_of_gyration,
                "convex_hull_area": hull_area,
                "knn_max_radius_baseline": knn_max_radius,
                "knn_radius_90th_baseline": knn_radius_90th,
                "normalized_entropy": comp_entropy
            })
            
        return pd.DataFrame(metrics)

    def construct_consensus_niches(self, core_runs, threshold: float):
        num_seeds = len(core_runs)
        majority_thresh = (num_seeds // 2) + 1
        
        anchor_eligibility_counts = Counter()
        anchor_valid_niche_counts = Counter()
        anchor_member_counts = {}
        anchor_seed_memberships = {} 
        
        for run in core_runs:
            W_e = to_numpy(run["hyperedge_weights"]).reshape(-1)
            i_idx = to_numpy(run["member_indices"]).astype(np.int64)
            j_idx = to_numpy(run["anchor_indices"]).astype(np.int64)
            H_val = to_numpy(run["membership_weights"]).reshape(-1)
            
            eligible_anchors = np.where(W_e > self.MIN_HYPEREDGE_WEIGHT)[0]
            for anchor in eligible_anchors:
                anchor_eligibility_counts[anchor] += 1
                mask = (j_idx == anchor) & (H_val > threshold)
                members_set = set(i_idx[mask]) 
                
                # =====================================================================
                # FIX: Point 1 (v8) - Accumulate membership ONLY from VALID seed-level niches
                # =====================================================================
                if len(members_set) >= 3:
                    anchor_valid_niche_counts[anchor] += 1
                    
                    if anchor not in anchor_seed_memberships:
                        anchor_seed_memberships[anchor] = []
                    anchor_seed_memberships[anchor].append(members_set)
                    
                    if anchor not in anchor_member_counts:
                        anchor_member_counts[anchor] = Counter()
                    anchor_member_counts[anchor].update(members_set)
                    
        consensus_niches = {}
        for anchor, counts in anchor_member_counts.items():
            # =====================================================================
            # FIX: Point 1 (v8) - Require BOTH majority eligibility AND valid-niche recurrence
            # =====================================================================
            if (
                anchor_eligibility_counts[anchor] >= majority_thresh
                and anchor_valid_niche_counts[anchor] >= majority_thresh
            ):
                consensus_members = [
                    cell for cell, frequency in counts.items() 
                    if frequency >= majority_thresh
                ]
                
                if len(consensus_members) >= 3:
                    consensus_niches[anchor] = np.asarray(consensus_members, dtype=int)
                    
        return consensus_niches, anchor_eligibility_counts, anchor_valid_niche_counts, anchor_seed_memberships

    def calculate_consensus_enrichment(self, core_runs, threshold: float, n_permutations: int,
                                       consensus_niches, anchor_eligibility_counts, 
                                       anchor_valid_niche_counts, anchor_seed_memberships):
        if not consensus_niches:
            return pd.DataFrame()
            
        first_run = core_runs[0]
        coords = to_numpy(first_run["raw_coordinates"])
        true_labels = to_numpy(first_run["true_cell_types"]).reshape(-1)
        num_nodes = len(true_labels)
        num_seeds = len(core_runs)
        majority_thresh = (num_seeds // 2) + 1
        
        num_majority_eligible_anchors = sum(1 for c in anchor_eligibility_counts.values() if c >= majority_thresh)
        num_majority_valid_niche_anchors = sum(1 for c in anchor_valid_niche_counts.values() if c >= majority_thresh)
        num_consensus_niches = len(consensus_niches)
        
        reproducibility_fraction = (
            num_consensus_niches / num_majority_eligible_anchors 
            if num_majority_eligible_anchors > 0 else np.nan
        )
            
        enrichment_records = []
        spatial_tree = cKDTree(coords)
        unique_cell_types = np.unique(true_labels)
        
        seed_str = f"{first_run['core_id']}_consensus_enrichment"
        seed_int = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed_int)
        
        for anchor, members in consensus_niches.items():
            hard_size = len(members)
            
            jaccards = []
            seed_sets = anchor_seed_memberships.get(anchor, [])
            n_sets = len(seed_sets)
            for i in range(n_sets):
                for j in range(i + 1, n_sets):
                    s1, s2 = seed_sets[i], seed_sets[j]
                    union_len = len(s1.union(s2))
                    if union_len > 0:
                        jaccards.append(len(s1.intersection(s2)) / union_len)
            mean_pw_jaccard = np.mean(jaccards) if jaccards else np.nan
            
            anchor_coord = coords[anchor]
            member_coords = coords[members]
            distances = np.linalg.norm(member_coords - anchor_coord, axis=1)
            observed_radius = np.max(distances)
            obs_counts = Counter(true_labels[members])
            
            local_null_dist = {ct: np.zeros(n_permutations) for ct in unique_cell_types}
            local_candidate_idx = np.asarray(spatial_tree.query_ball_point(anchor_coord, r=observed_radius * 1.5), dtype=int)
            valid_local_null = len(local_candidate_idx) >= hard_size
            
            if valid_local_null:
                for p in range(n_permutations):
                    sampled_members = rng.choice(local_candidate_idx, size=hard_size, replace=False)
                    sampled_counts = Counter(true_labels[sampled_members])
                    for ct in unique_cell_types:
                        local_null_dist[ct][p] = sampled_counts.get(ct, 0)
                        
            global_null_dist = {ct: np.zeros(n_permutations) for ct in unique_cell_types}
            for p in range(n_permutations):
                random_anchor = rng.integers(0, num_nodes)
                _, global_candidate_idx = spatial_tree.query(coords[random_anchor], k=hard_size)
                global_candidate_idx = np.asarray(global_candidate_idx, dtype=int)
                sampled_counts = Counter(true_labels[global_candidate_idx])
                for ct in unique_cell_types:
                    global_null_dist[ct][p] = sampled_counts.get(ct, 0)
                    
            for ct, observed_count in obs_counts.items():
                observed_fraction = observed_count / hard_size
                
                if valid_local_null:
                    local_counts = local_null_dist[ct]
                    local_exp_frac = np.mean(local_counts) / hard_size
                    local_diff = observed_fraction - local_exp_frac
                    local_fold = (observed_fraction + 1e-8) / (local_exp_frac + 1e-8)
                    local_pval = (np.sum(local_counts >= observed_count) + 1) / (n_permutations + 1)
                else:
                    local_exp_frac = local_diff = local_fold = local_pval = np.nan
                    
                global_counts = global_null_dist[ct]
                global_exp_frac = np.mean(global_counts) / hard_size
                global_diff = observed_fraction - global_exp_frac
                global_fold = (observed_fraction + 1e-8) / (global_exp_frac + 1e-8)
                global_pval = (np.sum(global_counts >= observed_count) + 1) / (n_permutations + 1)
                
                enrichment_records.append({
                    "patient_id": first_run["patient_id"],
                    "core_id": first_run["core_id"],
                    "fold": first_run["fold"],
                    "threshold": threshold,
                    "anchor_idx": anchor,
                    "cell_type": ct,
                    "hard_size": hard_size,
                    "observed_fraction": observed_fraction,
                    "local_expected_fraction": local_exp_frac,
                    "local_difference": local_diff,
                    "local_fold_enrichment": local_fold,
                    "local_p_value": local_pval,
                    "global_expected_fraction": global_exp_frac,
                    "global_difference": global_diff,
                    "global_fold_enrichment": global_fold,
                    "global_p_value": global_pval,
                    "eligible_seed_count": anchor_eligibility_counts[anchor],
                    "valid_niche_seed_count": anchor_valid_niche_counts[anchor],
                    "consensus_hard_size": hard_size,
                    "mean_pairwise_membership_jaccard": mean_pw_jaccard,
                    "num_majority_eligible_anchors": num_majority_eligible_anchors,
                    "num_majority_valid_niche_anchors": num_majority_valid_niche_anchors,
                    "num_consensus_niches": num_consensus_niches,
                    "reproducibility_fraction": reproducibility_fraction
                })
                
        return pd.DataFrame(enrichment_records)

    def calculate_consensus_compactness(self, core_runs, consensus_niches, threshold: float, n_permutations: int = 1000):
        if not consensus_niches:
            return pd.DataFrame()
            
        first_run = core_runs[0]
        coords = to_numpy(first_run["raw_coordinates"])
        compactness_records = []
        spatial_tree = cKDTree(coords)
        
        for anchor, members in consensus_niches.items():
            hard_size = len(members)
            anchor_coord = coords[anchor]
            member_coords = coords[members]
            
            seed_str = f"{first_run['core_id']}_fold{first_run['fold']}_thresh{threshold}_anchor{anchor}_compactness"
            seed_int = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed_int)

            distances = np.linalg.norm(member_coords - anchor_coord, axis=1)
            max_radius = np.max(distances)
            
            try:
                hull_area = ConvexHull(member_coords).volume if hard_size >= 4 else 0.0
            except QhullError:
                hull_area = 0.0
                
            local_cand_idx = spatial_tree.query_ball_point(anchor_coord, r=max_radius * 1.5)
            null_max_radii, null_hull_areas = [], []
            
            if len(local_cand_idx) >= hard_size:
                for _ in range(n_permutations):
                    null_members = rng.choice(local_cand_idx, size=hard_size, replace=False)
                    null_coords = coords[null_members]
                    
                    null_max_radii.append(np.max(np.linalg.norm(null_coords - anchor_coord, axis=1)))
                    try:
                        null_hull_areas.append(ConvexHull(null_coords).volume if hard_size >= 4 else 0.0)
                    except QhullError:
                        null_hull_areas.append(0.0)

            if len(null_max_radii) > 0:
                null_radii_arr = np.array(null_max_radii)
                null_areas_arr = np.array(null_hull_areas)
                
                mean_null_radius = np.mean(null_radii_arr)
                sd_null_radius = np.std(null_radii_arr)
                diff_radius = max_radius - mean_null_radius
                pval_radius = (np.sum(null_radii_arr <= max_radius) + 1) / (n_permutations + 1)
                
                mean_null_area = np.mean(null_areas_arr)
                sd_null_area = np.std(null_areas_arr)
                diff_area = hull_area - mean_null_area
                pval_area = (np.sum(null_areas_arr <= hull_area) + 1) / (n_permutations + 1)
            else:
                mean_null_radius = sd_null_radius = diff_radius = pval_radius = np.nan
                mean_null_area = sd_null_area = diff_area = pval_area = np.nan
                
            compactness_records.append({
                "patient_id": first_run["patient_id"],
                "core_id": first_run["core_id"],
                "fold": first_run["fold"],
                "threshold": threshold,
                "anchor_idx": anchor,
                "hard_size": hard_size,
                "max_radius": max_radius,
                "convex_hull_area": hull_area,
                "null_mean_radius": mean_null_radius,
                "null_sd_radius": sd_null_radius,
                "diff_radius_observed_minus_null": diff_radius,
                "pval_compactness_radius": pval_radius,
                "null_mean_area": mean_null_area,
                "null_sd_area": sd_null_area,
                "diff_area_observed_minus_null": diff_area,
                "pval_compactness_area": pval_area,
            })
            
        return pd.DataFrame(compactness_records)

    def analyze_redundancy(self, run_data, threshold: float):
        W_e = to_numpy(run_data["hyperedge_weights"]).reshape(-1)
        coords = to_numpy(run_data["raw_coordinates"])
        i_idx = to_numpy(run_data["member_indices"]).astype(np.int64)
        j_idx = to_numpy(run_data["anchor_indices"]).astype(np.int64)
        H_val = to_numpy(run_data["membership_weights"]).reshape(-1)

        eligible_anchors = np.where(W_e > self.MIN_HYPEREDGE_WEIGHT)[0]
        eligible_anchors = eligible_anchors[np.argsort(W_e[eligible_anchors])[::-1]]

        member_sets = {}
        for anchor in eligible_anchors:
            mask = (j_idx == anchor) & (H_val > threshold)
            members = set(i_idx[mask])
            if len(members) >= 3:
                member_sets[anchor] = members

        anchors = list(member_sets.keys())
        if len(anchors) < 2:
            return None

        anchor_coords = coords[anchors]
        anchor_tree = cKDTree(anchor_coords)
        
        MAX_SAFE_PAIRS = 2_000_000
        original_num_pairs = 0
        sampled_pairs = []
        
        seed_str = f"{run_data['core_id']}_fold{run_data['fold']}_seed{run_data['seed']}_thresh{threshold}_redundancy"
        rng = np.random.default_rng(int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32))
        
        for i in range(len(anchors)):
            neighbors = anchor_tree.query_ball_point(anchor_coords[i], r=self.MAX_REDUNDANCY_RADIUS)
            for j in neighbors:
                if i < j:
                    original_num_pairs += 1
                    if len(sampled_pairs) < MAX_SAFE_PAIRS:
                        sampled_pairs.append((i, j))
                    else:
                        r = rng.integers(0, original_num_pairs)
                        if r < MAX_SAFE_PAIRS:
                            sampled_pairs[r] = (i, j)
                            
        print(f"Redundancy Check (Core {run_data['core_id']}, Seed {run_data['seed']}): {len(anchors)} niches, {original_num_pairs} nearby pairs.")
        
        if original_num_pairs > MAX_SAFE_PAIRS:
            print(f"WARNING: Graph truncated from {original_num_pairs} to {MAX_SAFE_PAIRS} pairs. "
                  f"Overlap and NMS results are approximate (using 200µm radius).")

        nearby_dict = {i: [] for i in range(len(anchors))}
        for i, j in sampled_pairs:
            nearby_dict[i].append(j)
            nearby_dict[j].append(i)

        nonredundant_anchors = []
        retained_indices = set() 

        for idx, anchor in enumerate(anchors):
            s1 = member_sets[anchor]
            is_redundant = False
            for neighbor_idx in nearby_dict[idx]:
                if neighbor_idx in retained_indices:
                    s_nr = member_sets[anchors[neighbor_idx]]
                    union_size = len(s1.union(s_nr))
                    if union_size > 0:
                        iou = len(s1.intersection(s_nr)) / union_size
                        if iou > 0.50:
                            is_redundant = True
                            break
            if not is_redundant:
                nonredundant_anchors.append(anchor)
                retained_indices.add(idx)

        spatial_tree = cKDTree(coords)
        knn_sets = {}
        for anchor in anchors:
            hs = len(member_sets[anchor])
            _, knn_members = spatial_tree.query(coords[anchor], k=max(1, hs))
            knn_members = [knn_members] if hs == 1 else knn_members
            knn_sets[anchor] = set(knn_members)

        nearby_overlaps, knn_nearby_overlaps = [], []
        
        for (i, j) in sampled_pairs:
            anchor_i, anchor_j = anchors[i], anchors[j]
            s1, s2 = member_sets[anchor_i], member_sets[anchor_j]
            union_size = len(s1.union(s2))
            if union_size > 0:
                nearby_overlaps.append(len(s1.intersection(s2)) / union_size)
                
            k1, k2 = knn_sets[anchor_i], knn_sets[anchor_j]
            k_union_size = len(k1.union(k2))
            if k_union_size > 0:
                knn_nearby_overlaps.append(len(k1.intersection(k2)) / k_union_size)

        nearby_overlaps = np.asarray(nearby_overlaps)
        knn_nearby_overlaps = np.asarray(knn_nearby_overlaps)
        
        total_possible_pairs = (len(anchors) * (len(anchors) - 1)) // 2
        proportion_nearby = original_num_pairs / total_possible_pairs if total_possible_pairs > 0 else 0.0
        was_subsampled = original_num_pairs > MAX_SAFE_PAIRS

        return {
            "patient_id": run_data["patient_id"],
            "fold": run_data["fold"],
            "core_id": run_data["core_id"],
            "seed": run_data["seed"],
            "threshold": threshold,
            "num_evaluated_hyperedges": len(member_sets),
            "effective_nonredundant_niches": len(nonredundant_anchors),
            "original_nearby_pairs_count": original_num_pairs,
            "is_sampled_approximate": was_subsampled,
            "proportion_nearby_pairs": proportion_nearby,
            "nearby_mean_overlap": np.mean(nearby_overlaps) if len(nearby_overlaps) > 0 else 0.0,
            "nearby_median_overlap": np.median(nearby_overlaps) if len(nearby_overlaps) > 0 else 0.0,
            "nearby_fraction_overlap_gt_0_50": np.mean(nearby_overlaps > 0.50) if len(nearby_overlaps) > 0 else 0.0,
            "knn_nearby_mean_overlap": np.mean(knn_nearby_overlaps) if len(knn_nearby_overlaps) > 0 else 0.0,
            "knn_nearby_median_overlap": np.median(knn_nearby_overlaps) if len(knn_nearby_overlaps) > 0 else 0.0,
            "knn_nearby_fraction_overlap_gt_0_50": np.mean(knn_nearby_overlaps > 0.50) if len(knn_nearby_overlaps) > 0 else 0.0,
            "nonredundant_anchor_ids": nonredundant_anchors
        }
    
    def analyze_cross_seed_stability(self, core_runs, top_k=30, threshold=0.1):
        core_id = core_runs[0]["core_id"]
        fold = core_runs[0]["fold"]
        num_seeds = len(core_runs)
        
        weights_matrix = np.array([to_numpy(run['hyperedge_weights']).reshape(-1) for run in core_runs])
        spearman_corrs = []
        top_k_jaccards = []
        
        for i in range(num_seeds):
            for j in range(i + 1, num_seeds):
                if np.std(weights_matrix[i]) > 0 and np.std(weights_matrix[j]) > 0:
                    corr, _ = spearmanr(weights_matrix[i], weights_matrix[j])
                    spearman_corrs.append(corr if not np.isnan(corr) else 0.0)
                else:
                    spearman_corrs.append(0.0)
                
                num_nodes_for_top_k = len(weights_matrix[0])
                safe_top_k = min(top_k, num_nodes_for_top_k)
                
                top_i = set(np.argsort(weights_matrix[i])[-safe_top_k:])
                top_j = set(np.argsort(weights_matrix[j])[-safe_top_k:])
                union_len = len(top_i.union(top_j))
                if union_len > 0:
                    top_k_jaccards.append(len(top_i.intersection(top_j)) / union_len)
                
        anchor_recurrence = Counter()
        for W_e in weights_matrix:
            safe_top_k = min(top_k, len(W_e))
            anchor_recurrence.update(np.argsort(W_e)[-safe_top_k:])
            
        freq_dist = Counter(anchor_recurrence.values())
        
        stability_dict = {
            "patient_id": core_runs[0]["patient_id"],
            "core_id": core_id,
            "fold": fold,
            "mean_spearman_corr": np.mean(spearman_corrs) if spearman_corrs else 0.0,
            "mean_top_k_jaccard": np.mean(top_k_jaccards) if top_k_jaccards else 0.0
        }
        for k in range(num_seeds, 0, -1):
            stability_dict[f"anchors_{k}_out_of_{num_seeds}"] = freq_dist.get(k, 0)
            
        majority_threshold = (num_seeds // 2) + 1
        recurring_anchors = [node for node, count in anchor_recurrence.items() if count >= majority_threshold]
        
        num_nodes = len(weights_matrix[0])
        coords = to_numpy(core_runs[0]['raw_coordinates'])
        true_labels = to_numpy(core_runs[0]['true_cell_types']).reshape(-1)
        unique_cell_types = np.unique(true_labels)
        
        mem_jaccards, mem_cosines, comp_cosines, size_cvs, radius_cvs = [], [], [], [], []
        excluded_due_to_invalid_membership = 0  
        anchor_level_records = []
        
        for anchor in recurring_anchors:
            anchor_coord = coords[anchor]
            seed_hard_members, seed_dense_weights, seed_compositions, seed_sizes, seed_radii = [], [], [], [], []
            
            for run in core_runs:
                i_idx = to_numpy(run['member_indices']).astype(np.int64)
                j_idx = to_numpy(run['anchor_indices']).astype(np.int64)
                H_val = to_numpy(run['membership_weights']).reshape(-1)
                
                mask = (j_idx == anchor)
                members = i_idx[mask]
                weights = H_val[mask]
                
                hard_mask = weights > threshold
                hard_members = set(members[hard_mask])
                
                if len(hard_members) < 3:
                    continue
                    
                seed_hard_members.append(hard_members)
                
                dense_w = np.zeros(num_nodes)
                dense_w[members] = weights
                seed_dense_weights.append(dense_w)
                
                comp_counts = Counter(true_labels[list(hard_members)])
                comp_vec = np.array([comp_counts.get(ct, 0) for ct in unique_cell_types])
                comp_sum = np.sum(comp_vec)
                comp_vec = comp_vec / comp_sum if comp_sum > 0 else comp_vec
                seed_compositions.append(comp_vec)
                
                seed_sizes.append(len(hard_members))
                member_coords = coords[list(hard_members)]
                distances = np.linalg.norm(member_coords - anchor_coord, axis=1)
                seed_radii.append(np.max(distances))
                
            valid_seeds = len(seed_hard_members)
            
            if valid_seeds >= majority_threshold:
                anchor_jaccards, anchor_cosines, anchor_comp_sims = [], [], []
                
                for i in range(valid_seeds):
                    for j in range(i + 1, valid_seeds):
                        s1, s2 = seed_hard_members[i], seed_hard_members[j]
                        union_len = len(s1.union(s2))
                        if union_len > 0:
                            anchor_jaccards.append(len(s1.intersection(s2)) / union_len)
                            
                        w1, w2 = seed_dense_weights[i], seed_dense_weights[j]
                        n1, n2 = np.linalg.norm(w1), np.linalg.norm(w2)
                        if n1 > 0 and n2 > 0:
                            anchor_cosines.append(np.dot(w1, w2) / (n1 * n2))
                            
                        c1, c2 = seed_compositions[i], seed_compositions[j]
                        cn1, cn2 = np.linalg.norm(c1), np.linalg.norm(c2)
                        if cn1 > 0 and cn2 > 0:
                            anchor_comp_sims.append(np.dot(c1, c2) / (cn1 * cn2))
                            
                mean_jac = np.mean(anchor_jaccards) if anchor_jaccards else np.nan
                mean_cos = np.mean(anchor_cosines) if anchor_cosines else np.nan
                mean_comp = np.mean(anchor_comp_sims) if anchor_comp_sims else np.nan
                mean_size_cv_val = np.std(seed_sizes) / np.mean(seed_sizes) if np.mean(seed_sizes) > 0 else np.nan
                mean_rad_cv_val = np.std(seed_radii) / np.mean(seed_radii) if np.mean(seed_radii) > 0 else np.nan
                
                mem_jaccards.append(mean_jac)
                mem_cosines.append(mean_cos)
                comp_cosines.append(mean_comp)
                size_cvs.append(mean_size_cv_val)
                radius_cvs.append(mean_rad_cv_val)
                
                anchor_level_records.append({
                    "patient_id": core_runs[0]["patient_id"],
                    "core_id": core_id,
                    "fold": fold,
                    "anchor_idx": anchor,
                    "valid_seeds": valid_seeds,
                    "eligible_seed_count": sum(1 for r in core_runs if to_numpy(r['hyperedge_weights']).reshape(-1)[anchor] > self.MIN_HYPEREDGE_WEIGHT),
                    "top_k_recurrence_count": anchor_recurrence[anchor],
                    "mean_membership_jaccard": mean_jac,
                    "mean_membership_cosine": mean_cos,
                    "mean_composition_cosine": mean_comp,
                    "size_cv": mean_size_cv_val,
                    "radius_cv": mean_rad_cv_val
                })
            else:
                excluded_due_to_invalid_membership += 1

        stability_dict.update({
            "recurring_anchors_evaluated": len(mem_jaccards),
            "anchors_excluded_invalid_membership": excluded_due_to_invalid_membership,
            "mean_membership_jaccard": np.mean(mem_jaccards) if mem_jaccards else np.nan,
            "mean_membership_cosine": np.mean(mem_cosines) if mem_cosines else np.nan,
            "mean_composition_cosine": np.mean(comp_cosines) if comp_cosines else np.nan,
            "mean_size_cv": np.mean(size_cvs) if size_cvs else np.nan,
            "mean_radius_cv": np.mean(radius_cvs) if radius_cvs else np.nan
        })
        
        return stability_dict, anchor_level_records

    def extract_cell_type_map(self, h5ad_path: str):
        print(f"Loading metadata from: {h5ad_path}")
        try:
            adata = ad.read_h5ad(h5ad_path, backed='r')
            if 'class_names' in adata.uns:
                class_names = adata.uns['class_names']
                self.cell_type_map = {i: str(name) for i, name in enumerate(class_names)}
            else:
                raise KeyError("'class_names' not found in adata.uns.")
            adata.file.close() 
        except Exception as e:
            print(f"Error loading h5ad file: {e}. Falling back to default labels.")
            self.cell_type_map = {}

    def generate_spatial_visualizations(self, core_runs, threshold: float):
        first_run = core_runs[0]
        red_res = self.analyze_redundancy(first_run, threshold)
        if not red_res or "nonredundant_anchor_ids" not in red_res: 
            return
            
        valid_nonredundant_anchors = set(red_res["nonredundant_anchor_ids"])
        
        W_e = to_numpy(first_run['hyperedge_weights']).reshape(-1)
        coords = to_numpy(first_run['raw_coordinates'])
        true_labels = to_numpy(first_run['true_cell_types'])
        core_id = first_run['core_id']
        
        eligible_anchors = np.where(W_e > self.MIN_HYPEREDGE_WEIGHT)[0]
        sorted_anchors = eligible_anchors[np.argsort(W_e[eligible_anchors])[::-1]]
        target_anchor_idx = None
        required_seeds = len(core_runs) // 2 + 1 
        
        for anchor in sorted_anchors:
            if anchor not in valid_nonredundant_anchors:
                continue

            seed_memberships = []
            for run in core_runs:
                run_H = to_numpy(run['membership_weights']).reshape(-1)
                run_j = to_numpy(run['anchor_indices']).astype(np.int64)
                run_i = to_numpy(run['member_indices']).astype(np.int64)
                
                mask = (run_j == anchor) & (run_H > threshold)
                members = set(run_i[mask])
                
                if len(members) >= 3:
                    seed_memberships.append(members)
                    
            if len(seed_memberships) < required_seeds:
                continue
                
            jaccards = []
            valid_seed_count = len(seed_memberships)
            for i in range(valid_seed_count):
                for j in range(i + 1, valid_seed_count):
                    s1, s2 = seed_memberships[i], seed_memberships[j]
                    union_len = len(s1.union(s2))
                    if union_len > 0:
                        jaccards.append(len(s1.intersection(s2)) / union_len)
                        
            mean_jaccard = np.mean(jaccards) if jaccards else 0.0
            
            if mean_jaccard >= 0.50:
                target_anchor_idx = anchor
                break
                
        if target_anchor_idx is None:
            return
        
        num_seeds = len(core_runs)
        fig, axes = plt.subplots(1, num_seeds + 1, figsize=(6 * (num_seeds + 1), 6))
        unique_labels = np.unique(true_labels)
        palette = sns.color_palette("tab20", len(unique_labels))
        
        for idx, lbl in enumerate(unique_labels):
            mask = (true_labels == lbl)
            lbl_name = self.cell_type_map.get(lbl, f"Type_{lbl}")
            axes[0].scatter(coords[mask, 0], coords[mask, 1], color=palette[idx], s=10, label=lbl_name, alpha=0.7, edgecolor='none')
        axes[0].set_title(f"Tissue Context\n{core_id}", fontsize=14)
        axes[0].invert_yaxis()
        axes[0].axis('equal')
        
        for idx, run in enumerate(core_runs):
            ax = axes[idx + 1]
            ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=10, alpha=0.3, edgecolor='none')
            
            run_H = to_numpy(run['membership_weights']).reshape(-1)
            run_j = to_numpy(run['anchor_indices']).astype(np.int64)
            run_i = to_numpy(run['member_indices']).astype(np.int64)
            
            niche_mask = (run_j == target_anchor_idx) & (run_H > threshold)
            if np.sum(niche_mask) > 0:
                member_coords = coords[run_i[niche_mask]]
                ax.scatter(member_coords[:, 0], member_coords[:, 1], c=run_H[niche_mask], cmap='viridis', 
                             s=40, edgecolor='black', linewidth=0.5, vmin=0, vmax=1)
                
            ax.scatter(coords[target_anchor_idx, 0], coords[target_anchor_idx, 1], c='red', marker='*', s=250, edgecolor='black')
            ax.set_title(f"Seed: {run['seed']}", fontsize=14)
            ax.invert_yaxis()
            ax.axis('equal')
            
        plt.tight_layout()
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        fig.savefig(os.path.join(vis_dir, f"spatial_grid_{core_id}_anchor_{target_anchor_idx}.pdf"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(vis_dir, f"spatial_grid_{core_id}_anchor_{target_anchor_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def run_full_pipeline(self):
        self.load_and_validate_data()
        
        all_metrics_list = []
        all_enrichment_list = []
        all_compactness_list = [] 
        stability_records = []
        all_anchor_stability = []
        redundancy_records = []
        
        prebuilt_consensus_data = []
        total_consensus_niches_exact = 0
        consensus_summary_records = []

        # ------------------------------------------------------------
        # Load the .h5ad only once
        # ------------------------------------------------------------
        import anndata as ad
        h5ad_path = ("/project/banerjee/MIM/PSB_Hyperniche/BrCr/DataProcessing_brcr/post_data/ishtyaq&jonny_hyper_2023_xenium_breast_tTMA1_with_patients.h5ad")
    

        print(f"Loading cell metadata once from: {h5ad_path}")
        adata_full = ad.read_h5ad(h5ad_path)

        adata_full.obs_names = adata_full.obs_names.astype(str)

        if not adata_full.obs_names.is_unique:
            raise ValueError(
                "The h5ad observation names are not unique; cell IDs cannot "
                "be aligned unambiguously."
            )

        # # =====================================================================
        # # TEMPORARY VERIFICATION BLOCK (Advisor's Check)
        # # =====================================================================
        # import sys
        # first_run = self.dataset[0][0]
        
        # print("\n--- VERIFICATION OUTPUT ---")
        # print("1. run['cell_ids'][:10]:")
        # print(to_numpy(first_run["cell_ids"]).reshape(-1)[:10])
        
        # print("\n2. adata_full.obs_names[:10]:")
        # print(adata_full.obs_names[:10].tolist())
        
        # print("\n3. adata_full.obs.columns:")
        # print(adata_full.obs.columns.tolist())
        # print("---------------------------\n")
        
        # sys.exit("Stopping script after verification print. Check the output above!")
        # # ==

        print("\n--- Phase 1: Pre-computing Consensus and Descriptive Metrics ---")
        for core_runs in self.dataset:
            self.generate_spatial_visualizations(core_runs, threshold=self.PRIMARY_MEMBERSHIP_THRESHOLD)
            
            stab_res, anchor_stab_res = self.analyze_cross_seed_stability(core_runs, threshold=self.PRIMARY_MEMBERSHIP_THRESHOLD)
            stability_records.append(stab_res)
            all_anchor_stability.extend(anchor_stab_res)
            
            for thresh in self.thresholds:
                for run in core_runs:
                    metrics_df = self.calculate_niche_metrics(run, threshold=thresh)
                    all_metrics_list.append(metrics_df)
                    
                    if thresh == self.PRIMARY_MEMBERSHIP_THRESHOLD:
                        red_res = self.analyze_redundancy(run, threshold=thresh)
                        if red_res: redundancy_records.append(red_res)

                    # =====================================================================
                    # NEW CALLING BLOCK
                    # =====================================================================
                    # EXPORT_NICHE_SEED = 42

                    # if (
                    #     int(run["seed"]) == EXPORT_NICHE_SEED
                    #     and np.isclose(thresh, self.PRIMARY_MEMBERSHIP_THRESHOLD)
                    # ):
                    #     print(
                    #         f"Exporting niche cell-type counts for "
                    #         f"core {run['core_id']}, seed {EXPORT_NICHE_SEED}..."
                    #     )

                    if np.isclose(thresh, self.PRIMARY_MEMBERSHIP_THRESHOLD):
                        current_seed = int(run["seed"])
                        print(
                            f"Exporting niche cell-type counts for "
                            f"core {run['core_id']}, seed {current_seed}..."
                        )

                        # ------------------------------------------------------------
                        # Align AnnData precisely to the cell order saved in this run
                        # ------------------------------------------------------------
                        run_cell_ids = (
                            pd.Index(to_numpy(run["cell_ids"]).reshape(-1).astype(str))
                        )

                        if run_cell_ids.has_duplicates:
                            raise ValueError(
                                f"Duplicate cell IDs found in core {run['core_id']}."
                            )

                        missing_cell_ids = run_cell_ids.difference(adata_full.obs_names.astype(str))

                        if len(missing_cell_ids) > 0:
                            raise ValueError(
                                f"Core {run['core_id']}: {len(missing_cell_ids)} saved cell IDs "
                                "were not found in the h5ad file. "
                                f"Examples: {missing_cell_ids[:5].tolist()}"
                            )

                        # The resulting observation order must equal run_cell_ids exactly.
                        adata_core = adata_full[run_cell_ids.tolist()].copy()

                        observed_cell_ids = adata_core.obs_names.astype(str).to_numpy()

                        if not np.array_equal(observed_cell_ids, run_cell_ids.to_numpy()):
                            raise RuntimeError(
                                f"AnnData ordering could not be aligned for core {run['core_id']}."
                            )

                        # ------------------------------------------------------------
                        # Validate coordinates against the saved run
                        # ------------------------------------------------------------
                        saved_coordinates = to_numpy(run["raw_coordinates"])

                        if adata_core.n_obs != len(saved_coordinates):
                            raise ValueError(
                                f"Core {run['core_id']}: AnnData contains {adata_core.n_obs} "
                                f"cells, but the saved run contains {len(saved_coordinates)}."
                            )

                        if not np.allclose(
                            np.asarray(adata_core.obsm["spatial"]),
                            saved_coordinates,
                            rtol=1e-5,
                            atol=1e-5,
                        ):
                            raise ValueError(
                                f"Core {run['core_id']}: h5ad coordinates do not match the "
                                "coordinates saved in the .pt file."
                            )

                        # ------------------------------------------------------------
                        # Extract sparse hard memberships
                        # ------------------------------------------------------------
                        run_H = to_numpy(run["membership_weights"]).reshape(-1)
                        run_anchor = (
                            to_numpy(run["anchor_indices"]).reshape(-1).astype(np.int64)
                        )
                        run_member = (
                            to_numpy(run["member_indices"]).reshape(-1).astype(np.int64)
                        )
                        hyperedge_weights = (
                            to_numpy(run["hyperedge_weights"]).reshape(-1)
                        )

                        if not (
                            len(run_H) == len(run_anchor) == len(run_member)
                        ):
                            raise ValueError(
                                f"Core {run['core_id']}: sparse-array lengths do not agree."
                            )

                        hard_edge_mask = run_H > thresh

                        hard_anchor = run_anchor[hard_edge_mask]
                        hard_member = run_member[hard_edge_mask]

                        # Count hard members per anchor.
                        anchor_indices, niche_sizes = np.unique(
                            hard_anchor,
                            return_counts=True,
                        )

                        # Match calculate_niche_metrics:
                        #   hyperedge weight > 0.05
                        #   hard niche size >= 3
                        eligible_by_weight = (
                            hyperedge_weights[anchor_indices]
                            > self.MIN_HYPEREDGE_WEIGHT
                        )
                        # eligible_by_size = niche_sizes >= 3
                        eligible_by_size = niche_sizes >= 3

                        valid_anchor_indices = anchor_indices[
                            eligible_by_weight & eligible_by_size
                        ]

                        patient_id = str(run["patient_id"])
                        core_id = str(run["core_id"])
                        output_dir = Path(self.output_dir)

                        niche_summary_df, celltype_column_map = (
                            export_niche_celltype_counts(
                                hard_anchor=hard_anchor,
                                hard_member=hard_member,
                                valid_anchor_indices=valid_anchor_indices,
                                adata=adata_core,
                                output_csv=(
                                    output_dir
                                    / (
                                        f"{patient_id}_{core_id}_seed"
                                        f"{current_seed}_niche_celltype_counts.csv" # FIXED HERE
                                    )
                                ),
                                patient_id=patient_id,
                                core_id=core_id,
                                seed=current_seed, # FIXED HERE
                                celltype_key="cell_type",
                            )
                        )
                    # =====================================================================
            
            c_niches, el_counts, val_counts, seed_mems = self.construct_consensus_niches(
                core_runs, self.PRIMARY_MEMBERSHIP_THRESHOLD
            )
            
            num_seeds = len(core_runs)
            majority_thresh = (num_seeds // 2) + 1
            consensus_summary_records.append({
                "patient_id": core_runs[0]["patient_id"],
                "core_id": core_runs[0]["core_id"],
                "fold": core_runs[0]["fold"],
                "num_majority_eligible_anchors": sum(count >= majority_thresh for count in el_counts.values()),
                "num_majority_valid_niche_anchors": sum(count >= majority_thresh for count in val_counts.values()),
                "num_consensus_niches": len(c_niches)
            })
            
            prebuilt_consensus_data.append({
                "core_runs": core_runs,
                "consensus_niches": c_niches,
                "eligibility_counts": el_counts,
                "valid_niche_counts": val_counts,
                "seed_memberships": seed_mems
            })
            total_consensus_niches_exact += len(c_niches)
            
        print("\n--- Phase 2: Projected Workload Estimation ---")
        print(f"Constructed EXACTLY {total_consensus_niches_exact} consensus niches across all cores.")
        if total_consensus_niches_exact > 0:
            print(f"Projected Permutations: ~{total_consensus_niches_exact * 3000:,} (1k local + 1k global + 1k compactness).")
            print("Note: Actual count may be lower if local valid null sets are unavailable.")
        print("----------------------------------------------\n")

        print("--- Phase 3: Running Permutation Inferences ---")
        for data in prebuilt_consensus_data:
            consensus_df = self.calculate_consensus_enrichment(
                data["core_runs"], self.PRIMARY_MEMBERSHIP_THRESHOLD, 1000,
                data["consensus_niches"], data["eligibility_counts"],
                data["valid_niche_counts"], data["seed_memberships"]
            )
            if not consensus_df.empty:
                all_enrichment_list.append(consensus_df)
                
            compactness_df = self.calculate_consensus_compactness(
                data["core_runs"], data["consensus_niches"], self.PRIMARY_MEMBERSHIP_THRESHOLD
            )
            if not compactness_df.empty:
                all_compactness_list.append(compactness_df)
                    
        all_metrics_df = pd.concat(all_metrics_list, ignore_index=True) if all_metrics_list else pd.DataFrame()
        consensus_enrichment = pd.concat(all_enrichment_list, ignore_index=True) if all_enrichment_list else pd.DataFrame()
        consensus_compactness = pd.concat(all_compactness_list, ignore_index=True) if all_compactness_list else pd.DataFrame()

        if not consensus_enrichment.empty:
            print(f"Applying Global FDR Correction on enrichment tests...")
            for pval_col in ["local_p_value", "global_p_value"]:
                mask = consensus_enrichment[pval_col].notna()
                if mask.sum() > 0:
                    reject, q_values = fdrcorrection(consensus_enrichment.loc[mask, pval_col].to_numpy())
                    consensus_enrichment.loc[mask, f"{pval_col.replace('p_value', 'fdr_q_value')}"] = q_values
                    consensus_enrichment.loc[mask, f"{pval_col.replace('p_value', 'fdr_reject')}"] = reject
                    
        if not consensus_compactness.empty:
            print(f"Applying Global FDR Correction on {len(consensus_compactness)} consensus compactness tests...")
            for pval_col in ["pval_compactness_radius", "pval_compactness_area"]:
                mask = consensus_compactness[pval_col].notna()
                if mask.sum() > 0:
                    reject, q_values = fdrcorrection(consensus_compactness.loc[mask, pval_col].to_numpy())
                    consensus_compactness.loc[mask, f"{pval_col.replace('pval_', 'fdr_q_value_')}"] = q_values
                    consensus_compactness.loc[mask, f"{pval_col.replace('pval_', 'fdr_reject_')}"] = reject

        all_metrics_df.to_csv(os.path.join(self.output_dir, "hyperedge_metrics_sensitivity_analysis.csv"), index=False)
        pd.DataFrame(stability_records).to_csv(os.path.join(self.output_dir, "stability_cross_seed.csv"), index=False)
        if all_anchor_stability:
            pd.DataFrame(all_anchor_stability).to_csv(os.path.join(self.output_dir, "anchor_level_stability.csv"), index=False)
        pd.DataFrame(redundancy_records).to_csv(os.path.join(self.output_dir, "redundancy_overlap.csv"), index=False)
        
        if not consensus_enrichment.empty:
            consensus_enrichment.to_csv(os.path.join(self.output_dir, "enrichment_fdr_consensus_results.csv"), index=False)
        if not consensus_compactness.empty:
            consensus_compactness.to_csv(os.path.join(self.output_dir, "compactness_fdr_consensus_results.csv"), index=False)

        core_consensus_df = pd.DataFrame(consensus_summary_records)
        core_consensus_df.to_csv(os.path.join(self.output_dir, "core_level_consensus_reproducibility.csv"), index=False)

        print("\nAggregating Results to Patient Level with Uncertainty, Median, and IQR...")
        
        metric_cols = [
            "soft_mass", "hard_size", "weighted_radius", "max_radius", "radius_90th",
            "radius_of_gyration", "convex_hull_area", "knn_max_radius_baseline",
            "knn_radius_90th_baseline", "normalized_entropy"
        ]
        
        redundancy_cols = [
            "num_evaluated_hyperedges", "effective_nonredundant_niches", "proportion_nearby_pairs",
            "nearby_mean_overlap", "nearby_median_overlap", "nearby_fraction_overlap_gt_0_50",
            "knn_nearby_mean_overlap", "knn_nearby_median_overlap", "knn_nearby_fraction_overlap_gt_0_50"
        ]
        
        stability_cols = [
            "mean_spearman_corr", "mean_top_k_jaccard", "recurring_anchors_evaluated", "anchors_excluded_invalid_membership",
            "mean_membership_jaccard", "mean_membership_cosine", "mean_composition_cosine",
            "mean_size_cv", "mean_radius_cv"
        ]

        def iqr(x):
            return x.quantile(0.75) - x.quantile(0.25)

        if not all_metrics_df.empty:
            df = all_metrics_df[all_metrics_df["threshold"] == self.PRIMARY_MEMBERSHIP_THRESHOLD].copy()
            existing_cols = [c for c in metric_cols if c in df.columns]
            
            patient_stats = df.groupby("patient_id").agg(
                num_cores=("core_id", "nunique"),
                num_seeds=("seed", "nunique"),
                total_eligible_niches=("anchor_idx", "count")
            )
            
            seed_agg = df.groupby(["patient_id", "core_id", "fold", "seed"])[existing_cols].mean().reset_index()
            core_agg = seed_agg.groupby(["patient_id", "core_id", "fold"])[existing_cols].mean().reset_index()
            
            mean_df = core_agg.groupby("patient_id")[existing_cols].mean().add_suffix("_mean")
            std_df = core_agg.groupby("patient_id")[existing_cols].std().add_suffix("_std_across_cores")
            
            median_df = df.groupby("patient_id")[existing_cols].median().add_suffix("_median")
            iqr_df = df.groupby("patient_id")[existing_cols].agg(iqr).add_suffix("_iqr")
            
            patient_agg = pd.concat([patient_stats, mean_df, std_df, median_df, iqr_df], axis=1).reset_index()
            patient_agg.to_csv(os.path.join(self.output_dir, "patient_level_metrics.csv"), index=False)

        patient_repro_stats = core_consensus_df.groupby("patient_id").agg(
            num_cores=("core_id", "nunique"),
            num_majority_eligible_anchors=("num_majority_eligible_anchors", "sum"),
            num_majority_valid_niche_anchors=("num_majority_valid_niche_anchors", "sum"),
            num_consensus_niches=("num_consensus_niches", "sum")
        ).reset_index()
        
        patient_repro_stats["patient_reproducibility_fraction"] = np.where(
            patient_repro_stats["num_majority_eligible_anchors"] > 0,
            patient_repro_stats["num_consensus_niches"] / patient_repro_stats["num_majority_eligible_anchors"],
            np.nan 
        )

        patient_agg = patient_repro_stats.copy()
        patient_agg["num_local_significant_niches"] = 0
        patient_agg["fraction_consensus_niches_local_significant"] = np.nan

        if not consensus_enrichment.empty:
            df = consensus_enrichment.copy()
            for col in ["local_fdr_reject", "global_fdr_reject"]:
                if col not in df.columns:
                    df[col] = False
                df[col] = df[col].astype(float)
                
            sig_tests = df[df["local_fdr_reject"] == 1.0].copy()

            if not sig_tests.empty:
                sig_niches_per_patient = (
                    sig_tests[["patient_id", "core_id", "anchor_idx"]]
                    .drop_duplicates()
                    .groupby("patient_id")
                    .size()
                    .reset_index(name="num_local_significant_niches")
                )
                
                sig_effects_per_patient = sig_tests.groupby("patient_id").agg(
                    median_sig_local_fold_enrichment=("local_fold_enrichment", "median"),
                    median_sig_local_difference=("local_difference", "median")
                ).reset_index()
                
                patient_agg = patient_agg.drop(columns=["num_local_significant_niches"])
                patient_agg = patient_agg.merge(sig_niches_per_patient, on="patient_id", how="left")
                patient_agg["num_local_significant_niches"] = patient_agg["num_local_significant_niches"].fillna(0)
                
                patient_agg = patient_agg.merge(sig_effects_per_patient, on="patient_id", how="left")
                
                strongest_per_patient = sig_tests.sort_values("local_fold_enrichment", ascending=False).drop_duplicates(["patient_id"])
                strongest_per_patient_summary = strongest_per_patient[[
                    "patient_id", "cell_type", "local_fold_enrichment", "local_fdr_q_value"
                ]].rename(columns={
                    "cell_type": "top_enriched_cell_type_overall",
                    "local_fold_enrichment": "max_local_fold_enrichment",
                    "local_fdr_q_value": "top_enriched_q_value"
                })
                
                patient_agg = patient_agg.merge(strongest_per_patient_summary, on="patient_id", how="left")
                
                strongest_per_niche = sig_tests.sort_values("local_fold_enrichment", ascending=False).drop_duplicates(["patient_id", "core_id", "anchor_idx"])
                strongest_per_niche.to_csv(os.path.join(self.output_dir, "strongest_significant_enrichment_per_niche.csv"), index=False)
            
            patient_agg["fraction_consensus_niches_local_significant"] = np.where(
                patient_agg["num_consensus_niches"] > 0,
                patient_agg["num_local_significant_niches"] / patient_agg["num_consensus_niches"],
                np.nan
            )
        patient_agg.to_csv(os.path.join(self.output_dir, "patient_level_enrichment_summary.csv"), index=False)

        if redundancy_records:
            df = pd.DataFrame(redundancy_records)
            existing_cols = [c for c in redundancy_cols if c in df.columns]
            
            patient_stats = df.groupby("patient_id").agg(
                num_cores=("core_id", "nunique"),
                num_seeds=("seed", "nunique")
            )
            
            core_agg = df.groupby(["patient_id", "core_id", "fold"])[existing_cols].mean().reset_index()
            mean_df = core_agg.groupby("patient_id")[existing_cols].mean().add_suffix("_mean")
            std_df = core_agg.groupby("patient_id")[existing_cols].std().add_suffix("_std_across_cores")
            
            patient_agg = pd.concat([patient_stats, mean_df, std_df], axis=1).reset_index()
            patient_agg.to_csv(os.path.join(self.output_dir, "patient_level_redundancy.csv"), index=False)

        if stability_records:
            df = pd.DataFrame(stability_records)
            existing_cols = [c for c in stability_cols if c in df.columns]
            
            patient_stats = df.groupby("patient_id").agg(
                num_cores=("core_id", "nunique")
            )
            
            mean_df = df.groupby("patient_id")[existing_cols].mean().add_suffix("_mean")
            std_df = df.groupby("patient_id")[existing_cols].std().add_suffix("_std_across_cores")
            
            patient_agg = pd.concat([patient_stats, mean_df, std_df], axis=1).reset_index()
            patient_agg.to_csv(os.path.join(self.output_dir, "patient_level_stability.csv"), index=False)
            
        print(f"Done! All evaluations saved in '{self.output_dir}'")


if __name__ == "__main__":
    INPUT_DIR = "/project/Hyperniche_main_model_output"
    OUTPUT_DIR = "./biology_analysis"
    H5AD_PATH = "/project/data.h5ad"
    
    evaluator = HyperNicheEvaluator(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    evaluator.extract_cell_type_map(H5AD_PATH)
    evaluator.run_full_pipeline()