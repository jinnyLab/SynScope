#!/usr/bin/env python3
"""
Feature computation for pairwise overlap models.

This module computes morphological, spatial, and overlap features.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations
from collections import defaultdict

from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, disk, ball
from scipy.ndimage import binary_erosion, binary_dilation

import networkx as nx

try:
    from .mGRASP_puncta_core_functions import fit_threshold_gmm
except ImportError:
    from mGRASP_puncta_core_functions import fit_threshold_gmm


def compute_local_gmm_threshold(
    pixels,
    fallback_percentile=95,
    dprime_thresh=1.0,
    channel_num=None
):
    """Local GMM threshold with d′ check."""
    thr, _ = fit_threshold_gmm(
        np.array(pixels),
        fallback_percentile=fallback_percentile,
        dprime_thresh=dprime_thresh,
        return_dprime=True,
    )
    return float(thr)


def calculate_overlap_from_z_maps(
    zmap1: Dict[int, Tuple[np.ndarray, Tuple]],
    zmap2: Dict[int, Tuple[np.ndarray, Tuple]]
) -> float:
    """
    Calculate overlap score between two channel z-mask maps using multiscale overlap.
    Combines masks across z-slices and computes multiscale overlap.

    Returns:
        Overlap score (weighted score from multiscale overlap)
    """
    shared_z = set(zmap1.keys()) & set(zmap2.keys())
    if not shared_z:
        return 0.0

    # Combine masks across z-slices
    m1_combined = None
    m2_combined = None

    for z in shared_z:
        m1, bb1 = zmap1[z]
        m2, bb2 = zmap2[z]

        # Align masks if bounding boxes differ
        if bb1 != bb2:
            y1a, y2a, x1a, x2a = bb1
            y1b, y2b, x1b, x2b = bb2
            yi1, yi2 = max(y1a, y1b), min(y2a, y2b)
            xi1, xi2 = max(x1a, x1b), min(x2a, x2b)
            if yi2 <= yi1 or xi2 <= xi1:
                continue
            m1_aligned = m1[yi1 - y1a: yi2 - y1a, xi1 - x1a: xi2 - x1a]
            m2_aligned = m2[yi1 - y1b: yi2 - y1b, xi1 - x1b: xi2 - x1b]
            if m1_aligned.shape != m2_aligned.shape:
                continue
            m1, m2 = m1_aligned, m2_aligned

        # Combine across z-slices
        if m1_combined is None:
            m1_combined = m1.copy()
            m2_combined = m2.copy()
        else:
            # Need to align to same bounding box
            if m1_combined.shape == m1.shape:
                m1_combined = m1_combined | m1
                m2_combined = m2_combined | m2
            else:
                # Different sizes - use union of bounding boxes
                h1, w1 = m1_combined.shape
                h2, w2 = m1.shape
                h_max, w_max = max(h1, h2), max(w1, w2)
                m1_new = np.zeros((h_max, w_max), dtype=bool)
                m2_new = np.zeros((h_max, w_max), dtype=bool)
                m1_new[:h1, :w1] = m1_combined
                m2_new[:h1, :w1] = m2_combined
                m1_new[:h2, :w2] = m1_new[:h2, :w2] | m1
                m2_new[:h2, :w2] = m2_new[:h2, :w2] | m2
                m1_combined = m1_new
                m2_combined = m2_new

    if m1_combined is None or m2_combined is None:
        return 0.0

    # Compute multiscale overlap
    overlap_metrics = compute_multiscale_overlap(
        m1_combined, m2_combined,
        scale_factors=[0.8, 1.0, 1.2],
        metrics=['jaccard', 'dice'],
        weights=[0.5, 0.5]
    )

    # Return weighted score (0.5*containment + 0.5*jaccard)
    return float(overlap_metrics.get('weighted_score', 0.0))


def compute_spatial_features_mask_only(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute spatial features using only the mask (no intensity).

    Args:
        mask: Binary mask

    Returns:
        Dictionary of spatial features
    """
    features = {}

    if not np.any(mask):
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'compactness': 0.0,
            'spatial_entropy': 0.0,
            'skeleton_length': 0.0
        }

    # Basic spatial features
    features['area'] = float(np.sum(mask))

    # Perimeter (approximate)
    eroded = binary_erosion(mask)
    perimeter_mask = mask & ~eroded
    features['perimeter'] = float(np.sum(perimeter_mask))

    # Compactness (perimeter^2 / area)
    if features['area'] > 0:
        features['compactness'] = features['perimeter'] ** 2 / features['area']
    else:
        features['compactness'] = 0.0

    # Spatial entropy (distribution of pixels)
    labeled = label(mask.astype(np.uint8), connectivity=1)
    props = regionprops(labeled)

    if props:
        # Use largest component for spatial entropy
        largest_prop = max(props, key=lambda x: x.area)
        coords = largest_prop.coords

        if len(coords) > 1:
            # Calculate spatial distribution entropy
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            y_std, x_std = np.std(y_coords), np.std(x_coords)
            spatial_entropy = -np.log(y_std * x_std + 1e-8)
            features['spatial_entropy'] = float(spatial_entropy)
        else:
            features['spatial_entropy'] = 0.0
    else:
        features['spatial_entropy'] = 0.0

    # Skeleton length
    try:
        skeleton = skeletonize(mask.astype(bool))
        features['skeleton_length'] = float(np.sum(skeleton))
    except:
        features['skeleton_length'] = 0.0

    return features


def compute_multiscale_overlap(
    mask1: np.ndarray,
    mask2: np.ndarray,
    scale_factors: List[float] = None,
    metrics: List[str] = None,
    weights: List[float] = None
) -> Dict[str, float]:
    """
    Compute overlap metrics at multiple scales for robust comparison.

    Args:
        mask1, mask2: Binary masks to compare
        scale_factors: List of scale factors to apply
        metrics: List of metrics to compute
        weights: Weights for combining metrics

    Returns:
        Dictionary of combined overlap scores
    """
    if scale_factors is None:
        scale_factors = [0.8, 1.0, 1.2]
    if metrics is None:
        metrics = ['jaccard', 'dice']
    if weights is None:
        weights = [0.5, 0.5]

    results = defaultdict(list)

    for scale in scale_factors:
        if scale == 1.0:
            m1_scaled, m2_scaled = mask1, mask2
        else:
            # Scale masks using morphological operations
            if scale < 1.0:
                # Shrink
                kernel_size = max(1, int(1/scale))
                kernel = disk(kernel_size) if len(mask1.shape) == 2 else ball(kernel_size)
                m1_scaled = binary_erosion(mask1, kernel)
                m2_scaled = binary_erosion(mask2, kernel)
            else:
                # Expand
                kernel_size = max(1, int(scale))
                kernel = disk(kernel_size) if len(mask1.shape) == 2 else ball(kernel_size)
                m1_scaled = binary_dilation(mask1, kernel)
                m2_scaled = binary_dilation(mask2, kernel)

        # Compute metrics
        intersection = np.sum(m1_scaled & m2_scaled)
        union = np.sum(m1_scaled | m2_scaled)

        if union > 0:
            jaccard = intersection / union
            dice = 2 * intersection / (np.sum(m1_scaled) + np.sum(m2_scaled))

            results['jaccard'].append(jaccard)
            results['dice'].append(dice)

    # Combine results across scales
    combined = {}
    for metric in metrics:
        if metric in results and results[metric]:
            combined[f'{metric}_mean'] = np.mean(results[metric])
            combined[f'{metric}_std'] = np.std(results[metric])
            combined[f'{metric}_min'] = np.min(results[metric])
            combined[f'{metric}_max'] = np.max(results[metric])

    # Weighted combination
    weighted_score = 0.0
    for i, metric in enumerate(metrics):
        if f'{metric}_mean' in combined:
            weighted_score += weights[i] * combined[f'{metric}_mean']

    combined['weighted_score'] = weighted_score
    return combined


def find_dominant_channel(
    z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
    channel_map: Dict[int, np.ndarray]
) -> Optional[int]:
    """Find dominant channel using spatial extent (number of z-slices)."""
    channel_scores = {}

    for ch, zdict in z_mask_map.items():
        z_extent = len(zdict)
        channel_scores[ch] = z_extent

    if not channel_scores:
        return None

    dominant_ch = max(channel_scores.items(), key=lambda x: x[1])[0]
    return dominant_ch


def calculate_clique_score(
    clique: List[int],
    z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]]
) -> float:
    """Calculate confidence score for a clique using multiscale overlap."""
    if len(clique) < 2:
        return 0.0

    scores = []
    for c1, c2 in combinations(clique, 2):
        score = calculate_overlap_from_z_maps(z_mask_map[c1], z_mask_map[c2])
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def compute_enhanced_features_for_punctum(
    i: int,
    punctum,
    z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
    channel_map: Dict[int, np.ndarray],
    config,
    adaptive_thresholds: Dict[int, float],
    axon_mask: np.ndarray = None,
    dendrite_mask: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute comprehensive features for a single punctum.

    Args:
        i: Punctum index
        punctum: Punctum object
        z_mask_map: Per-channel, per-z mask dictionary
        channel_map: Channel images
        config: Configuration parameters (with use_* flags)
        adaptive_thresholds: Pre-computed adaptive thresholds
        axon_mask: Axon mask from ch2 separation
        dendrite_mask: Dendrite mask from ch2 separation

    Returns:
        Dictionary of features matching feature_names.json
    """
    feat = {
        "punctum_id": i,
        "x": float(punctum.x),
        "y": float(punctum.y),
        "z": float(punctum.z)
    }

    chans = sorted(z_mask_map.keys())

    # =========================
    # Basic Channel Features
    # =========================

    total_pix_all = 0
    per_ch_pix = {ch: 0 for ch in [1, 2, 3, 4, 5]}

    # Note: Channel 4 is mGRASP (excluded), but feature names use "ch4" for channel 5
    # Map actual channel 5 to feature name "ch4"
    for ch in [1, 2, 3, 4, 5]:
        zmap = z_mask_map.get(ch, {})
        pix, bboxes = 0, []

        # Channel-specific morphological features
        morph_features = {
            'eccentricity': [],
            'skeleton_len': []
        }

        for z, (m, (y1, y2, x1, x2)) in zmap.items():
            pix += int(m.sum())
            bboxes.append((y2 - y1) * (x2 - x1))

            # Morphological features
            if getattr(config, 'use_morphological_features', True) and np.any(m):
                labeled = label(m.astype(np.uint8), connectivity=1)
                props_list = regionprops(labeled)

                if props_list:
                    for rp in props_list:
                        eccentricity = float(getattr(rp, "eccentricity", 0.0))
                        morph_features['eccentricity'].append(eccentricity)

                    # Skeleton length (medial axis length)
                    sk = skeletonize(m.astype(bool))
                    morph_features['skeleton_len'].append(float(sk.sum()))

        # Map channel 5 to feature name "ch4" (channel 4 is mGRASP, excluded)
        feature_ch = 4 if ch == 5 else ch

        # Store channel features
        feat[f"ch{feature_ch}_pix"] = pix
        feat[f"ch{feature_ch}_zcount"] = float(len(zmap))
        feat[f"ch{feature_ch}_bbox_area"] = float(np.mean(bboxes)) if bboxes else 0.0

        # Morphological features
        for morph_name, morph_values in morph_features.items():
            if morph_values:
                feat[f"ch{feature_ch}_{morph_name}_mean"] = float(np.mean(morph_values))
                feat[f"ch{feature_ch}_{morph_name}_std"] = float(np.std(morph_values))
            else:
                feat[f"ch{feature_ch}_{morph_name}_mean"] = 0.0
                feat[f"ch{feature_ch}_{morph_name}_std"] = 0.0

        per_ch_pix[ch] = pix
        total_pix_all += pix

    # Relative pixel counts (map channel 5 to ch4)
    denom = max(total_pix_all, 1)
    for ch in [1, 2, 3, 4, 5]:
        feature_ch = 4 if ch == 5 else ch
        feat[f"ch{feature_ch}_rel_pix"] = float(per_ch_pix[ch] / denom)

    # =========================
    # Enhanced Overlap Features
    # =========================

    if getattr(config, 'use_enhanced_features', True) and len(chans) > 1:
        pairs = list(combinations(chans, 2))
        jacc_list, dice_list = [], []

        for c1, c2 in pairs:
            shared = set(z_mask_map[c1].keys()) & set(z_mask_map[c2].keys())
            if not shared:
                continue

            # Combine masks across z-slices
            m1_combined = None
            m2_combined = None

            for z in shared:
                (m1, bb1) = z_mask_map[c1][z]
                (m2, bb2) = z_mask_map[c2][z]

                if m1_combined is None:
                    m1_combined = m1.copy()
                    m2_combined = m2.copy()
                else:
                    m1_combined = m1_combined | m1
                    m2_combined = m2_combined | m2

            if m1_combined is not None and m2_combined is not None:
                # Multi-scale overlap analysis
                overlap_metrics = compute_multiscale_overlap(
                    m1_combined, m2_combined,
                    scale_factors=[0.8, 1.0, 1.2],
                    metrics=['jaccard', 'dice'],
                    weights=[0.5, 0.5]
                )

                feat[f"pair_{c1}_{c2}_jacc_mean"] = overlap_metrics.get('jaccard_mean', 0.0)
                feat[f"pair_{c1}_{c2}_jacc_std"] = overlap_metrics.get('jaccard_std', 0.0)
                feat[f"pair_{c1}_{c2}_dice_mean"] = overlap_metrics.get('dice_mean', 0.0)
                feat[f"pair_{c1}_{c2}_weighted_score"] = overlap_metrics.get('weighted_score', 0.0)

                jacc_list.append(overlap_metrics.get('jaccard_mean', 0.0))
                dice_list.append(overlap_metrics.get('dice_mean', 0.0))

        # Aggregate overlap features
        feat["pairs_jacc_mean"] = float(np.mean(jacc_list)) if jacc_list else 0.0
        feat["pairs_dice_mean"] = float(np.mean(dice_list)) if dice_list else 0.0
        feat["pairs_jacc_max"] = float(np.max(jacc_list)) if jacc_list else 0.0
        feat["pairs_jacc_min"] = float(np.min(jacc_list)) if jacc_list else 0.0

    # =========================
    # Graph Features
    # =========================

    if getattr(config, 'use_graph_features', True) and len(chans) > 1:
        # Build overlap graph
        G = nx.Graph()
        G.add_nodes_from(chans)

        overlap_thresh = getattr(config, 'overlap_thresh', 0.75)
        for c1, c2 in combinations(chans, 2):
            jacc_key = f"pair_{c1}_{c2}_jacc_mean"
            if jacc_key in feat and feat[jacc_key] >= overlap_thresh:
                G.add_edge(c1, c2)

        # Graph metrics
        feat["graph_edge_count"] = G.number_of_edges()
        feat["graph_node_count"] = G.number_of_nodes()
        feat["graph_density"] = nx.density(G)

        # Clique features
        cliques = list(nx.find_cliques(G))
        valid_cliques = [c for c in cliques if len(c) >= 2]
        feat["graph_clique_count"] = len(valid_cliques)
        feat["graph_max_clique_size"] = max([len(c) for c in valid_cliques]) if valid_cliques else 0

        # Clique scores
        if valid_cliques:
            best_clique = max(valid_cliques, key=len)
            clique_scores = []
            for c1, c2 in combinations(best_clique, 2):
                jacc_key = f"pair_{c1}_{c2}_jacc_mean"
                if jacc_key in feat:
                    clique_scores.append(feat[jacc_key])
            feat["best_clique_score"] = float(np.mean(clique_scores)) if clique_scores else 0.0
        else:
            feat["best_clique_score"] = 0.0

    # =========================
    # Spatial Features
    # =========================

    if getattr(config, 'use_spatial_features', True):
        for ch in chans:
            zmap = z_mask_map.get(ch, {})
            if not zmap:
                continue

            # Combine masks across z-slices for spatial analysis
            combined_mask = None

            for z, (m, (y1, y2, x1, x2)) in zmap.items():
                if combined_mask is None:
                    combined_mask = m.copy()
                else:
                    combined_mask = combined_mask | m

            if combined_mask is not None:
                spatial_features = compute_spatial_features_mask_only(combined_mask)

                for key, value in spatial_features.items():
                    feat[f"ch{ch}_spatial_{key}"] = value

    # =========================
    # Axon/Dendrite Features
    # =========================

    if axon_mask is not None and dendrite_mask is not None:
        # Get punctum location
        x, y, z = int(punctum.x), int(punctum.y), int(punctum.z)

        # Check bounds
        if (0 <= z < axon_mask.shape[0] and
            0 <= y < axon_mask.shape[1] and
            0 <= x < axon_mask.shape[2]):

            # Axon/dendrite overlap at punctum location
            feat['axon_overlap'] = float(axon_mask[z, y, x])
            feat['dendrite_overlap'] = float(dendrite_mask[z, y, x])

            # Local region analysis (5x5x3 region around punctum)
            z_start, z_end = max(0, z-1), min(axon_mask.shape[0], z+2)
            y_start, y_end = max(0, y-2), min(axon_mask.shape[1], y+3)
            x_start, x_end = max(0, x-2), min(axon_mask.shape[2], x+3)

            local_axon = axon_mask[z_start:z_end, y_start:y_end, x_start:x_end]
            local_dendrite = dendrite_mask[z_start:z_end, y_start:y_end, x_start:x_end]

            feat['local_axon_density'] = float(np.mean(local_axon))
            feat['local_dendrite_density'] = float(np.mean(local_dendrite))
            feat['axon_dendrite_ratio'] = (feat['local_axon_density'] /
                                         max(feat['local_dendrite_density'], 1e-6))
        else:
            feat['axon_overlap'] = 0.0
            feat['dendrite_overlap'] = 0.0
            feat['local_axon_density'] = 0.0
            feat['local_dendrite_density'] = 0.0
            feat['axon_dendrite_ratio'] = 0.0
    else:
        feat['axon_overlap'] = 0.0
        feat['dendrite_overlap'] = 0.0
        feat['local_axon_density'] = 0.0
        feat['local_dendrite_density'] = 0.0
        feat['axon_dendrite_ratio'] = 0.0

    # =========================
    # Rule-Based Features
    # =========================

    detected_channels = list(z_mask_map.keys())

    # Rule-based features: Overlap scores between channel pairs
    overlap_scores = {}
    if len(detected_channels) > 1:
        pairs = list(combinations(detected_channels, 2))

        for c1, c2 in pairs:
            try:
                overlap_score = calculate_overlap_from_z_maps(
                    z_mask_map[c1], z_mask_map[c2]
                )
                overlap_scores[(c1, c2)] = overlap_score
                feat[f"rule_overlap_{c1}_{c2}"] = overlap_score
            except Exception:
                feat[f"rule_overlap_{c1}_{c2}"] = 0.0
                overlap_scores[(c1, c2)] = 0.0

    # Initialize overlap features for all possible pairs (even if not detected)
    for c1 in [1, 2, 3, 5]:
        for c2 in [1, 2, 3, 5]:
            if c1 < c2:
                if f"rule_overlap_{c1}_{c2}" not in feat:
                    feat[f"rule_overlap_{c1}_{c2}"] = 0.0

    # Rule-based features: Clique scores
    if len(detected_channels) > 1:
        # Build overlap graph using rule-based overlap threshold
        G = nx.Graph()
        G.add_nodes_from(detected_channels)

        overlap_thresh = getattr(config, 'overlap_thresh', 0.75)
        for (c1, c2), ov_score in overlap_scores.items():
            if ov_score >= overlap_thresh:
                G.add_edge(c1, c2)

        # Find cliques
        cliques = list(nx.find_cliques(G))
        valid_cliques = [c for c in cliques if len(c) >= 2]

        if valid_cliques:
            best_clique = max(valid_cliques, key=len)
            clique_score = calculate_clique_score(best_clique, z_mask_map)
            feat["rule_clique_score"] = clique_score
            feat["rule_clique_size"] = len(best_clique)
        else:
            feat["rule_clique_score"] = 0.0
            feat["rule_clique_size"] = 0
    else:
        feat["rule_clique_score"] = 0.0
        feat["rule_clique_size"] = 0

    # Rule-based features: Dominance (spatial extent)
    dominant_ch = find_dominant_channel(z_mask_map, channel_map)
    if dominant_ch:
        feat["rule_dominant_channel"] = dominant_ch
        feat["rule_dominant_z_extent"] = len(z_mask_map[dominant_ch])
    else:
        feat["rule_dominant_channel"] = 0
        feat["rule_dominant_z_extent"] = 0

    # Rule-based features: Aggregate overlap scores
    if overlap_scores:
        feat["rule_mean_overlap"] = float(np.mean(list(overlap_scores.values())))
        feat["rule_max_overlap"] = float(max(overlap_scores.values()))
    else:
        feat["rule_mean_overlap"] = 0.0
        feat["rule_max_overlap"] = 0.0

    # =========================
    # Pair-Specific Features
    # =========================

    # Add pair-specific overlap and presence features
    for c1, c2 in combinations([1, 2, 3, 5], 2):
        pair_key = f"pair_{c1}_{c2}"
        feat[f"{pair_key}_overlap"] = feat.get(f"rule_overlap_{c1}_{c2}", 0.0)
        feat[f"{pair_key}_ch1_present"] = 1.0 if c1 in detected_channels else 0.0
        feat[f"{pair_key}_ch2_present"] = 1.0 if c2 in detected_channels else 0.0

    # Pixel ratio features
    for c1, c2 in combinations([1, 2, 3, 5], 2):
        pix1 = feat.get(f"ch{c1}_pix", 0.0)
        pix2 = feat.get(f"ch{c2}_pix", 0.0)
        if pix2 > 0:
            feat[f"ch{c1}_pix_over_ch{c2}_pix"] = pix1 / pix2
        else:
            feat[f"ch{c1}_pix_over_ch{c2}_pix"] = 0.0

        rel1 = feat.get(f"ch{c1}_rel_pix", 0.0)
        rel2 = feat.get(f"ch{c2}_rel_pix", 0.0)
        feat[f"ch{c1}_rel_minus_ch{c2}_rel"] = rel1 - rel2

    return feat
