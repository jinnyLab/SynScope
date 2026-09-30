#!/usr/bin/env python3
"""
Inference using pairwise overlap prediction models.

Uses trained binary classifiers to predict which channel pairs overlap,
then builds graphs and finds cliques to determine the final class.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# Import from local util folder (relative imports)
util_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(util_dir))

# Try to import zimg from parent directory if needed
try:
    from zimg import ZImg, ZPuncta
except ImportError:
    parent_dir = util_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from zimg import ZImg, ZPuncta

from mGRASP_puncta_core_functions import (
    process_punctum_channels,
    normalize_channel_combination,
)
from mGRASP_puncta_feature import (
    compute_enhanced_features_for_punctum,
    calculate_overlap_from_z_maps,
)

from dataclasses import dataclass

@dataclass
class LightGBMConfig:
    """Configuration for feature computation."""
    use_enhanced_features: bool = True
    use_spatial_features: bool = True
    use_graph_features: bool = True
    use_morphological_features: bool = True
    use_axon_dendrite_features: bool = True
    use_rule_based_features: bool = False
    overlap_thresh: float = 0.60

def normalize_pair_overlap_thresholds(
    thresholds: Optional[Dict[Union[Tuple[int, int], str], float]] = None,
) -> Dict[Tuple[int, int], float]:
    """Normalize pair overlap thresholds to (min_ch, max_ch) tuples."""
    if not thresholds:
        return {}

    normalized: Dict[Tuple[int, int], float] = {}
    for key, value in thresholds.items():
        if isinstance(key, tuple):
            ch1, ch2 = int(key[0]), int(key[1])
        elif isinstance(key, str):
            parts = [p.strip() for p in key.replace("_", ",").split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid pair overlap threshold key: {key!r}. "
                    "Use (1, 2) or '1_2' format."
                )
            ch1, ch2 = int(parts[0]), int(parts[1])
        else:
            raise ValueError(
                f"Invalid pair overlap threshold key type: {type(key).__name__}. "
                "Use (ch1, ch2) tuples or 'ch1_ch2' strings."
            )
        normalized[(min(ch1, ch2), max(ch1, ch2))] = float(value)
    return normalized


def repair_persisted_model(model: Any) -> Any:
    """Rebuild LightGBM boosters pickled without a live C++ handle."""
    if lgb is None:
        return model
    if not isinstance(model, lgb.Booster):
        return model
    if hasattr(model, 'handle'):
        return model
    model_str = getattr(model, '_handle', None)
    if not isinstance(model_str, str) or not model_str:
        return model
    try:
        return lgb.Booster(model_str=model_str)
    except Exception:
        return model


class PairwiseOverlapClassifier:
    """Classifier that uses unified overlap model to predict channel combinations."""

    def __init__(
        self,
        model_dir: str,
        overlap_thresh: float = None,
        ch5_overlap_thresh: float = None,
        confidence_threshold: float = None,
        rule_overlap_thresh: float = 0.60,
        use_model_overlap_threshold: bool = False,
        pair_overlap_thresholds: Optional[Dict[Union[Tuple[int, int], str], float]] = None,
        ch2_overlap_thresh: float = None,
        allow_single_channel_2: bool = False,
        include_detected_channel_2: bool = False,
        ch3_overlap_thresh: float = None,
        allow_single_channel_3: bool = False,
        include_detected_channel_3: bool = False,
        allow_single_channel_4: bool = True,
        assign_all_detected_channels: bool = False,
    ):
        """
        Initialize the classifier.

        Args:
            model_dir: Directory containing trained unified overlap model
            overlap_thresh: Overlap probability threshold for channel pairs. When set and
                           use_model_overlap_threshold is False, this value is used instead
                           of the model's optimal threshold.
            ch5_overlap_thresh: Threshold for pairs involving channel 5 when channel 5 is
                               not detected. Defaults to overlap_thresh when not set.
            ch2_overlap_thresh: Threshold for pairs involving channel 2. Defaults to a
                               lenient value when include_detected_channel_2 is True.
            allow_single_channel_2: If True, allow predictions of channel 2 alone.
            include_detected_channel_2: If True, add channel 2 to the final prediction when
                                        it is detected and overlaps any other predicted
                                        channel above ch2_overlap_thresh.
            ch3_overlap_thresh: Threshold for pairs involving channel 3.
            allow_single_channel_3: If True, allow channel-3-only predictions when applicable.
            include_detected_channel_3: If True, add channel 3 to the final prediction when
                                        detected with sufficient overlap to another channel.
            assign_all_detected_channels: If True, assign all intensity-detected channels
                                          at the punctum (e.g. 1_2_3_4) instead of the
                                          largest ML overlap clique (e.g. 1_4).
            confidence_threshold: Minimum confidence to accept prediction (default: None, no thresholding)
                                 If set, predictions below this threshold will be marked as 'low_confidence'
            rule_overlap_thresh: Overlap threshold for rule-based features in feature computation (default: 0.60)
            use_model_overlap_threshold: If True, use thresholds stored in the trained model.
                                        If False, use overlap_thresh / pair_overlap_thresholds.
            pair_overlap_thresholds: Optional per-pair overlap thresholds, e.g.
                                     {(1, 2): 0.4, "2_3": 0.5}. Overrides overlap_thresh.
        """
        self.model_dir = model_dir
        self.rule_overlap_thresh = rule_overlap_thresh
        self.overlap_thresh_override = overlap_thresh
        self.use_model_overlap_threshold = use_model_overlap_threshold
        self.pair_overlap_thresholds = normalize_pair_overlap_thresholds(pair_overlap_thresholds)
        self._ch2_overlap_thresh_override = ch2_overlap_thresh
        self._ch3_overlap_thresh_override = ch3_overlap_thresh
        self._include_detected_channel_2 = include_detected_channel_2
        self._include_detected_channel_3 = include_detected_channel_3
        self._allow_single_channel_2 = allow_single_channel_2
        self._allow_single_channel_3 = allow_single_channel_3
        self._allow_single_channel_4 = allow_single_channel_4
        self.assign_all_detected_channels = assign_all_detected_channels
        self.channel_pair_overlap_thresholds: Dict[int, float] = {}
        self.include_detected_channels = set()
        self.allow_single_channels = set()
        # overlap_thresh will be set from model's optimal_threshold after loading
        self.overlap_thresh = None
        # Use very lenient threshold for channel 5 pairs to reduce false negatives
        # Analysis shows 81.0% of errors are removing channel 5 incorrectly
        # Recall for channel 5 classes is extremely low (2.4% for 1_2_3_5, 3.1% for 1_3_5)
        # Lowered to 0.20 to be much more inclusive (was 0.30, originally 0.40)
        if ch5_overlap_thresh is None:
            self.ch5_overlap_thresh = None
        else:
            self.ch5_overlap_thresh = ch5_overlap_thresh
        self.confidence_threshold = confidence_threshold  # For filtering low-confidence predictions
        self.unified_model = None  # Single unified model
        self.optimal_thresholds = {}  # Stores optimal threshold (from training)
        self.feature_names = None
        self._model_predict_warned = False

        # Load model
        self._load_models()
        if self.overlap_thresh_override is not None:
            self.overlap_thresh = float(self.overlap_thresh_override)
        if self.overlap_thresh is None:
            self.overlap_thresh = 0.5
        if self.ch5_overlap_thresh is None:
            self.ch5_overlap_thresh = self.overlap_thresh
        self._configure_special_channel_overlap_settings()
        self._log_overlap_thresholds()

    def _configure_special_channel_overlap_settings(self) -> None:
        """Resolve per-channel overlap thresholds and inclusion settings."""
        channel_settings = (
            (2, self._ch2_overlap_thresh_override, self._include_detected_channel_2, self._allow_single_channel_2),
            (3, self._ch3_overlap_thresh_override, self._include_detected_channel_3, self._allow_single_channel_3),
            (4, None, False, self._allow_single_channel_4),
        )
        for ch, override, include_flag, allow_flag in channel_settings:
            if override is not None:
                self.channel_pair_overlap_thresholds[ch] = float(override)
            elif include_flag:
                self.channel_pair_overlap_thresholds[ch] = min(self.overlap_thresh, 0.25)
            else:
                self.channel_pair_overlap_thresholds[ch] = self.overlap_thresh
            if include_flag:
                self.include_detected_channels.add(ch)
            if allow_flag:
                self.allow_single_channels.add(ch)

        self.ch2_overlap_thresh = self.channel_pair_overlap_thresholds.get(2, self.overlap_thresh)
        self.ch3_overlap_thresh = self.channel_pair_overlap_thresholds.get(3, self.overlap_thresh)
        self.include_detected_channel_2 = 2 in self.include_detected_channels
        self.include_detected_channel_3 = 3 in self.include_detected_channels
        self.allow_single_channel_2 = 2 in self.allow_single_channels
        self.allow_single_channel_3 = 3 in self.allow_single_channels
        self.allow_single_channel_4 = 4 in self.allow_single_channels

    @staticmethod
    def _channel_mask_score(
        ch: int,
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
    ) -> float:
        """Total foreground mask pixels for a channel at this punctum."""
        if ch not in z_mask_map:
            return 0.0
        score = 0.0
        for _z, (mask, _bbox) in z_mask_map[ch].items():
            score += float(np.sum(mask))
        return score

    def _dominant_channel(
        self,
        channels: List[int],
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
    ) -> int:
        """Pick the detected channel with the strongest mask at this punctum."""
        if not channels:
            raise ValueError("channels must not be empty")
        if len(channels) == 1:
            return channels[0]
        return max(
            channels,
            key=lambda ch: (self._channel_mask_score(ch, z_mask_map), -ch),
        )

    def _pair_overlap_score(
        self,
        ml_prob: float,
        actual_overlap: float,
    ) -> float:
        """Combine ML overlap probability with measured mask overlap for graph edges."""
        return float(max(ml_prob, actual_overlap))

    def _refine_single_channel_pair(
        self,
        prediction: str,
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
    ) -> str:
        """Keep single 1 or 4 when channels weakly co-localize at a 1+4 punctum."""
        pred_channels = [int(c) for c in str(prediction).split('_') if c.isdigit()]
        if set(pred_channels) != {1, 4}:
            return prediction

        score_1 = self._channel_mask_score(1, z_mask_map)
        score_4 = self._channel_mask_score(4, z_mask_map)
        if score_1 <= 0 and score_4 <= 0:
            return prediction
        if score_1 <= 0:
            return '4'
        if score_4 <= 0:
            return '1'

        actual_overlap = calculate_overlap_from_z_maps(
            z_mask_map.get(1, {}),
            z_mask_map.get(4, {}),
        )
        if actual_overlap < self.overlap_thresh:
            return '4' if score_4 >= score_1 else '1'

        ratio_1_to_4 = score_1 / score_4
        if ratio_1_to_4 < 0.35:
            return '4'
        if ratio_1_to_4 > 2.5:
            return '1'
        return '1_4'

    def _log_overlap_thresholds(self) -> None:
        threshold_source = "model" if self.use_model_overlap_threshold else "user"
        channel_parts = ", ".join(
            f"channel {ch} pairs={thr:.3f}"
            for ch, thr in sorted(self.channel_pair_overlap_thresholds.items())
        )
        print(
            f"[INFO] Overlap thresholds ({threshold_source}): "
            f"general={self.overlap_thresh:.3f}, "
            f"{channel_parts}, "
            f"channel 5 pairs={self.ch5_overlap_thresh:.3f}"
        )
        if self.pair_overlap_thresholds:
            print(f"[INFO] Pair-specific overlap thresholds: {self.pair_overlap_thresholds}")

    def overlap_threshold_dataframe(self) -> pd.DataFrame:
        """Return overlap thresholds used for pairwise channel classification."""
        rows = [
            {
                "threshold_type": "overlap_general",
                "channel_or_pair": "all_pairs",
                "threshold": float(self.overlap_thresh),
            },
            {
                "threshold_type": "overlap_channel_2_pairs",
                "channel_or_pair": "channel_2_pairs",
                "threshold": float(self.ch2_overlap_thresh),
            },
            {
                "threshold_type": "overlap_channel_3_pairs",
                "channel_or_pair": "channel_3_pairs",
                "threshold": float(self.ch3_overlap_thresh),
            },
            {
                "threshold_type": "overlap_channel_5_pairs",
                "channel_or_pair": "channel_5_pairs",
                "threshold": float(self.ch5_overlap_thresh),
            },
        ]
        for (ch1, ch2), thresh in sorted(self.pair_overlap_thresholds.items()):
            rows.append(
                {
                    "threshold_type": "overlap_pair",
                    "channel_or_pair": f"{ch1}_{ch2}",
                    "threshold": float(thresh),
                }
            )
        return pd.DataFrame(rows)

    def print_overlap_thresholds(self) -> None:
        """Print overlap thresholds used during classification."""
        print("  Overlap thresholds:")
        print(f"    General pairs: {self.overlap_thresh:.4f}")
        for ch, thresh in sorted(self.channel_pair_overlap_thresholds.items()):
            print(f"    Channel {ch} pairs: {thresh:.4f}")
        print(f"    Channel 5 pairs: {self.ch5_overlap_thresh:.4f}")
        for (ch1, ch2), thresh in sorted(self.pair_overlap_thresholds.items()):
            print(f"    Pair ({ch1}, {ch2}): {thresh:.4f}")

    def _load_models(self):
        """Load unified overlap model."""
        unified_model_path = None
        for name in ('unified_overlap_model.pkl', 'model.pkl'):
            candidate = os.path.join(self.model_dir, name)
            if os.path.exists(candidate):
                unified_model_path = candidate
                break
        if unified_model_path is None:
            raise FileNotFoundError(
                f"No model file found in {self.model_dir}\n"
                "Expected unified_overlap_model.pkl or model.pkl"
            )

        print(f"[INFO] Loading unified overlap model from {unified_model_path}")
        with open(unified_model_path, 'rb') as f:
            loaded_data = pickle.load(f)

        if isinstance(loaded_data, dict):
            self.unified_model = loaded_data.get('model', loaded_data)
            self.feature_names = loaded_data.get('feature_names', None)
            optimal_thresh = loaded_data.get('optimal_threshold', 0.5)
            if optimal_thresh is None:
                optimal_thresh = 0.5
            self.overlap_thresh = float(optimal_thresh)
            self.optimal_thresholds['default'] = float(optimal_thresh)
        else:
            # Direct model object - use default
            self.unified_model = loaded_data
            self.overlap_thresh = 0.5

        self.unified_model = repair_persisted_model(self.unified_model)

        # Try to load feature names from JSON if not in model dict
        if self.feature_names is None:
            feature_path = os.path.join(self.model_dir, 'feature_names.json')
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)

        if self.feature_names is None:
            raise FileNotFoundError(
                f"Feature names not found. Expected in {os.path.join(self.model_dir, 'feature_names.json')} or in unified model dictionary."
            )

        print(f"[INFO] Unified model loaded successfully")
        print(f"[INFO] Feature names: {len(self.feature_names)} features")

    def _predict_overlap_probability(
        self,
        feature_array: np.ndarray,
        actual_overlap: float,
        ch1: int,
        ch2: int,
    ) -> float:
        """Run unified overlap model; fall back to measured overlap on failure."""
        model = self.unified_model
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(feature_array)[0, 1]
            elif hasattr(model, 'predict'):
                prob = float(model.predict(feature_array)[0])
                if prob in (0, 1):
                    prob = float(actual_overlap)
            else:
                raise AttributeError("Unified model has no predict or predict_proba method")
            return float(prob)
        except Exception as e:
            if not self._model_predict_warned:
                print(
                    f"  [WARN] Unified model prediction failed (e.g. pair ({ch1}, {ch2})): "
                    f"{e}; using actual overlap for all pairs"
                )
                self._model_predict_warned = True
            return float(actual_overlap)

    def _validate_channel_5_prediction(
        self,
        prediction: str,
        overlap_probs: Dict[Tuple[int, int], float],
        detected_channels: List[int],
        overall_confidence: float = None
    ) -> str:
        """
        Validate that channel 5 is actually present with sufficient overlap.
        If prediction contains channel 5 but overlap is too low, remove it.

        Args:
            prediction: Predicted label (e.g., "1_3_5")
            overlap_probs: Dictionary of overlap probabilities for each pair
            detected_channels: List of detected channels
            overall_confidence: Overall confidence score for the prediction (optional)

        Returns:
            Validated prediction (may have channel 5 removed)
        """
        # Check if prediction contains channel 5 (or channel 4, which may be mapped to 5)
        pred_channels = [int(c) for c in str(prediction).split('_') if c.isdigit()]
        has_ch5 = 5 in pred_channels or 4 in pred_channels

        if not has_ch5:
            return prediction

        # Single-channel 4 is valid on its own; do not strip it during validation.
        if pred_channels == [4]:
            return prediction

        # Check overlap with channel 5
        ch5_pairs = []
        for ch in pred_channels:
            if ch != 5 and ch != 4:
                # Check pairs (ch, 5) and (ch, 4)
                if 5 in detected_channels:
                    ch5_pairs.append((min(ch, 5), max(ch, 5)))
                if 4 in detected_channels:
                    ch5_pairs.append((min(ch, 4), max(ch, 4)))

        # Get maximum overlap with channel 5 and all channel 5 pair probabilities
        max_ch5_overlap = 0.0
        all_ch5_probs = []
        for pair in ch5_pairs:
            prob = 0.0
            if pair in overlap_probs:
                prob = overlap_probs[pair]
                max_ch5_overlap = max(max_ch5_overlap, prob)
            # Also check reverse
            rev_pair = (pair[1], pair[0])
            if rev_pair in overlap_probs:
                prob = overlap_probs[rev_pair]
                max_ch5_overlap = max(max_ch5_overlap, prob)
            if prob > 0:
                all_ch5_probs.append((pair, prob))

        # Use a more lenient threshold for validation to reduce false negatives
        # Analysis shows 82.9% of errors are removing channel 5 incorrectly
        # Use a lower threshold that matches the graph building logic
        # If optimal thresholds are available, use them with the same lenient multiplier
        strict_threshold = self.ch5_overlap_thresh  # Default fallback

        # If we have optimal thresholds, use them with lenient multiplier
        if all_ch5_probs:
            # Use default optimal threshold if available
            if 'default' in self.optimal_thresholds and self.optimal_thresholds['default'] is not None:
                # CRITICAL: Models predict extremely low probabilities (mean 0.022)
                # Use extremely low threshold as workaround
                min_optimal = self.optimal_thresholds['default']
                strict_threshold = max(min_optimal * 0.3, 0.05)  # Very aggressive multiplier + low minimum
            else:
                # No optimal thresholds, use extremely low default due to model under-prediction
                strict_threshold = 0.05  # Extremely low due to model under-prediction

        # DETECTION-BASED APPROACH: Since we're using detection as the primary signal,
        # be extremely conservative about removing channel 5. Only remove if:
        # 1. Channel 5 is NOT detected (shouldn't happen if added by post-processing), OR
        # 2. Confidence is extremely low (< 0.01) AND overlap is 0.0
        # This ensures channel 5 added by detection-based post-processing is rarely removed

        # Check if channel 5 is actually detected
        ch5_actually_detected = 5 in detected_channels or 4 in detected_channels
        if not ch5_actually_detected:
            # Channel 5 not detected - remove it (shouldn't happen if added correctly)
            pred_channels = [c for c in pred_channels if c != 5 and c != 4]
            if pred_channels:
                from mGRASP_puncta_core_functions import normalize_channel_combination
                new_pred = normalize_channel_combination(pred_channels)
                return new_pred

        # Only remove channel 5 if BOTH:
        # 1. Overlap is exactly 0.0 (no probability at all) AND
        # 2. Confidence is extremely low (< 0.01)
        # This is extremely conservative - we want to keep channel 5 if it's detected
        if max_ch5_overlap == 0.0 and overall_confidence is not None and overall_confidence < 0.01:
            pred_channels = [c for c in pred_channels if c != 5 and c != 4]
            if pred_channels:
                from mGRASP_puncta_core_functions import normalize_channel_combination
                new_pred = normalize_channel_combination(pred_channels)
                return new_pred
            else:
                # All channels removed - return low confidence
                return 'low_confidence'
        return prediction

    def _check_channel_2_axon_dendrite_coverage(
        self,
        ch2_z_masks: Dict[int, Tuple[np.ndarray, Tuple]],
        axon_dendrite_channel: int = 2
    ) -> Tuple[float, float]:
        """
        Check axon and dendrite coverage for channel 2.

        Returns:
            Tuple of (axon_coverage, dendrite_coverage) as ratios (0.0 to 1.0)
        """
        if not hasattr(self, 'axon_mask') or self.axon_mask is None:
            return (0.0, 0.0)
        if not hasattr(self, 'dendrite_mask') or self.dendrite_mask is None:
            return (0.0, 0.0)

        axon_mask = self.axon_mask
        dendrite_mask = self.dendrite_mask

        total_ch2_pixels = 0
        axon_ch2_pixels = 0
        dendrite_ch2_pixels = 0

        for z, (mask, (y1, y2, x1, x2)) in ch2_z_masks.items():
            if 0 <= z < axon_mask.shape[0]:
                if (0 <= y1 < axon_mask.shape[1] and y2 <= axon_mask.shape[1] and
                    0 <= x1 < axon_mask.shape[2] and x2 <= axon_mask.shape[2]):
                    axon_roi = axon_mask[z, y1:y2, x1:x2]
                    dendrite_roi = dendrite_mask[z, y1:y2, x1:x2]
                    total_ch2_pixels += np.sum(mask)
                    axon_ch2_pixels += np.sum(mask & axon_roi.astype(bool))
                    dendrite_ch2_pixels += np.sum(mask & dendrite_roi.astype(bool))

        if total_ch2_pixels == 0:
            return (0.0, 0.0)

        axon_coverage = axon_ch2_pixels / total_ch2_pixels
        dendrite_coverage = dendrite_ch2_pixels / total_ch2_pixels

        return (axon_coverage, dendrite_coverage)

    def _get_pair_threshold(
        self,
        pair_key: Tuple[int, int],
        ch1: int,
        ch2: int,
        detected_channels: List[int]
    ) -> float:
        """Get threshold for a channel pair."""
        normalized_pair = (min(ch1, ch2), max(ch1, ch2))
        if normalized_pair in self.pair_overlap_thresholds:
            return self.pair_overlap_thresholds[normalized_pair]

        if not self.use_model_overlap_threshold:
            is_ch5_pair = 5 in (ch1, ch2) or 4 in (ch1, ch2)
            if is_ch5_pair:
                ch5_detected = (5 in detected_channels and 5 in (ch1, ch2)) or \
                              (4 in detected_channels and 4 in (ch1, ch2))
                if ch5_detected:
                    return 0.0
                return self.ch5_overlap_thresh
            for ch in sorted(self.channel_pair_overlap_thresholds):
                if ch in (ch1, ch2):
                    return self.channel_pair_overlap_thresholds[ch]
            return self.overlap_thresh

        is_ch5_pair = 5 in (ch1, ch2) or 4 in (ch1, ch2)
        is_ch2_pair = 2 in (ch1, ch2)

        if is_ch5_pair:
            ch5_detected = (5 in detected_channels and 5 in (ch1, ch2)) or \
                          (4 in detected_channels and 4 in (ch1, ch2))
            if ch5_detected:
                return 0.0  # Always include if channel 5 is detected
            else:
                # Use optimal threshold if available, otherwise use ch5 threshold
                if 'default' in self.optimal_thresholds and self.optimal_thresholds['default'] is not None:
                    return self.optimal_thresholds['default']
                return self.ch5_overlap_thresh
        elif is_ch2_pair:
            # Use optimal threshold if available, otherwise use default
            if 'default' in self.optimal_thresholds and self.optimal_thresholds['default'] is not None:
                return max(self.optimal_thresholds['default'] * 1.5, 0.6)
            return 0.6
        else:
            # Use optimal threshold if available, otherwise use default overlap threshold
            if 'default' in self.optimal_thresholds and self.optimal_thresholds['default'] is not None:
                return self.optimal_thresholds['default']
            return self.overlap_thresh

    def _should_skip_ch2_edge(
        self,
        ch1: int,
        ch2: int,
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]]
    ) -> bool:
        """Check if channel 2 edge should be skipped based on axon/dendrite coverage."""
        if 2 not in (ch1, ch2):
            return False
        if not hasattr(self, 'dendrite_mask') or self.dendrite_mask is None:
            return False
        if 2 not in z_mask_map:
            return False

        axon_coverage, dendrite_coverage = self._check_channel_2_axon_dendrite_coverage(z_mask_map[2])
        return dendrite_coverage > 0.8 and axon_coverage < 0.2

    def _build_overlap_graph(
        self,
        overlap_probs: Dict[Tuple[int, int], float],
        detected_channels: List[int],
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]]
    ) -> Tuple[nx.Graph, float]:
        """Build graph from overlap probabilities and return graph with minimum confidence."""
        G = nx.Graph()
        G.add_nodes_from(detected_channels)
        min_confidence = 1.0

        for (ch1, ch2), prob in overlap_probs.items():
            # Skip channel 2 edges if in dendrite regions
            if self._should_skip_ch2_edge(ch1, ch2, z_mask_map):
                continue

            # Get threshold for this pair
            pair_key = (ch1, ch2) if (ch1, ch2) in self.optimal_thresholds else (ch2, ch1)
            threshold = self._get_pair_threshold(pair_key, ch1, ch2, detected_channels)

            if prob >= threshold:
                G.add_edge(ch1, ch2)
                min_confidence = min(min_confidence, prob)

        return G, min_confidence

    def _score_clique(
        self,
        clique: List[int],
        overlap_probs: Dict[Tuple[int, int], float],
        detected_channels: List[int]
    ) -> Tuple[float, float]:
        """Score a clique by average probability and size bonus."""
        clique_probs = []
        for ch1, ch2 in combinations(clique, 2):
            pair_key = (ch1, ch2) if (ch1, ch2) in overlap_probs else (ch2, ch1)
            if pair_key in overlap_probs:
                clique_probs.append(overlap_probs[pair_key])

        avg_prob = float(np.mean(clique_probs)) if clique_probs else 0.0

        # Size bonus
        size_bonus = 0.0
        ch5_bonus = 0.0
        ch5_detected = 5 in detected_channels or 4 in detected_channels

        if ch5_detected and (5 in clique or 4 in clique):
            ch5_bonus = 0.3
            size_bonus = 0.5

        score = (len(clique) + size_bonus, avg_prob + ch5_bonus)
        return avg_prob, score

    def _get_pair_overlap_prob(
        self,
        ch1: int,
        ch2: int,
        overlap_probs: Dict[Tuple[int, int], float],
    ) -> float:
        pair = (min(ch1, ch2), max(ch1, ch2))
        if pair in overlap_probs:
            return float(overlap_probs[pair])
        reverse_pair = (pair[1], pair[0])
        return float(overlap_probs.get(reverse_pair, 0.0))

    def _maybe_include_detected_channels(
        self,
        prediction: str,
        detected_channels: List[int],
        overlap_probs: Dict[Tuple[int, int], float],
    ) -> str:
        """Add specially handled channels when detected with sufficient pairwise overlap."""
        if not self.include_detected_channels or prediction == "low_confidence":
            return prediction

        pred_channels = [int(c) for c in str(prediction).split("_") if c.isdigit()]
        updated = False

        for ch in sorted(self.include_detected_channels):
            if ch not in detected_channels or ch in pred_channels:
                continue
            pair_thresh = self.channel_pair_overlap_thresholds.get(ch, self.overlap_thresh)
            for other_ch in detected_channels:
                if other_ch == ch:
                    continue
                prob = self._get_pair_overlap_prob(ch, other_ch, overlap_probs)
                if prob >= pair_thresh:
                    pred_channels.append(ch)
                    updated = True
                    break

        if updated:
            return normalize_channel_combination(sorted(set(pred_channels)))
        return prediction

    def _expand_prediction_to_detected_channels(
        self,
        prediction: str,
        detected_channels: List[int],
        overlap_probs: Dict[Tuple[int, int], float],
    ) -> str:
        """Iteratively add detected channels that overlap the current prediction set."""
        if prediction == "low_confidence":
            return prediction

        pred_channels = {int(c) for c in str(prediction).split("_") if c.isdigit()}
        changed = True
        while changed:
            changed = False
            for ch in detected_channels:
                if ch in pred_channels:
                    continue
                pair_thresh = self.channel_pair_overlap_thresholds.get(ch, self.overlap_thresh)
                for other_ch in pred_channels:
                    prob = self._get_pair_overlap_prob(ch, other_ch, overlap_probs)
                    if prob >= pair_thresh:
                        pred_channels.add(ch)
                        changed = True
                        break
        return normalize_channel_combination(sorted(pred_channels))

    def _prediction_from_all_detected_channels(
        self,
        detected_channels: List[int],
        overlap_probs: Dict[Tuple[int, int], float],
    ) -> Dict[str, Any]:
        """Build a prediction label from every channel detected at the punctum."""
        pair_probs = [
            self._get_pair_overlap_prob(ch1, ch2, overlap_probs)
            for ch1, ch2 in combinations(detected_channels, 2)
        ]
        confidence = float(np.mean(pair_probs)) if pair_probs else 0.0
        prediction = normalize_channel_combination(detected_channels)
        if self.confidence_threshold is not None and confidence < self.confidence_threshold:
            prediction = "low_confidence"
        return {
            "prediction": prediction,
            "confidence": confidence,
            "method": "all_detected_channels",
        }

    def _handle_single_channel(self, ch: int) -> Dict[str, Any]:
        """Handle single channel case."""
        if ch in self.allow_single_channels or ch not in (2, 5):
            return {
                'prediction': str(ch),
                'confidence': 0.8,
                'method': 'single_channel'
            }
        return {
            'prediction': 'low_confidence',
            'confidence': 0.0,
            'method': 'single_channel_rejected'
        }


    def _predict_overlaps_simple(
        self,
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
        detected_channels: List[int]
    ) -> Dict[Tuple[int, int], float]:
        """
        Simple fallback method that uses actual overlap scores when feature computation is not available.
        Uses multiscale overlap for robustness.

        Returns:
            Dictionary mapping (ch1, ch2) to overlap probability (using actual overlap as proxy)
        """
        overlap_probs = {}
        for ch1, ch2 in combinations(detected_channels, 2):
            actual_overlap = calculate_overlap_from_z_maps(
                z_mask_map[ch1], z_mask_map[ch2]
            )
            # Use actual overlap as probability (normalize to 0-1 range)
            overlap_probs[(ch1, ch2)] = float(actual_overlap)
        return overlap_probs

    def predict_overlaps(
        self,
        punctum,
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
        channel_map: Dict[int, np.ndarray],
        adaptive_thresholds: Dict[int, float],
        axon_mask: Optional[np.ndarray] = None,
        dendrite_mask: Optional[np.ndarray] = None,
        config: Optional[LightGBMConfig] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Predict overlap probabilities for all channel pairs.

        Returns:
            Dictionary mapping (ch1, ch2) to overlap probability
        """
        detected_channels = sorted(z_mask_map.keys())

        if len(detected_channels) < 2:
            return {}

        # Use feature computation if models are available
        if config is None:
            config = LightGBMConfig(
                use_enhanced_features=True,
                use_spatial_features=True,
                use_graph_features=True,
                use_morphological_features=True,
                use_axon_dendrite_features=True,
                use_rule_based_features=False,
                overlap_thresh=self.rule_overlap_thresh
            )

        try:
            # Compute features for this punctum
            features = compute_enhanced_features_for_punctum(
                0, punctum, z_mask_map, channel_map, config,
                adaptive_thresholds, axon_mask, dendrite_mask
            )
        except Exception:
            # Fallback to simple overlap if feature computation fails
            return self._predict_overlaps_simple(z_mask_map, detected_channels)

        if features is None:
            return self._predict_overlaps_simple(z_mask_map, detected_channels)

        # Predict overlap for each pair using models
        overlap_probs = {}

        # Use unified model
        # For unified model, we need to add pair-specific features for each pair
        # (the model was trained with pair-specific features)
        for ch1, ch2 in combinations(detected_channels, 2):
            # Create pair-specific features
            pair_features = features.copy()

            # Calculate actual overlap for this pair
            actual_overlap = calculate_overlap_from_z_maps(
                z_mask_map[ch1], z_mask_map[ch2]
            )

            # Add pair-specific features (matching training format)
            pair_features[f'pair_{ch1}_{ch2}_overlap'] = actual_overlap
            pair_features[f'pair_{ch1}_{ch2}_ch1_present'] = 1.0
            pair_features[f'pair_{ch1}_{ch2}_ch2_present'] = 1.0

            # Convert to feature array
            feature_array = np.array([[pair_features.get(f, 0.0) for f in self.feature_names]])

            prob = self._predict_overlap_probability(
                feature_array, actual_overlap, ch1, ch2,
            )
            overlap_probs[(ch1, ch2)] = self._pair_overlap_score(prob, actual_overlap)

        return overlap_probs

    def predict_single_punctum(
        self,
        punctum,
        z_mask_map: Dict[int, Dict[int, Tuple[np.ndarray, Tuple]]],
        channel_map: Dict[int, np.ndarray],
        adaptive_thresholds: Dict[int, float],
        axon_mask: Optional[np.ndarray] = None,
        dendrite_mask: Optional[np.ndarray] = None,
        mgrasp_channel: int = 4
    ) -> Dict[str, Any]:
        """
        Predict classification for a single punctum.

        Returns:
            Dictionary with prediction, confidence, and method
        """
        # Store masks for validation
        self.axon_mask = axon_mask
        self.dendrite_mask = dendrite_mask

        detected_channels = sorted(z_mask_map.keys())

        # Single channel case
        if len(detected_channels) == 1:
            return self._handle_single_channel(detected_channels[0])

        # Multi-channel case: predict overlaps and build graph
        overlap_probs = self.predict_overlaps(
            punctum, z_mask_map, channel_map, adaptive_thresholds,
            axon_mask, dendrite_mask
        )

        if not overlap_probs:
            return {
                'prediction': 'low_confidence',
                'confidence': 0.0,
                'method': 'no_overlap_predictions'
            }

        if self.assign_all_detected_channels and len(detected_channels) >= 2:
            return self._prediction_from_all_detected_channels(detected_channels, overlap_probs)

        # Build graph from predicted overlaps
        G, min_confidence = self._build_overlap_graph(overlap_probs, detected_channels, z_mask_map)

        # Find cliques
        cliques = list(nx.find_cliques(G))
        valid_cliques = [sorted(c) for c in cliques if len(c) >= 2]

        if not valid_cliques:
            connected_components = [sorted(comp) for comp in nx.connected_components(G) if len(comp) >= 2]
            if connected_components:
                best_component = max(connected_components, key=len)
                component_probs = []
                for ch1, ch2 in combinations(best_component, 2):
                    component_probs.append(self._get_pair_overlap_prob(ch1, ch2, overlap_probs))
                confidence = float(np.mean(component_probs)) if component_probs else min_confidence
                prediction = self._expand_prediction_to_detected_channels(
                    self._maybe_include_detected_channels(
                        normalize_channel_combination(best_component),
                        detected_channels,
                        overlap_probs,
                    ),
                    detected_channels,
                    overlap_probs,
                )

                if self.confidence_threshold is not None and confidence < self.confidence_threshold:
                    prediction = 'low_confidence'

                prediction = self._refine_single_channel_pair(prediction, z_mask_map)
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'method': 'pairwise_overlap_connected_component'
                }

            if len(detected_channels) == 1:
                return self._handle_single_channel(detected_channels[0])

            isolated_nodes = [ch for ch in detected_channels if G.degree(ch) == 0]
            if isolated_nodes:
                dominant = self._dominant_channel(isolated_nodes, z_mask_map)
                return self._handle_single_channel(dominant)

            return {
                'prediction': 'low_confidence',
                'confidence': 0.0,
                'method': 'no_cliques'
            }

        # Score and select best clique
        ch5_detected = 5 in detected_channels or 4 in detected_channels
        clique_scores = []
        for clique in valid_cliques:
            avg_prob, score = self._score_clique(clique, overlap_probs, detected_channels)
            clique_scores.append((clique, avg_prob, score))

        # Sort by score (size first, then probability)
        clique_scores.sort(key=lambda x: x[2], reverse=True)
        best_clique, best_avg_prob, _ = clique_scores[0]

        # Calculate confidence and normalize prediction
        confidence = best_avg_prob if best_avg_prob > 0 else min_confidence
        prediction = self._expand_prediction_to_detected_channels(
            self._maybe_include_detected_channels(
                normalize_channel_combination(best_clique),
                detected_channels,
                overlap_probs,
            ),
            detected_channels,
            overlap_probs,
        )
        prediction = self._refine_single_channel_pair(prediction, z_mask_map)

        # Validate channel 5 predictions
        prediction = self._validate_channel_5_prediction(
            prediction, overlap_probs, detected_channels, overall_confidence=confidence
        )

        # Apply confidence thresholding if enabled
        if self.confidence_threshold is not None and confidence < self.confidence_threshold:
            prediction = 'low_confidence'

        return {
            'prediction': prediction,
            'confidence': confidence,
            'method': 'pairwise_overlap_graph'
        }
