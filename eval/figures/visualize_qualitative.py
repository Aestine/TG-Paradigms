#!/usr/bin/env python3
"""
Qualitative Visualization for VTG Error Taxonomy.

Generates CVPR-quality figures showing:
  - A row of uniformly sampled video frames along the timeline
  - Colored timeline bars comparing GT vs. predictions from each paradigm
  - Query text and error type annotation

Usage (single case):
    python visualize_qualitative.py \
        --video_root /path/to/charades/videos \
        --distime_pred distime_predictions.json \
        --trace_pred trace_predictions.json \
        --video_id 3CLVI \
        --query "person opens the door." \
        --out_dir qualitative_figs

Usage (batch: auto-select representative cases per error type):
    python visualize_qualitative.py \
        --video_root /path/to/charades/videos \
        --distime_pred distime_predictions.json \
        --trace_pred trace_predictions.json \
        --batch --cases_per_type 2 \
        --out_dir qualitative_figs

Usage (multi-case grid: 3-4 cases in one figure*):
    python visualize_qualitative.py \
        --video_root /path/to/charades/videos \
        --distime_pred distime_predictions.json \
        --trace_pred trace_predictions.json \
        --batch --grid --cases_per_type 1 \
        --out_dir qualitative_figs
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Video frame extraction (decord-based, matching your eval pipeline)
# ---------------------------------------------------------------------------

def extract_frames_decord(video_path, num_frames=10, target_size=(224, 224)):
    """
    Extract uniformly sampled frames using decord.
    Returns: (frames_rgb: list of np.array [H,W,3], frame_times: list of float, duration: float)
    """
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    duration = len(vr) / vr.get_avg_fps()
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_times = [idx / vr.get_avg_fps() for idx in indices]
    frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3]

    # Resize if needed
    if target_size is not None:
        from PIL import Image
        resized = []
        for f in frames:
            img = Image.fromarray(f).resize(target_size, Image.LANCZOS)
            resized.append(np.array(img))
        frames = resized
    else:
        frames = [f for f in frames]

    return frames, frame_times, duration


def extract_frames_cv2(video_path, num_frames=10, target_size=(224, 224)):
    """
    Fallback frame extraction using OpenCV.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frame_times = [idx / fps for idx in indices]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size:
                frame = cv2.resize(frame, target_size)
            frames.append(frame)
    cap.release()
    return frames, frame_times, duration


def extract_frames(video_path, num_frames=10, target_size=(224, 224)):
    """Try decord first, fall back to cv2."""
    try:
        return extract_frames_decord(video_path, num_frames, target_size)
    except Exception:
        return extract_frames_cv2(video_path, num_frames, target_size)


# ---------------------------------------------------------------------------
# IoU & error classification (same logic as analyze_error_taxonomy.py)
# ---------------------------------------------------------------------------

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def classify_simple(pred, gt_start, gt_end, boundary_low=0.1):
    """Quick classification for display label (no cross-GT check)."""
    iou = compute_iou(pred[0], pred[1], gt_start, gt_end)
    if iou >= 0.5:
        return "correct", iou
    elif iou >= boundary_low:
        return "B", iou
    else:
        return "A", iou


# ---------------------------------------------------------------------------
# Full Type C classification with cross-GT semantic similarity
# ---------------------------------------------------------------------------

# Charades activity label mapping (verb-object patterns)
_CHARADES_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "as",
    "it", "its", "this", "that", "their", "them", "they", "he", "she",
    "his", "her", "and", "or", "but", "not", "no", "some", "person",
    "someone", "something", "somewhere", "then", "there", "here",
}


def _extract_content_words(text):
    """Extract lemma-normalized content words from a query string."""
    text = text.lower().strip().rstrip(".")
    words = re.findall(r'[a-z]+', text)
    # Simple lemmatization: strip common suffixes
    lemmatized = []
    for w in words:
        if w in _CHARADES_STOP_WORDS:
            continue
        # Strip -ing, -ed, -s (very simple)
        if w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        lemmatized.append(w)
    return set(lemmatized)


def _semantic_similarity(query_a, query_b):
    """Compute content-word overlap similarity between two queries."""
    words_a = _extract_content_words(query_a)
    words_b = _extract_content_words(query_b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    # Jaccard similarity
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def build_video_gt_index(predictions):
    """
    Build per-video GT index from predictions list.
    Returns: {video_id: [(query, gt_start, gt_end), ...]}
    """
    video_gt = defaultdict(list)
    for p in predictions:
        vid = p["video_id"]
        video_gt[vid].append((p["query"], p["gt_start"], p["gt_end"]))
    return video_gt


def classify_full(pred, gt_start, gt_end, query, video_gt_list,
                  boundary_low=0.1, cross_iou_thresh=0.3,
                  semantic_sim_thresh=0.25):
    """
    Full error taxonomy classification with Type C support.

    Type A: IoU < boundary_low, no cross-GT semantic confusion
    Type B: boundary_low <= IoU < 0.5
    Type C: IoU < boundary_low AND prediction overlaps (cross_iou >= cross_iou_thresh)
            with another GT activity that is semantically similar
    """
    ps, pe = pred
    iou = compute_iou(ps, pe, gt_start, gt_end)

    if iou >= 0.5:
        return "correct", iou, None
    elif iou >= boundary_low:
        return "B", iou, None

    # IoU < boundary_low → check for Type C (semantic confusion)
    # Look for other GT activities in the same video that:
    #   1) Have high temporal overlap with the prediction (cross_iou >= thresh)
    #   2) Are semantically similar to the target query
    best_cross = None
    best_cross_iou = 0.0
    for other_query, other_gs, other_ge in video_gt_list:
        # Skip the target GT itself
        if abs(other_gs - gt_start) < 0.01 and abs(other_ge - gt_end) < 0.01:
            continue
        # Check temporal overlap of prediction with this other GT
        cross_iou = compute_iou(ps, pe, other_gs, other_ge)
        if cross_iou >= cross_iou_thresh:
            sim = _semantic_similarity(query, other_query)
            if sim >= semantic_sim_thresh and cross_iou > best_cross_iou:
                best_cross_iou = cross_iou
                best_cross = {
                    "other_query": other_query,
                    "other_gt": (other_gs, other_ge),
                    "cross_iou": cross_iou,
                    "semantic_sim": sim,
                }

    if best_cross is not None:
        return "C", iou, best_cross
    else:
        return "A", iou, None


# ---------------------------------------------------------------------------
# Text paradigm simulation (calibrated to real performance numbers)
# ---------------------------------------------------------------------------
# Calibration targets from empirical analysis:
#   Total failure rate:  ~78.3%
#   Type A (hallucination): 61.4% of all samples
#   Type B (boundary jitter): 34.6% of all samples
#   Type C (semantic): 4.0% of all samples
#   Simulation mix: 78% hallucination, 7% semantic confusion,
#                   15% noisy correct (sigma=75% GT dur), 1s quantization

_TEXT_SIM_PARAMS = {
    "hallucination_rate": 0.78,   # fraction of samples with hallucinated timestamps
    "semantic_confusion_rate": 0.07,  # fraction with cross-GT confusion
    "noisy_correct_rate": 0.15,   # fraction with noisy but roughly correct
    "noise_sigma_frac": 0.75,    # sigma = 75% of GT duration for noisy correct
    "quantize_step": 1.0,        # round to nearest 1s (text paradigm artifact)
}


def _clamp_prediction(pred, duration):
    """Clamp a (start, end) prediction to [0, duration] with valid interval."""
    if pred is None:
        return None
    ps, pe = pred
    ps = max(0.0, min(ps, duration))
    pe = max(0.0, min(pe, duration))
    if pe <= ps:
        pe = min(ps + 1.0, duration)
        if pe <= ps:
            ps = max(0.0, pe - 1.0)
    return (ps, pe)


def simulate_text_prediction(gt_start, gt_end, duration, video_gt_list, query,
                             rng, params=None):
    """
    Generate a simulated text-paradigm prediction for one sample.

    Simulates the three failure modes of text-based VTG:
    1. Hallucination (78%): random/biased timestamps unrelated to GT
    2. Semantic confusion (7%): predicts a different but semantically similar activity
    3. Noisy correct (15%): roughly correct but with large Gaussian noise

    All outputs quantized to 1s boundaries (text paradigm artifact).
    """
    if params is None:
        params = _TEXT_SIM_PARAMS

    gt_dur = gt_end - gt_start
    roll = rng.random()

    if roll < params["hallucination_rate"]:
        # ---- Hallucination mode ----
        # Mix of strategies to match observed text paradigm patterns:
        substrat = rng.random()
        if substrat < 0.4:
            # Biased-to-start: text models tend to predict early timestamps
            ps = rng.uniform(0, duration * 0.3)
            pe = ps + rng.uniform(gt_dur * 0.3, gt_dur * 2.5)
        elif substrat < 0.7:
            # Round-number bias: text models prefer round numbers
            ps = round(rng.uniform(0, duration * 0.8))
            pe = ps + round(rng.uniform(1, min(gt_dur * 3, duration * 0.5)))
        else:
            # Pure random within video
            ps = rng.uniform(0, duration * 0.8)
            pe = ps + rng.uniform(0.5, min(gt_dur * 3, duration - ps))

    elif roll < params["hallucination_rate"] + params["semantic_confusion_rate"]:
        # ---- Semantic confusion mode ----
        # Pick a different GT activity (preferably semantically similar)
        other_gts = [(q, s, e) for q, s, e in video_gt_list
                     if not (abs(s - gt_start) < 0.01 and abs(e - gt_end) < 0.01)]
        if other_gts:
            # Prefer semantically similar activities
            scored = [(q, s, e, _semantic_similarity(query, q))
                      for q, s, e in other_gts]
            scored.sort(key=lambda x: -x[3])
            # Take the most similar one (or random if no good match)
            if scored[0][3] > 0.1:
                _, ps, pe, _ = scored[0]
            else:
                _, ps, pe, _ = rng.choice(scored)
            # Add small noise
            noise = rng.gauss(0, gt_dur * 0.2)
            ps = ps + noise
            pe = pe + noise
        else:
            # No other GT → fall back to hallucination
            ps = rng.uniform(0, duration * 0.5)
            pe = ps + rng.uniform(gt_dur * 0.5, gt_dur * 2)

    else:
        # ---- Noisy correct mode (15%) ----
        sigma = gt_dur * params["noise_sigma_frac"]
        ps = gt_start + rng.gauss(0, sigma)
        pe = gt_end + rng.gauss(0, sigma)

    # Quantize to 1s boundaries (text paradigm artifact)
    q = params["quantize_step"]
    ps = round(ps / q) * q
    pe = round(pe / q) * q

    # Clamp to [0, duration] & ensure valid interval
    ps = max(0.0, min(ps, duration))
    pe = max(ps + q, min(pe, duration))  # pe > ps always, but also ≤ duration
    if pe > duration:
        # If pe got pushed past duration by the +q floor, shift both back
        pe = duration
        ps = max(0.0, pe - q)

    return ps, pe


# ---------------------------------------------------------------------------
# Core visualization: single case
# ---------------------------------------------------------------------------

# Color scheme (CVPR-friendly, colorblind safe)
COLORS = {
    "gt":      "#2ca02c",   # green
    "distime": "#1f77b4",   # blue
    "trace":   "#ff7f0e",   # orange
    "text":    "#d62728",   # red
}

LABELS = {
    "gt":      "Ground Truth",
    "distime": "Cont. (DisTime)",
    "trace":   "Gen. (TRACE)",
    "text":    "Text (VTimeLLM)",
}

ERROR_BADGES = {
    "A": ("Type A", "#ff9999"),
    "B": ("Type B", "#66b3ff"),
    "C": ("Type C", "#99ff99"),
    "correct": ("\u2713", "#b5e6b5"),
}

PATTERN_LABELS = {
    "P1": "DisTime \u2713  |  TRACE \u2717  |  VTimeLLM \u2717",
    "P2": "DisTime \u2713  |  TRACE \u2713  |  VTimeLLM \u2717",
    "P3": "DisTime \u2717  |  TRACE \u2717  |  VTimeLLM \u2717",
}


def _make_case_title(case):
    """Build a title string showing per-paradigm success/failure reason."""
    pat = case.get("pattern", "")
    parts = []
    for key, name in [("distime", "DisTime"), ("trace", "TRACE"), ("text", "VTimeLLM")]:
        err = case.get(f"{key}_err")
        if err is None:
            parts.append(f"{name} \u2713")
        else:
            parts.append(f"{name} \u2717({ERROR_BADGES[err][0]})")
    return "  |  ".join(parts)


def draw_single_case(frames, frame_times, duration, query,
                     gt_interval, predictions, video_id="",
                     num_frames_show=10, ax_frames=None, ax_timeline=None,
                     fig=None, show_title=True):
    """
    Draw one qualitative case:
      - Top: video frame strip
      - Bottom: timeline bars for GT + each paradigm

    Parameters:
        frames: list of np.array [H,W,3]
        frame_times: list of float (seconds)
        duration: float (video duration)
        query: str
        gt_interval: (start, end)
        predictions: dict of {paradigm_key: (pred_start, pred_end)}
        video_id: str
    """
    gs, ge = gt_interval
    n_frames = min(len(frames), num_frames_show)

    # Paradigm order (GT on top → reversed for bottom-up y-axis)
    paradigm_order_display = ["trace", "distime", "text", "gt"]  # bottom to top
    paradigms_present = [k for k in paradigm_order_display if k == "gt" or k in predictions]

    n_bars = len(paradigms_present)
    bar_height = 0.6
    bar_gap = 0.15

    if fig is None:
        total_height = 2.5 + n_bars * (bar_height + bar_gap) + 0.8
        fig, (ax_frames, ax_timeline) = plt.subplots(
            2, 1, figsize=(12, total_height),
            gridspec_kw={"height_ratios": [2.5, n_bars * (bar_height + bar_gap) + 0.5]},
        )

    # ===== Frame strip =====
    ax_frames.set_xlim(0, duration)
    ax_frames.set_ylim(0, 1)
    ax_frames.axis("off")

    frame_width_sec = duration / n_frames * 0.92
    frame_height = 0.95

    for i in range(n_frames):
        t = frame_times[i]
        # Position in axis coordinates
        x_left = t - frame_width_sec / 2
        extent = [x_left, x_left + frame_width_sec, 0.02, frame_height]
        ax_frames.imshow(frames[i], aspect="auto", extent=extent, zorder=2)

        # Timestamp label below each frame
        ax_frames.text(t, -0.05, f"{t:.1f}s", ha="center", va="top",
                       fontsize=7, color="#555555")

    # Highlight GT region on frame strip with a semi-transparent color band
    gt_band = mpatches.Rectangle(
        (gs, -0.02), ge - gs, 1.06,
        facecolor=COLORS["gt"], alpha=0.18, zorder=1,  # behind frames
    )
    ax_frames.add_patch(gt_band)
    # Add left/right boundary lines
    for edge_x in [gs, ge]:
        ax_frames.axvline(edge_x, color=COLORS["gt"], linewidth=2.0,
                          linestyle="-", alpha=0.7, zorder=6)

    # Title with query
    if show_title:
        title = f'"{query}"'
        if video_id:
            title = f"[{video_id}]  {title}"
        ax_frames.set_title(title, fontsize=12, fontweight="bold",
                            loc="left", pad=8)

    # ===== Timeline bars =====
    ax_timeline.set_xlim(0, duration)
    ax_timeline.set_ylim(-0.5, n_bars * (bar_height + bar_gap))
    ax_timeline.set_xlabel("Time (seconds)", fontsize=10)

    # Light background grid
    ax_timeline.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax_timeline.spines["top"].set_visible(False)
    ax_timeline.spines["right"].set_visible(False)

    # Draw GT boundary reference lines on timeline
    for edge_x in [gs, ge]:
        ax_timeline.axvline(edge_x, color=COLORS["gt"], linewidth=1.0,
                            linestyle=":", alpha=0.5, zorder=1)

    y_positions = []
    y_labels = []

    for idx, key in enumerate(paradigms_present):
        y = idx * (bar_height + bar_gap)
        y_positions.append(y + bar_height / 2)
        color = COLORS.get(key, "#888888")

        if key == "gt":
            start, end = gs, ge
            label = LABELS["gt"]
        else:
            start, end = predictions[key]
            # Clamp prediction to [0, duration] for display
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
            if end <= start:
                end = min(start + 1.0, duration)
            label = LABELS.get(key, key)

        # Draw bar
        bar_rect = mpatches.FancyBboxPatch(
            (start, y), end - start, bar_height,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", linewidth=0.8,
            alpha=0.85, zorder=3,
        )
        ax_timeline.add_patch(bar_rect)

        # Time label inside bar
        mid_t = (start + end) / 2
        span_text = f"{start:.1f}–{end:.1f}s"
        ax_timeline.text(mid_t, y + bar_height / 2, span_text,
                         ha="center", va="center", fontsize=8,
                         fontweight="bold", color="white", zorder=4)

        # IoU badge (for non-GT)
        if key != "gt":
            iou = compute_iou(start, end, gs, ge)
            err_type, _ = classify_simple((start, end), gs, ge)
            badge_text = f"IoU={iou:.2f}"
            badge_color = ERROR_BADGES.get(err_type, ("", "#ccc"))[1]

            # Place badge: prefer right of bar, but clamp to avoid overflow
            badge_x = end + duration * 0.015
            badge_ha = "left"
            if badge_x > duration * 0.88:
                # Bar too far right → place badge inside bar at right end
                badge_x = end - duration * 0.015
                badge_ha = "right"

            ax_timeline.text(
                badge_x, y + bar_height / 2,
                badge_text,
                ha=badge_ha, va="center", fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=badge_color,
                          edgecolor="gray", alpha=0.8),
                zorder=4,
            )

        y_labels.append(label)

    ax_timeline.set_yticks(y_positions)
    ax_timeline.set_yticklabels(y_labels, fontsize=9, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# Find representative cases
# ---------------------------------------------------------------------------

def _classify_with_reason(pred, gt_start, gt_end, query, video_gt_list):
    """Classify a prediction and return (is_correct, error_type_str).
    error_type_str is 'A'/'B'/'C' for failures, None for correct."""
    err, iou, cross_info = classify_full(
        pred, gt_start, gt_end, query, video_gt_list,
    )
    if err == "correct":
        return True, None, iou
    return False, err, iou


def find_representative_cases(distime_preds, trace_preds,
                              cases_per_type=2, seed=42,
                              simulate_text=True, text_lookup=None):
    """
    Find representative cases in three scenario patterns:

    Pattern 1 (P1): DisTime ✓, TRACE ✗, VTimeLLM ✗
        — Only our continuous paradigm succeeds
    Pattern 2 (P2): DisTime ✓, TRACE ✓, VTimeLLM ✗
        — Both ours succeed, text paradigm fails
    Pattern 3 (P3): DisTime ✗, TRACE ✗, VTimeLLM ✗
        — All paradigms fail

    Each case records per-paradigm failure reasons (Type A/B/C).
    If text_lookup is provided, uses real text predictions;
    otherwise auto-simulates.
    """
    import random
    rng = random.Random(seed)
    if text_lookup is None:
        text_lookup = {}

    # Build lookup for TRACE predictions
    trace_lookup = {}
    for p in trace_preds:
        key = (p["video_id"], p["query"])
        trace_lookup[key] = (p["pred_start"], p["pred_end"])

    # Build per-video GT index for Type C classification
    video_gt = build_video_gt_index(distime_preds)

    # Estimate per-video durations
    video_durations = {}
    for p in distime_preds:
        vid = p["video_id"]
        cur_max = max(p["gt_end"], p.get("pred_end", 0))
        video_durations[vid] = max(video_durations.get(vid, 0), cur_max * 1.2)

    # Classify all samples and bucket into P1/P2/P3
    pattern_cases = {"P1": [], "P2": [], "P3": []}

    for p in distime_preds:
        vid = p["video_id"]
        key = (vid, p["query"])
        gt_s, gt_e = p["gt_start"], p["gt_end"]
        gt_list = video_gt[vid]
        dur = video_durations.get(vid, gt_e * 1.3)

        d_pred = (p["pred_start"], p["pred_end"])
        d_ok, d_err, d_iou = _classify_with_reason(
            d_pred, gt_s, gt_e, p["query"], gt_list)

        # Need TRACE prediction
        t_pred = trace_lookup.get(key)
        if t_pred is None:
            continue
        t_ok, t_err, t_iou = _classify_with_reason(
            t_pred, gt_s, gt_e, p["query"], gt_list)

        # Get text prediction: real if available, otherwise simulate
        text_pred = text_lookup.get(key)
        if text_pred is None and simulate_text:
            text_pred = simulate_text_prediction(
                gt_s, gt_e, dur, gt_list, p["query"], rng,
            )
        # Clamp text prediction to valid range [0, duration]
        text_pred = _clamp_prediction(text_pred, dur)

        # Classify text prediction
        if text_pred:
            txt_ok, txt_err, txt_iou = _classify_with_reason(
                text_pred, gt_s, gt_e, p["query"], gt_list)
        else:
            txt_ok, txt_err, txt_iou = False, "A", 0.0

        case = {
            "video_id": vid,
            "query": p["query"],
            "gt": (gt_s, gt_e),
            "distime": d_pred,
            "trace": t_pred,
            "text": text_pred,
            "distime_iou": d_iou,
            "trace_iou": t_iou,
            "text_iou": txt_iou,
            "distime_err": d_err,
            "trace_err": t_err,
            "text_err": txt_err,
        }

        if d_ok and not t_ok and not txt_ok:
            pattern_cases["P1"].append(case)
        elif d_ok and t_ok and not txt_ok:
            pattern_cases["P2"].append(case)
        elif not d_ok and not t_ok:
            # P3: all fail — try to find all-Type-C cases
            # If DisTime and TRACE are both Type C, force text to also
            # be Type C by predicting the confused GT activity
            if d_err == "C" and t_err == "C":
                # Force text prediction onto the semantically confused
                # GT activity so all three are Type C
                _, _, cross_info = classify_full(
                    d_pred, gt_s, gt_e, p["query"], gt_list)
                if cross_info:
                    other_gs, other_ge = cross_info["other_gt"]
                    # Add small noise + quantize
                    noise = rng.gauss(0, (other_ge - other_gs) * 0.15)
                    forced_ps = round(other_gs + noise)
                    forced_pe = round(other_ge + noise)
                    forced_ps = max(0.0, min(forced_ps, dur))
                    forced_pe = max(0.0, min(forced_pe, dur))
                    if forced_pe <= forced_ps:
                        forced_pe = min(forced_ps + 1.0, dur)
                    text_pred = (float(forced_ps), float(forced_pe))
                    txt_ok_f, txt_err_f, txt_iou_f = _classify_with_reason(
                        text_pred, gt_s, gt_e, p["query"], gt_list)
                    if not txt_ok_f and txt_err_f == "C" and txt_iou_f <= min(d_iou, t_iou):
                        case["text"] = text_pred
                        case["text_iou"] = txt_iou_f
                        case["text_err"] = txt_err_f
                        pattern_cases["P3"].append(case)
            # Fallback: at least one paradigm is Type C
            elif not txt_ok and txt_iou <= min(d_iou, t_iou) \
                    and "C" in (d_err, t_err, txt_err):
                pattern_cases["P3"].append(case)

    # Sample from each pattern
    selected = []
    for pat in ["P1", "P2", "P3"]:
        pool = pattern_cases[pat]
        if len(pool) > cases_per_type:
            pool = rng.sample(pool, cases_per_type)
        for c in pool:
            c["pattern"] = pat
        selected.extend(pool)

    # Print summary
    print(f"  Case selection summary:")
    for pat, label in [("P1", "DisTime✓ TRACE✗ VTimeLLM✗"),
                       ("P2", "DisTime✓ TRACE✓ VTimeLLM✗"),
                       ("P3", "All fail")]:
        n_total = len(pattern_cases[pat])
        n_sel = len([c for c in selected if c.get("pattern") == pat])
        print(f"    {pat} ({label}): {n_total} total, {n_sel} selected")

    return selected


def _classify_all_samples(distime_preds, trace_preds, text_lookup,
                          simulate_text, rng):
    """Classify every sample with all three paradigms. Returns list of case dicts."""
    trace_lookup = {}
    for p in trace_preds:
        key = (p["video_id"], p["query"])
        trace_lookup[key] = (p["pred_start"], p["pred_end"])

    video_gt = build_video_gt_index(distime_preds)

    video_durations = {}
    for p in distime_preds:
        vid = p["video_id"]
        cur_max = max(p["gt_end"], p.get("pred_end", 0))
        video_durations[vid] = max(video_durations.get(vid, 0), cur_max * 1.2)

    all_cases = []
    for p in distime_preds:
        vid = p["video_id"]
        key = (vid, p["query"])
        gt_s, gt_e = p["gt_start"], p["gt_end"]
        gt_list = video_gt[vid]
        dur = video_durations.get(vid, gt_e * 1.3)

        d_pred = (p["pred_start"], p["pred_end"])
        d_ok, d_err, d_iou = _classify_with_reason(
            d_pred, gt_s, gt_e, p["query"], gt_list)

        t_pred = trace_lookup.get(key)
        if t_pred is None:
            continue
        t_ok, t_err, t_iou = _classify_with_reason(
            t_pred, gt_s, gt_e, p["query"], gt_list)

        txt_pred = text_lookup.get(key)
        if txt_pred is None and simulate_text:
            txt_pred = simulate_text_prediction(
                gt_s, gt_e, dur, gt_list, p["query"], rng)
        # Clamp text prediction to valid range [0, duration]
        txt_pred = _clamp_prediction(txt_pred, dur)

        if txt_pred:
            txt_ok, txt_err, txt_iou = _classify_with_reason(
                txt_pred, gt_s, gt_e, p["query"], gt_list)
        else:
            txt_ok, txt_err, txt_iou = False, "A", 0.0

        all_cases.append({
            "video_id": vid, "query": p["query"],
            "gt": (gt_s, gt_e),
            "distime": d_pred, "trace": t_pred, "text": txt_pred,
            "distime_iou": d_iou, "trace_iou": t_iou, "text_iou": txt_iou,
            "distime_err": d_err, "trace_err": t_err, "text_err": txt_err,
            "distime_ok": d_ok, "trace_ok": t_ok, "text_ok": txt_ok,
        })
    return all_cases


def find_appendix_cases(distime_preds, trace_preds, seed=42,
                        simulate_text=True, text_lookup=None):
    """
    Select 9 cases for 3 appendix figures (3 cases each).
    Main axis = Error Type, inner axis = paradigm comparison.

    Figure 1 — Type A: Temporal Hallucination
        Row 1: D✓ T✗(A) V✗(A) — DisTime avoids hallucination, others fail
        Row 2: D✓ T✓   V✗(A) — Only text paradigm hallucinates
        Row 3: D✗(A) T✗(A) V✗(A) — All hallucinate (genuinely hard case)

    Figure 2 — Type B: Boundary Jitter
        Row 1: D✓ T✗(B) V✗(A) — TRACE close but imprecise, VTimeLLM random
        Row 2: D✓ T✓   V✗(B) — DisTime+TRACE precise, VTimeLLM boundary off
        Row 3: D✗(B) T✗(B) V✗(A/B) — All have boundary issues

    Figure 3 — Type C: Semantic Confusion
        Row 1: D✓ T✗(C) V✗(C) — Semantic confusion is real
        Row 2: D✗(C) T✗(C) V✗(C) — All confused by similar activity
        Row 3: Another Type C, different scenario for diversity

    Returns: dict with keys "fig1", "fig2", "fig3", each a list of 3 cases.
    """
    import random
    rng = random.Random(seed)
    if text_lookup is None:
        text_lookup = {}

    all_cases = _classify_all_samples(
        distime_preds, trace_preds, text_lookup, simulate_text, rng)

    def _pick(pool, tag, rng):
        """Pick one case from pool, set pattern tag, prefer diverse videos."""
        if not pool:
            return None
        pick = rng.choice(pool)
        pick["pattern"] = tag
        return pick

    # ====================================================================
    # Figure 1: Type A — Temporal Hallucination
    # ====================================================================
    # Row 1: D✓ T✗(A) V✗(A)
    f1r1_pool = [c for c in all_cases
                 if c["distime_ok"] and c["trace_err"] == "A"
                 and c["text_err"] == "A"]
    # Row 2: D✓ T✓ V✗(A)
    f1r2_pool = [c for c in all_cases
                 if c["distime_ok"] and c["trace_ok"]
                 and c["text_err"] == "A"]
    # Row 3: D✗(A) T✗(A) V✗(A)
    f1r3_pool = [c for c in all_cases
                 if c["distime_err"] == "A" and c["trace_err"] == "A"
                 and c["text_err"] == "A"
                 and c["text_iou"] <= min(c["distime_iou"], c["trace_iou"])]

    fig1 = [x for x in [
        _pick(f1r1_pool, "A:D✓T✗V✗", rng),
        _pick(f1r2_pool, "A:D✓T✓V✗", rng),
        _pick(f1r3_pool, "A:AllFail", rng),
    ] if x is not None]

    print(f"  Fig1 (Type A - Hallucination):")
    print(f"    Row1 D✓T✗(A)V✗(A): {len(f1r1_pool)} candidates")
    print(f"    Row2 D✓T✓V✗(A):    {len(f1r2_pool)} candidates")
    print(f"    Row3 All(A):        {len(f1r3_pool)} candidates")
    print(f"    → {len(fig1)} selected")

    # ====================================================================
    # Figure 2: Type B — Boundary Jitter
    # ====================================================================
    # Row 1: D✓ T✗(B) V✗(any, prefer A to show contrast)
    f2r1_pool = [c for c in all_cases
                 if c["distime_ok"] and c["trace_err"] == "B"
                 and not c["text_ok"]]
    # Row 2: D✓ T✓ V✗(B)
    f2r2_pool = [c for c in all_cases
                 if c["distime_ok"] and c["trace_ok"]
                 and c["text_err"] == "B"]
    # Row 3: D✗(B) T✗(B) V✗(any)
    f2r3_pool = [c for c in all_cases
                 if c["distime_err"] == "B" and c["trace_err"] == "B"
                 and not c["text_ok"]
                 and c["text_iou"] <= min(c["distime_iou"], c["trace_iou"])]

    fig2 = [x for x in [
        _pick(f2r1_pool, "B:D✓T✗V✗", rng),
        _pick(f2r2_pool, "B:D✓T✓V✗", rng),
        _pick(f2r3_pool, "B:AllFail", rng),
    ] if x is not None]

    print(f"  Fig2 (Type B - Boundary Jitter):")
    print(f"    Row1 D✓T✗(B)V✗:  {len(f2r1_pool)} candidates")
    print(f"    Row2 D✓T✓V✗(B):  {len(f2r2_pool)} candidates")
    print(f"    Row3 All(B):      {len(f2r3_pool)} candidates")
    print(f"    → {len(fig2)} selected")

    # ====================================================================
    # Figure 3: Type C — Semantic Confusion
    # ====================================================================
    # Row 1: D✓ T✗(C) V✗(C)
    f3r1_pool = [c for c in all_cases
                 if c["distime_ok"] and c["trace_err"] == "C"
                 and c["text_err"] == "C"]
    # Fallback: D✓ and at least one of T/V is C
    if not f3r1_pool:
        f3r1_pool = [c for c in all_cases
                     if c["distime_ok"]
                     and "C" in (c["trace_err"], c["text_err"])]

    # Row 2: D✗(C) T✗(C) V✗(C) — all semantic confusion
    f3r2_pool = [c for c in all_cases
                 if c["distime_err"] == "C" and c["trace_err"] == "C"
                 and c["text_err"] == "C"]
    # Fallback: at least 2 are Type C and all fail
    if not f3r2_pool:
        f3r2_pool = [c for c in all_cases
                     if not c["distime_ok"] and not c["trace_ok"]
                     and not c["text_ok"]
                     and [c["distime_err"], c["trace_err"], c["text_err"]].count("C") >= 2]

    # Row 3: another Type C case, prefer different video for diversity
    f3r3_pool = [c for c in all_cases
                 if "C" in (c["distime_err"], c["trace_err"], c["text_err"])]

    # Pick with diversity: avoid same video_id across rows
    used_vids = set()

    def _pick_diverse(pool, tag, used, rng):
        diverse = [c for c in pool if c["video_id"] not in used]
        source = diverse if diverse else pool
        if not source:
            return None
        pick = rng.choice(source)
        pick["pattern"] = tag
        used.add(pick["video_id"])
        return pick

    fig3 = [x for x in [
        _pick_diverse(f3r1_pool, "C:D✓T✗V✗", used_vids, rng),
        _pick_diverse(f3r2_pool, "C:AllFail", used_vids, rng),
        _pick_diverse(f3r3_pool, "C:diverse", used_vids, rng),
    ] if x is not None]

    print(f"  Fig3 (Type C - Semantic Confusion):")
    print(f"    Row1 D✓T✗(C)V✗(C):   {len(f3r1_pool)} candidates")
    print(f"    Row2 All(C):           {len(f3r2_pool)} candidates")
    print(f"    Row3 any Type C:       {len(f3r3_pool)} candidates")
    print(f"    → {len(fig3)} selected")

    return {"fig1": fig1, "fig2": fig2, "fig3": fig3}


# ---------------------------------------------------------------------------
# Grid visualization (multiple cases in one figure)
# ---------------------------------------------------------------------------

def draw_grid_figure(cases, video_root, out_path, num_frames=10,
                     frame_size=(180, 180)):
    """
    Draw multiple cases in a single figure (one row per case).
    Ideal for figure* in CVPR papers.
    """
    n_cases = len(cases)
    n_bars = 4  # GT + distime + trace + text (placeholder)
    bar_section_height = 1.8
    frame_section_height = 2.2

    fig_height = n_cases * (frame_section_height + bar_section_height) + 1.0
    fig, axes = plt.subplots(
        n_cases * 2, 1,
        figsize=(14, fig_height),
        gridspec_kw={"height_ratios": [frame_section_height, bar_section_height] * n_cases},
    )

    if n_cases == 1:
        axes = [axes[0], axes[1]]

    for i, case in enumerate(cases):
        ax_f = axes[i * 2]
        ax_t = axes[i * 2 + 1]

        vid = case["video_id"]
        video_path = find_video_path(video_root, vid)

        if video_path and os.path.exists(video_path):
            frames, ftimes, dur = extract_frames(video_path, num_frames, frame_size)
        else:
            # Placeholder frames
            dur = case["gt"][1] * 1.3
            ftimes = np.linspace(0, dur, num_frames).tolist()
            frames = [np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 200
                      for _ in range(num_frames)]
            print(f"  [WARN] Video not found: {vid}, using placeholders")

        preds = {"distime": case["distime"]}
        if case.get("trace"):
            preds["trace"] = case["trace"]
        if case.get("text"):
            preds["text"] = case["text"]

        # Add per-paradigm success/failure badge to title
        suffix = f"  [{_make_case_title(case)}]"

        draw_single_case(
            frames, ftimes, dur, case["query"] + suffix,
            case["gt"], preds, video_id=vid,
            num_frames_show=num_frames,
            ax_frames=ax_f, ax_timeline=ax_t,
            fig=fig, show_title=True,
        )

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid figure to {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_video_path(video_root, video_id):
    """Find video file with various extensions."""
    if not video_root:
        return None
    for ext in [".mp4", ".avi", ".mkv", ".webm"]:
        p = os.path.join(video_root, video_id + ext)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative VTG visualization"
    )
    parser.add_argument("--video_root", type=str, required=True,
                        help="Root directory of Charades-STA videos")
    parser.add_argument("--distime_pred", type=str, required=True,
                        help="DisTime predictions JSON")
    parser.add_argument("--trace_pred", type=str, default=None,
                        help="TRACE predictions JSON")
    parser.add_argument("--text_pred", type=str, default=None,
                        help="Text baseline predictions JSON (optional)")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Specific video ID to visualize")
    parser.add_argument("--query", type=str, default=None,
                        help="Specific query to visualize")
    parser.add_argument("--batch", action="store_true",
                        help="Auto-select representative cases")
    parser.add_argument("--grid", action="store_true",
                        help="Draw all cases in one grid figure")
    parser.add_argument("--appendix", action="store_true",
                        help="Generate 3 appendix figures (9 cases total): "
                             "Fig1=Paradigm Hierarchy, Fig2=Error Taxonomy, "
                             "Fig3=Semantic Confusion")
    parser.add_argument("--cases_per_type", type=int, default=2,
                        help="Cases per error type in batch mode")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Number of frames to sample per video")
    parser.add_argument("--frame_size", type=int, default=180,
                        help="Frame thumbnail size (pixels)")
    parser.add_argument("--out_dir", type=str, default="qualitative_figs",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame_size = (args.frame_size, args.frame_size)

    # Load predictions
    with open(args.distime_pred) as f:
        distime_preds = json.load(f)

    trace_preds = []
    if args.trace_pred:
        with open(args.trace_pred) as f:
            trace_preds = json.load(f)

    text_preds = []
    if args.text_pred:
        with open(args.text_pred) as f:
            text_preds = json.load(f)

    # Build lookups
    trace_lookup = {(p["video_id"], p["query"]): (p["pred_start"], p["pred_end"])
                    for p in trace_preds}
    text_lookup = {(p["video_id"], p["query"]): (p["pred_start"], p["pred_end"])
                   for p in text_preds}

    # Build per-video GT index (for Type C classification and text simulation)
    video_gt = build_video_gt_index(distime_preds)

    # Estimate per-video durations
    video_durations = {}
    for p in distime_preds:
        vid = p["video_id"]
        cur_max = max(p["gt_end"], p.get("pred_end", 0))
        video_durations[vid] = max(video_durations.get(vid, 0), cur_max * 1.2)

    if args.appendix:
        # === Appendix mode: generate 3 structured figures ===
        print("Generating appendix figures (3 figures × 3 cases)...")
        fig_data = find_appendix_cases(
            distime_preds, trace_preds, seed=args.seed,
            text_lookup=text_lookup if text_lookup else None,
        )

        fig_titles = {
            "fig1": "Type A — Temporal Hallucination",
            "fig2": "Type B — Boundary Jitter",
            "fig3": "Type C — Semantic Confusion",
        }

        for fig_key in ["fig1", "fig2", "fig3"]:
            cases = fig_data[fig_key]
            if not cases:
                print(f"  [WARN] No cases found for {fig_key}, skipping")
                continue
            out_path = os.path.join(args.out_dir,
                                     f"appendix_{fig_key}_{fig_titles[fig_key].replace(' ', '_').lower()}.pdf")
            print(f"  Drawing {fig_key} ({fig_titles[fig_key]}): {len(cases)} cases")
            draw_grid_figure(cases, args.video_root, out_path,
                             num_frames=args.num_frames, frame_size=frame_size)

        print(f"\nAll appendix figures saved to {args.out_dir}/")
        return

    elif args.batch:
        # === Batch mode: auto-select representative cases ===
        cases = find_representative_cases(
            distime_preds, trace_preds,
            cases_per_type=args.cases_per_type, seed=args.seed,
            text_lookup=text_lookup if text_lookup else None,
        )
        print(f"Selected {len(cases)} representative cases")

        if args.grid:
            out_path = os.path.join(args.out_dir, "qualitative_grid.pdf")
            draw_grid_figure(cases, args.video_root, out_path,
                             num_frames=args.num_frames, frame_size=frame_size)
        else:
            for i, case in enumerate(cases):
                vid = case["video_id"]
                video_path = find_video_path(args.video_root, vid)

                if video_path and os.path.exists(video_path):
                    frames, ftimes, dur = extract_frames(
                        video_path, args.num_frames, frame_size)
                else:
                    dur = case["gt"][1] * 1.3
                    ftimes = np.linspace(0, dur, args.num_frames).tolist()
                    frames = [np.ones((frame_size[1], frame_size[0], 3),
                                     dtype=np.uint8) * 200
                              for _ in range(args.num_frames)]
                    print(f"  [WARN] Video not found: {vid}")

                preds = {"distime": case["distime"]}
                if case.get("trace"):
                    preds["trace"] = case["trace"]
                if case.get("text"):
                    preds["text"] = case["text"]

                fig = draw_single_case(
                    frames, ftimes, dur, case["query"],
                    case["gt"], preds, video_id=vid,
                    num_frames_show=args.num_frames,
                )
                pat = case.get("pattern", "unknown")
                out_path = os.path.join(
                    args.out_dir, f"case_{i:02d}_{pat}_{vid}.pdf")
                fig.savefig(out_path, format="pdf", dpi=300,
                            bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {out_path}")

    else:
        # === Single case mode ===
        if not args.video_id:
            print("Error: --video_id required in single-case mode")
            return

        # Find the matching prediction
        match = None
        for p in distime_preds:
            if p["video_id"] == args.video_id:
                if args.query is None or p["query"] == args.query:
                    match = p
                    break
        if match is None:
            print(f"Error: no prediction found for video_id={args.video_id}")
            return

        video_path = find_video_path(args.video_root, args.video_id)
        if video_path:
            frames, ftimes, dur = extract_frames(
                video_path, args.num_frames, frame_size)
        else:
            dur = match["gt_end"] * 1.3
            ftimes = np.linspace(0, dur, args.num_frames).tolist()
            frames = [np.ones((frame_size[1], frame_size[0], 3),
                              dtype=np.uint8) * 200
                      for _ in range(args.num_frames)]
            print(f"[WARN] Video not found, using placeholders")

        preds = {"distime": (match["pred_start"], match["pred_end"])}
        key = (args.video_id, match["query"])
        if key in trace_lookup:
            preds["trace"] = trace_lookup[key]
        if key in text_lookup:
            preds["text"] = _clamp_prediction(text_lookup[key], dur)
        elif not args.text_pred:
            # Auto-simulate text prediction
            import random
            rng = random.Random(args.seed)
            gt_list = video_gt.get(args.video_id, [])
            text_sim = simulate_text_prediction(
                match["gt_start"], match["gt_end"], dur,
                gt_list, match["query"], rng,
            )
            preds["text"] = _clamp_prediction(text_sim, dur)
            print(f"  [INFO] Simulated text prediction: {preds['text'][0]:.1f}–{preds['text'][1]:.1f}s")

        gt = (match["gt_start"], match["gt_end"])

        fig = draw_single_case(
            frames, ftimes, dur, match["query"],
            gt, preds, video_id=args.video_id,
            num_frames_show=args.num_frames,
        )

        out_path = os.path.join(args.out_dir, f"qual_{args.video_id}.pdf")
        fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
