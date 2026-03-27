import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Callable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.neighbors import KernelDensity

from datasetModel import CurrentOnlyDataset, RISK_COLS
from modelArchitecture import BaselineCurrentOnlyModel


DEFAULT_EXP_ROOT = Path("/local/scratch/tpiltne/models/baselineModel")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 123
CI_ALPHA = 0.05
MIN_VALID_BOOT = 50

KDE_BW = 0.03

COLOR_OVERALL = "tab:blue"
COLOR_ACCEPTED = "tab:green"
COLOR_REJECTED = "tab:orange"
COLOR_CORRECT = "tab:green"
COLOR_INCORRECT = "tab:red"

ACCEPTED_CLIN_NAME = "accepted+clinician"
REJECTED_CLIN_NAME = "rejected+clinician"

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    eps = 1e-8
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def safe_auc_auprc(y: np.ndarray, s: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    y = y.astype(int)
    if (y == 1).sum() > 0 and (y == 0).sum() > 0:
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    return None, None


# bootstrap
def bootstrap_metric_samples(
    y: np.ndarray,
    s: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.default_rng(seed)
    samples: List[float] = []

    if len(y) < 2:
        return np.asarray(samples, dtype=float)

    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            val = metric_fn(y[idx], s[idx])
            if val is not None:
                samples.append(float(val))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        if len(uniq) < 2:
            return np.asarray(samples, dtype=float)

        group_to_idx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_groups], axis=0)
            val = metric_fn(y[idx], s[idx])
            if val is not None:
                samples.append(float(val))

    return np.asarray(samples, dtype=float)


def ci_from_samples(samples: np.ndarray, alpha: float = CI_ALPHA) -> Tuple[Optional[float], Optional[float], int]:
    n_valid = int(len(samples))
    if n_valid < MIN_VALID_BOOT:
        return None, None, n_valid
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi, n_valid


def metric_summary(
    y: np.ndarray,
    s: np.ndarray,
    label: str,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    auc_point, ap_point = safe_auc_auprc(y, s)

    if auc_point is not None:
        auc_samples = bootstrap_metric_samples(
            y, s, lambda yy, ss: safe_auc_auprc(yy.astype(int), ss)[0], groups=groups
        )
        auc_lo, auc_hi, auc_n = ci_from_samples(auc_samples)
        auc_ci = {"low": auc_lo, "high": auc_hi, "n_valid_boot": auc_n}
    else:
        auc_ci = None
        auc_samples = np.array([], dtype=float)

    if ap_point is not None:
        ap_samples = bootstrap_metric_samples(
            y, s, lambda yy, ss: safe_auc_auprc(yy.astype(int), ss)[1], groups=groups
        )
        ap_lo, ap_hi, ap_n = ci_from_samples(ap_samples)
        ap_ci = {"low": ap_lo, "high": ap_hi, "n_valid_boot": ap_n}
    else:
        ap_ci = None
        ap_samples = np.array([], dtype=float)

    return {
        "label": label,
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auc": auc_point,
        "auc_ci95": auc_ci,
        "auc_samples": auc_samples,
        "auprc": ap_point,
        "auprc_ci95": ap_ci,
        "auprc_samples": ap_samples,
    }

# checkpoint compatibility
def remap_legacy_cum_keys_for_baseline(state_dict: dict) -> dict:
    """
    Legacy checkpoint key shim:
      cum.base_hazard_fc -> cum.base
      cum.hazard_fc      -> cum.inc
    """
    sd = dict(state_dict)
    has_legacy = any(
        k.startswith("cum.hazard_fc.") or k.startswith("cum.base_hazard_fc.")
        for k in sd.keys()
    )
    if not has_legacy:
        return sd

    if "cum.base_hazard_fc.weight" in sd:
        sd["cum.base.weight"] = sd.pop("cum.base_hazard_fc.weight")
    if "cum.base_hazard_fc.bias" in sd:
        sd["cum.base.bias"] = sd.pop("cum.base_hazard_fc.bias")
    if "cum.hazard_fc.weight" in sd:
        sd["cum.inc.weight"] = sd.pop("cum.hazard_fc.weight")
    if "cum.hazard_fc.bias" in sd:
        sd["cum.inc.bias"] = sd.pop("cum.hazard_fc.bias")

    sd.pop("cum.upper_triangular_mask", None)
    return sd


# prediction helpers
def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


@torch.inference_mode()
def collect_logits_labels_masks(
    model: torch.nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

    model.eval()

    all_logits = []
    all_labels = []
    all_masks = []
    all_groups = []

    for batch in loader:
        if len(batch) < 5:
            raise ValueError("Batch must have at least 5 elements: imgs, delta_feat, has_prior_views, y, m")

        imgs, delta_feat, has_prior_views, y, m = batch[:5]
        group_ids = batch[5] if len(batch) >= 6 else None

        imgs = imgs.to(DEVICE, dtype=torch.float32)
        delta_feat = delta_feat.to(DEVICE, dtype=torch.float32)
        has_prior_views = has_prior_views.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)
        m = m.to(DEVICE, dtype=torch.float32)

        out = model(imgs, delta_feat, has_prior_views)
        logits = out["risk_prediction"]["pred_fused"]

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_masks.append(m.detach().cpu().numpy())

        if group_ids is not None:
            if torch.is_tensor(group_ids):
                group_ids = group_ids.detach().cpu().numpy()
            all_groups.append(np.asarray(group_ids))

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    groups = None
    if len(all_groups) > 0:
        groups = np.concatenate(all_groups, axis=0)
        if groups.shape[0] != logits.shape[0]:
            print("[WARN] group_ids shape mismatch; ignoring group bootstrap.")
            groups = None

    return logits, labels, masks, groups


@torch.no_grad()
def collect_mc_uncertainty(
    model: torch.nn.Module,
    loader: DataLoader,
    mc_samples: int = 30,
    uq_mode: str = "mutual_info",
) -> np.ndarray:
    """
    Returns uncertainty array of shape [N, H].
    Uses raw logits from MC dropout and converts each pass with sigmoid.
    """
    det_logits, _, _, _ = collect_logits_labels_masks(model, loader)
    n, h = det_logits.shape
    probs_mc = np.zeros((mc_samples, n, h), dtype=np.float64)

    for s in range(mc_samples):
        enable_mc_dropout(model)
        pass_logits = []

        for batch in loader:
            imgs, delta_feat, has_prior_views = batch[:3]

            imgs = imgs.to(DEVICE, dtype=torch.float32)
            delta_feat = delta_feat.to(DEVICE, dtype=torch.float32)
            has_prior_views = has_prior_views.to(DEVICE, dtype=torch.float32)

            out = model(imgs, delta_feat, has_prior_views)
            logits = out["risk_prediction"]["pred_fused"]
            pass_logits.append(logits.detach().cpu().numpy())

        logits_s = np.concatenate(pass_logits, axis=0)
        probs_mc[s] = sigmoid_np(logits_s)

    mean_probs = probs_mc.mean(axis=0)
    uq_mode = uq_mode.lower().strip()

    if uq_mode == "variance":
        uq = probs_mc.var(axis=0)
    elif uq_mode == "entropy_mean":
        uq = binary_entropy_bits(mean_probs)
    elif uq_mode in ("mutual_info", "mi"):
        entropy_of_mean = binary_entropy_bits(mean_probs)
        mean_entropy = binary_entropy_bits(probs_mc).mean(axis=0)
        uq = entropy_of_mean - mean_entropy
    else:
        raise ValueError(f"Unknown uq_mode: {uq_mode}")

    return uq.astype(np.float64)


# correctness / referral helpers
def preds_match_prevalence(p: np.ndarray, y: np.ndarray, tie_break_seed: int = 0) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    n = int(p.size)
    k = int((y == 1).sum())

    if n == 0:
        return np.asarray([], dtype=int)
    if k <= 0:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.ones(n, dtype=int)

    rng = np.random.default_rng(int(tie_break_seed))
    jitter = rng.normal(0.0, 1e-12, size=n)
    score = p + jitter

    topk_idx = np.argpartition(score, kth=n - k)[n - k:]
    pred = np.zeros(n, dtype=int)
    pred[topk_idx] = 1
    return pred


def compute_pred_info(p: np.ndarray, y: np.ndarray, tie_break_seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred = preds_match_prevalence(p, y, tie_break_seed=tie_break_seed)
    threshold = float(np.min(p[pred == 1])) if int(pred.sum()) > 0 else None

    info = {
        "mode": "match_prevalence",
        "n_obs": int(len(y)),
        "n_pos_true": int((y == 1).sum()),
        "n_pos_pred": int(pred.sum()),
        "implied_prob_threshold_min_selected": threshold,
        "tie_break_seed": int(tie_break_seed),
    }
    return pred, info


def fit_median_val_gate(val_uq: np.ndarray, val_masks: np.ndarray) -> Dict[str, Any]:
    gate = {}
    for t, h in enumerate(RISK_COLS):
        observed = val_masks[:, t] > 0.5
        u = val_uq[observed, t].astype(float)
        gate[h] = {"tau_cutoff": None if u.size == 0 else float(np.median(u))}
    return gate


def apply_gate_splits(
    probs: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    uq: np.ndarray,
    groups: Optional[np.ndarray],
    gate: Dict[str, Any],
) -> Dict[str, Any]:
    out = {}

    for t, h in enumerate(RISK_COLS):
        observed = masks[:, t] > 0.5
        y = labels[observed, t].astype(int)
        p = probs[observed, t].astype(float)
        u = uq[observed, t].astype(float)
        g = groups[observed] if groups is not None else None

        tau = gate[h]["tau_cutoff"]

        if y.size == 0 or tau is None:
            out[h] = {
                "gate": {"tau_cutoff": tau},
                "overall": {"y": y, "p": p, "g": g},
                ACCEPTED_CLIN_NAME: {"y": y, "p": np.array([], dtype=float), "g": g},
                REJECTED_CLIN_NAME: {"y": y, "p": np.array([], dtype=float), "g": g},
                "accepted": {"y": np.array([], dtype=int), "p": np.array([], dtype=float), "g": None},
                "rejected": {"y": np.array([], dtype=int), "p": np.array([], dtype=float), "g": None},
            }
            continue

        accepted_mask = u <= float(tau)
        rejected_mask = ~accepted_mask

        p_acc_clin = p.copy()
        p_acc_clin[rejected_mask] = y[rejected_mask].astype(float)

        p_rej_clin = p.copy()
        p_rej_clin[accepted_mask] = y[accepted_mask].astype(float)

        out[h] = {
            "gate": {"tau_cutoff": float(tau)},
            "overall": {"y": y, "p": p, "g": g},
            ACCEPTED_CLIN_NAME: {"y": y, "p": p_acc_clin, "g": g},
            REJECTED_CLIN_NAME: {"y": y, "p": p_rej_clin, "g": g},
            "accepted": {
                "y": y[accepted_mask],
                "p": p[accepted_mask],
                "g": g[accepted_mask] if g is not None else None,
            },
            "rejected": {
                "y": y[rejected_mask],
                "p": p[rejected_mask],
                "g": g[rejected_mask] if g is not None else None,
            },
        }

    return out


def build_uq_metrics_and_plot_data(uq_splits: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    uq_metrics = {}
    uq_plot_data = {}

    for h in RISK_COLS:
        d = uq_splits[h]

        overall = metric_summary(d["overall"]["y"], d["overall"]["p"], "overall", groups=d["overall"]["g"])
        acc_clin = metric_summary(
            d[ACCEPTED_CLIN_NAME]["y"],
            d[ACCEPTED_CLIN_NAME]["p"],
            ACCEPTED_CLIN_NAME,
            groups=d[ACCEPTED_CLIN_NAME]["g"],
        )
        rej_clin = metric_summary(
            d[REJECTED_CLIN_NAME]["y"],
            d[REJECTED_CLIN_NAME]["p"],
            REJECTED_CLIN_NAME,
            groups=d[REJECTED_CLIN_NAME]["g"],
        )
        accepted = metric_summary(d["accepted"]["y"], d["accepted"]["p"], "accepted", groups=d["accepted"]["g"])
        rejected = metric_summary(d["rejected"]["y"], d["rejected"]["p"], "rejected", groups=d["rejected"]["g"])

        coverage = (accepted["n"] / overall["n"]) if overall["n"] > 0 else None

        uq_metrics[h] = {
            "gate": d["gate"],
            "coverage_over_observed": coverage,
            "overall": {k: v for k, v in overall.items() if not k.endswith("_samples")},
            ACCEPTED_CLIN_NAME: {k: v for k, v in acc_clin.items() if not k.endswith("_samples")},
            REJECTED_CLIN_NAME: {k: v for k, v in rej_clin.items() if not k.endswith("_samples")},
            "accepted": {k: v for k, v in accepted.items() if not k.endswith("_samples")},
            "rejected": {k: v for k, v in rejected.items() if not k.endswith("_samples")},
        }

        uq_plot_data[h] = {
            "overall": overall,
            ACCEPTED_CLIN_NAME: acc_clin,
            REJECTED_CLIN_NAME: rej_clin,
        }

    return uq_metrics, uq_plot_data

# plotting
def style_axes():
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, alpha=0.25)


def plot_roc_all_horizons(labels, probs, masks, out_path: Path, title: str = "ROC Plot"):
    plt.figure(figsize=(11, 8.5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=2)

    for t, name in enumerate(RISK_COLS):
        observed = masks[:, t] > 0.5
        y = labels[observed, t].astype(int)
        p = probs[observed, t].astype(float)

        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.step(fpr, tpr, where="post", linewidth=2.5, label=f"{name} (AUC={auc:.3f})")

    plt.xlabel("False Positive Rate", fontsize=LABEL_FONTSIZE)
    plt.ylabel("True Positive Rate", fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    style_axes()
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pr_all_horizons(labels, probs, masks, out_path: Path, title: str = "PR Plot"):
    plt.figure(figsize=(11, 8.5))

    for t, name in enumerate(RISK_COLS):
        observed = masks[:, t] > 0.5
        y = labels[observed, t].astype(int)
        p = probs[observed, t].astype(float)

        if y.size == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)

        (line,) = plt.step(
            recall,
            precision,
            where="post",
            linewidth=2.5,
            label=f"{name} (AP={ap:.3f})",
        )

        prevalence = float((y == 1).sum() / len(y))
        plt.hlines(
            y=prevalence,
            xmin=0.0,
            xmax=1.0,
            colors=line.get_color(),
            linestyles="dashed",
            linewidth=2.0,
            label=f"{name} baseline (prev={prevalence:.3f})",
        )

    plt.xlabel("Recall", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Precision", fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    style_axes()
    plt.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def kde_line(x: np.ndarray, grid: np.ndarray, bandwidth: float = KDE_BW) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(x.reshape(-1, 1))
    log_d = kde.score_samples(grid.reshape(-1, 1))
    return np.exp(log_d)


def plot_uncertainty_count_histograms(
    labels: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray,
    uq: np.ndarray,
    out_path: Path,
    title: str,
    tie_break_seed: int = 0,
) -> Dict[str, Any]:
    """
    Makes correct-vs-incorrect uncertainty histograms.
    Correctness is defined using match-prevalence on deterministic probabilities.
    """
    pred_info_by_horizon = {}

    n_h = len(RISK_COLS)
    ncols = 2
    nrows = int(np.ceil(n_h / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 11))
    axes = np.array(axes).reshape(-1)

    for t, name in enumerate(RISK_COLS):
        ax = axes[t]

        observed = masks[:, t] > 0.5
        y = labels[observed, t].astype(int)
        p = probs[observed, t].astype(float)
        u = uq[observed, t].astype(float)

        if y.size == 0:
            ax.axis("off")
            pred_info_by_horizon[name] = {"n_obs": 0}
            continue

        pred, pred_info = compute_pred_info(p, y, tie_break_seed=tie_break_seed + t)
        pred_info_by_horizon[name] = pred_info

        correct = pred == y
        u_corr = u[correct]
        u_inc = u[~correct]

        lo = float(np.min(u))
        hi = float(np.max(u))
        if hi - lo < 1e-12:
            lo -= 1e-6
            hi += 1e-6

        bins = 24
        bin_width = (hi - lo) / bins

        ax.hist(u_corr, bins=bins, range=(lo, hi), alpha=0.50, color=COLOR_CORRECT, label="Correct")
        ax.hist(u_inc, bins=bins, range=(lo, hi), alpha=0.50, color=COLOR_INCORRECT, label="Incorrect")

        grid = np.linspace(lo, hi, 400)
        kde_corr = kde_line(u_corr, grid)
        kde_inc = kde_line(u_inc, grid)

        if kde_corr is not None:
            ax.plot(grid, kde_corr * len(u_corr) * bin_width, color=COLOR_CORRECT, linewidth=2)
        if kde_inc is not None:
            ax.plot(grid, kde_inc * len(u_inc) * bin_width, color=COLOR_INCORRECT, linewidth=2)

        ax.set_title(f"{name} (n={int(y.size)}, +{int(y.sum())})", fontsize=14)
        ax.set_xlabel("Uncertainty", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE)

    for j in range(n_h, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    return pred_info_by_horizon


def plot_uq_boxplots(
    uq_plot_data: Dict[str, Any],
    out_path: Path,
    metric: str,
    title: str,
):
    """
    3 boxplots per horizon:
      overall, accepted+clinician, rejected+clinician
    """
    order = ["overall", ACCEPTED_CLIN_NAME, REJECTED_CLIN_NAME]
    colors = [COLOR_OVERALL, COLOR_ACCEPTED, COLOR_REJECTED]

    box_arrays = []
    positions = []
    centers = []

    group_gap = 3.8
    inner_gap = 0.95
    pos = 0.0

    for h in RISK_COLS:
        these_positions = []
        for i, subset in enumerate(order):
            samples = uq_plot_data[h][subset][f"{metric}_samples"]
            if samples.size == 0:
                samples = np.array([np.nan], dtype=float)
            this_pos = pos + i * inner_gap
            positions.append(this_pos)
            these_positions.append(this_pos)
            box_arrays.append(samples)
        centers.append(float(np.mean(these_positions)))
        pos += group_gap

    plt.figure(figsize=(16, 7.5))
    bp = plt.boxplot(
        box_arrays,
        positions=positions,
        widths=0.60,
        showfliers=False,
        patch_artist=True,
        medianprops={"linewidth": 2},
    )

    for patch, idx in zip(bp["boxes"], range(len(box_arrays))):
        patch.set_facecolor(colors[idx % 3])
        patch.set_alpha(0.35)

    for item in ["whiskers", "caps"]:
        for line in bp[item]:
            line.set_alpha(0.7)

    point_idx = 0
    for h in RISK_COLS:
        for subset in order:
            point = uq_plot_data[h][subset][metric]
            if point is not None:
                plt.scatter([positions[point_idx]], [point], marker="o", s=40, zorder=3)
            point_idx += 1

    ylabel = "AUC" if metric == "auc" else "AUPRC"
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    plt.xticks(centers, RISK_COLS, fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, axis="y", alpha=0.25)

    legend_elements = [
        Patch(facecolor=COLOR_OVERALL, alpha=0.35, label="overall"),
        Patch(facecolor=COLOR_ACCEPTED, alpha=0.35, label=ACCEPTED_CLIN_NAME),
        Patch(facecolor=COLOR_REJECTED, alpha=0.35, label=REJECTED_CLIN_NAME),
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# model / exp helpers
def resolve_exp_dir(exp_dir: Optional[Path]) -> Path:
    if exp_dir is None:
        exp_dir = DEFAULT_EXP_ROOT
    exp_dir = exp_dir.expanduser().resolve()

    if exp_dir.is_dir():
        if (exp_dir / "final_best").exists() or (exp_dir / "best_config.json").exists():
            return exp_dir

    if exp_dir.name.startswith("baseline_grid_then_final_"):
        return exp_dir

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir does not exist: {exp_dir}")

    runs = sorted(exp_dir.glob("baseline_grid_then_final_*"), key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(f"No baseline_grid_then_final_* runs found under: {exp_dir}")
    return runs[-1]


def load_best_checkpoint_and_cfg(exp_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    ckpt_path = exp_dir / "final_best" / "best_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    run_cfg_path = exp_dir / "final_best" / "run_config.json"
    if run_cfg_path.exists():
        cfg = json.loads(run_cfg_path.read_text())
    else:
        best_cfg_path = exp_dir / "best_config.json"
        cfg = json.loads(best_cfg_path.read_text()) if best_cfg_path.exists() else {}

    return ckpt_path, cfg


def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    return BaselineCurrentOnlyModel(
        pretrained_encoder=True,
        num_years=len(RISK_COLS),
        dim=int(cfg.get("dim", 512)),
        mlp_layers=int(cfg.get("num_layers", 1)),
        mlp_hidden=int(cfg.get("hidden_units", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
        freeze_encoder=True,
    )


def make_loader(dataset: torch.utils.data.Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument(
        "--mc_uq_mode",
        type=str,
        default="mutual_info",
        choices=["mutual_info", "entropy_mean", "variance"],
    )
    parser.add_argument("--tie_break_seed", type=int, default=0)
    parser.add_argument(
        "--force_test_all_observed",
        action="store_true",
        help="If set, force all test horizons to be observed for plotting.",
    )
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(Path(args.exp_dir) if args.exp_dir else None)
    ckpt_path, cfg = load_best_checkpoint_and_cfg(exp_dir)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = exp_dir / f"eval_baseline{run_tag}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] EXP_DIR:", exp_dir, flush=True)
    print("[INFO] EVAL_DIR:", eval_dir, flush=True)
    print("[INFO] CHECKPOINT:", ckpt_path, flush=True)
    print("[INFO] MC SAMPLES:", args.mc_samples, flush=True)
    print("[INFO] MC UQ MODE:", args.mc_uq_mode, flush=True)

    # datasets and loaders
    val_ds = CurrentOnlyDataset(split="val")
    test_ds = CurrentOnlyDataset(split="test")

    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    print("[INFO] VAL size :", len(val_ds), flush=True)
    print("[INFO] TEST size:", len(test_ds), flush=True)

    # load model
    model = build_model_from_cfg(cfg).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    state_dict = remap_legacy_cum_keys_for_baseline(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[INFO] missing keys   :", len(missing), flush=True)
    print("[INFO] unexpected keys:", len(unexpected), flush=True)

    # deterministic probs
    val_logits, val_labels, val_masks, _ = collect_logits_labels_masks(model, val_loader)
    test_logits, test_labels, test_masks, test_groups = collect_logits_labels_masks(model, test_loader)

    if args.force_test_all_observed:
        test_masks = np.ones_like(test_labels, dtype=np.float32)

    val_probs = sigmoid_np(val_logits).astype(np.float64)
    test_probs = sigmoid_np(test_logits).astype(np.float64)

    # uncertainty from MC dropout
    val_uq = collect_mc_uncertainty(
        model,
        val_loader,
        mc_samples=int(args.mc_samples),
        uq_mode=args.mc_uq_mode,
    )
    test_uq = collect_mc_uncertainty(
        model,
        test_loader,
        mc_samples=int(args.mc_samples),
        uq_mode=args.mc_uq_mode,
    )

    # plots
    roc_path = eval_dir / "roc_plot.png"
    pr_path = eval_dir / "pr_plot.png"
    uq_hist_path = eval_dir / "uncertainty_correct_incorrect_count.png"

    plot_roc_all_horizons(test_labels, test_probs, test_masks, roc_path, title="ROC Plot")
    plot_pr_all_horizons(test_labels, test_probs, test_masks, pr_path, title="PR Plot")

    correctness_info = plot_uncertainty_count_histograms(
        labels=test_labels,
        probs=test_probs,
        masks=test_masks,
        uq=test_uq,
        out_path=uq_hist_path,
        title="Uncertainty: Correct vs Incorrect",
        tie_break_seed=int(args.tie_break_seed),
    )

    # referral gate from validation UQ
    gate_val = fit_median_val_gate(val_uq, val_masks)

    uq_splits = apply_gate_splits(
        probs=test_probs,
        labels=test_labels,
        masks=test_masks,
        uq=test_uq,
        groups=test_groups,
        gate=gate_val,
    )

    uq_metrics, uq_plot_data = build_uq_metrics_and_plot_data(uq_splits)

    uq_auc_path = eval_dir / "uq_boxplots_auc.png"
    uq_auprc_path = eval_dir / "uq_boxplots_auprc.png"

    plot_uq_boxplots(
        uq_plot_data,
        uq_auc_path,
        metric="auc",
        title="Uncertainty Quantification: AUROC",
    )
    plot_uq_boxplots(
        uq_plot_data,
        uq_auprc_path,
        metric="auprc",
        title="Uncertainty Quantification: AUPRC",
    )

    # deterministic per-horizon metrics
    deterministic_metrics = {}
    for t, h in enumerate(RISK_COLS):
        observed = test_masks[:, t] > 0.5
        y = test_labels[observed, t].astype(int)
        p = test_probs[observed, t].astype(float)
        g = test_groups[observed] if test_groups is not None else None
        deterministic_metrics[h] = {
            k: v for k, v in metric_summary(y, p, h, groups=g).items()
            if not k.endswith("_samples")
        }

    # save json
    metrics = {
        "run_tag": run_tag,
        "exp_dir": str(exp_dir),
        "checkpoint_used": str(ckpt_path),
        "n_val_total": int(len(val_ds)),
        "n_test_total": int(len(test_ds)),
        "mc_samples": int(args.mc_samples),
        "mc_uq_mode": args.mc_uq_mode,
        "force_test_all_observed": bool(args.force_test_all_observed),
        "probabilities": "raw sigmoid(logits)",
        "gate_design": "accept if UQ_test <= median(UQ_val) per horizon",
        "per_horizon_gate": gate_val,
        "correctness_info_match_prevalence": correctness_info,
        "deterministic_test_metrics_with_ci": deterministic_metrics,
        "uq_test_split_metrics_with_ci": uq_metrics,
        "paths": {
            "roc": str(roc_path),
            "pr": str(pr_path),
            "uncertainty_correct_incorrect_count": str(uq_hist_path),
            "uq_boxplots_auc": str(uq_auc_path),
            "uq_boxplots_auprc": str(uq_auprc_path),
        },
        "notes": {
            "mc_dropout_uq": "Uncertainty is computed from MC-dropout probabilities made from raw logits.",
            "accepted_plus_clinician": "Rejected cases are replaced with ground truth.",
            "rejected_plus_clinician": "Accepted cases are replaced with ground truth.",
        },
    }

    metrics_path = eval_dir / "metrics_LIGHT.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\n[INFO] Saved outputs to:", eval_dir, flush=True)
    print("[INFO] metrics json:", metrics_path, flush=True)


if __name__ == "__main__":
    main()