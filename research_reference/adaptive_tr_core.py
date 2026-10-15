# -*- coding: utf-8 -*-
"""Core implementation for V4-S3--S6 nested Adaptive-TR.

The state mapper is fitted only from the supplied training customers/pairs.
It deliberately has no label or score argument. Labels enter only model fitting
and downstream validation metrics.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03_代码" / "src" / "v3_1"))
from behavior_index import cluster_labels, spearman_pairwise_fast  # noqa: E402


def ecdf(values: np.ndarray, training: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    ok = np.isfinite(values)
    base = np.sort(np.asarray(training)[np.isfinite(training)])
    if base.size:
        out[ok] = np.searchsorted(base, values[ok], side="right") / base.size
    return out


def zero_ecdf(values: np.ndarray, training: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    out[finite & (values == 0)] = 0.0
    positive = finite & (values > 0)
    base = np.sort(np.asarray(training)[np.isfinite(training) & (training > 0)])
    if base.size:
        out[positive] = np.searchsorted(base, values[positive], side="right") / base.size
    return out


def quantile(values: np.ndarray, level: float) -> float:
    x = np.asarray(values)[np.isfinite(values)]
    return float(np.quantile(x, level)) if x.size else float("nan")


@dataclass
class DataBundle:
    cfg: dict
    panel: pd.DataFrame
    split: pd.DataFrame
    pairs: pd.DataFrame
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    groups: list[str]
    panel_customer: np.ndarray
    pair_customer: np.ndarray
    month: np.ndarray


def load_bundle(config_path: Path) -> DataBundle:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pth = cfg["paths"]
    vd = pd.read_csv(ROOT / pth["feature_dictionary"], encoding="utf-8-sig")
    kept = vd.loc[vd.keep, "variable"].tolist()
    model_text = next((ROOT / pth["historical_models"]).glob("M0_fold0.txt")).read_text(
        encoding="utf-8")
    features = [line.split("=", 1)[1].split() for line in model_text.splitlines()
                if line.startswith("feature_names=")][0]
    if len(features) != 160 or set(features) != set(kept):
        raise AssertionError("The frozen 160 model features do not match var_dict keep=1")
    categorical = [x for x in features if x in set(cfg["state"]["categorical_columns"])]
    numeric = [x for x in features if x not in categorical]
    panel = pd.read_parquet(ROOT / pth["panel"])
    panel["S_2"] = pd.to_datetime(panel["S_2"])
    panel = panel.sort_values(["customer_ID", "S_2"], kind="mergesort").reset_index(drop=True)
    for col in categorical:
        panel[col] = panel[col].astype("category")
    split = pd.read_parquet(ROOT / pth["split"])
    pairs = pd.read_parquet(ROOT / pth["adjacent_pairs"]).reset_index(drop=True)
    panel_customer = panel.customer_ID.to_numpy()
    pair_customer = pairs.customer_ID.to_numpy()
    if not np.array_equal(panel_customer[pairs.row_curr.to_numpy(int)], pair_customer):
        raise AssertionError("Adjacent-pair row positions do not align to the sorted panel")
    month = panel.S_2.dt.year.to_numpy() * 12 + panel.S_2.dt.month.to_numpy()
    return DataBundle(cfg, panel, split, pairs, features, numeric, categorical,
                      list(cfg["state"]["groups"]), panel_customer, pair_customer, month)


def fit_apply_state(bundle: DataBundle, train_customers: np.ndarray,
                    fit_key: int) -> tuple[pd.DataFrame, dict]:
    """Fit every state transform on train_customers and apply to all pairs.

    Applying to all pairs is safe: every empirical distribution, scale,
    cluster and threshold is estimated solely from training rows/pairs.
    Callers retain only their relevant validation rows.
    """
    cfg = bundle.cfg
    opt = cfg["state"]
    panel, pairs = bundle.panel, bundle.pairs
    train_customer_set = set(train_customers.tolist())
    train_row_mask = np.fromiter((x in train_customer_set for x in bundle.panel_customer),
                                 dtype=bool, count=len(bundle.panel_customer))
    train_pair_mask = np.fromiter((x in train_customer_set for x in bundle.pair_customer),
                                  dtype=bool, count=len(bundle.pair_customer))
    train_rows = np.flatnonzero(train_row_mask)
    train_pairs = np.flatnonzero(train_pair_mask)
    prev = pairs.row_prev.to_numpy(int)
    curr = pairs.row_curr.to_numpy(int)
    num = panel[bundle.numeric].to_numpy(dtype=np.float64)
    n_pair, n_num = len(pairs), len(bundle.numeric)
    ranks = np.full((n_pair, n_num), np.nan, dtype=np.float32)
    scales = {}
    for j, name in enumerate(bundle.numeric):
        vals = num[:, j]
        q25, q75 = np.nanquantile(vals[train_rows], [.25, .75])
        scale = max(float(q75 - q25), float(opt["iqr_epsilon"]))
        raw = np.abs(vals[curr] - vals[prev]) / scale
        clipped = np.minimum(raw, float(opt["change_cap"]))
        ranks[:, j] = ecdf(clipped, clipped[train_pairs])
        scales[name] = scale

    # All transition flags are input-only and observable at pair level.
    feature_pos = {x: i for i, x in enumerate(bundle.features)}
    m2o = np.zeros((n_pair, len(bundle.features)), dtype=bool)
    o2m = np.zeros_like(m2o)
    cat_change = np.zeros_like(m2o)
    for name in bundle.features:
        j = feature_pos[name]
        values = panel[name].to_numpy()
        pm, cm = pd.isna(values[prev]), pd.isna(values[curr])
        m2o[:, j], o2m[:, j] = pm & ~cm, ~pm & cm
        if name in bundle.categorical:
            cat_change[:, j] = ~pm & ~cm & (values[prev] != values[curr])

    # Correlation clusters are re-fitted in this exact training set.
    rng = np.random.RandomState(int(cfg["seed"]) + int(opt["cluster_seed_offset"])
                                + int(fit_key))
    ids = np.sort(np.asarray(train_customers))
    chosen = rng.choice(ids, size=min(int(opt["cluster_sample_customers"]), len(ids)),
                        replace=False)
    cluster_sample = np.flatnonzero(train_pair_mask & np.isin(bundle.pair_customer, chosen))
    num_pos = {x: i for i, x in enumerate(bundle.numeric)}
    group_u = np.full((n_pair, len(bundle.groups)), np.nan, dtype=np.float32)
    valid_by_group = np.zeros((n_pair, len(bundle.groups)), dtype=np.int16)
    extreme_by_group = np.zeros_like(valid_by_group)
    transition_by_group = np.zeros_like(valid_by_group)
    transition_units = np.zeros(len(bundle.groups), dtype=np.int16)
    signatures, tail_thresholds = {}, {}

    for gi, prefix in enumerate(bundle.groups):
        cols = [x for x in bundle.numeric if x.startswith(prefix)]
        positions = np.asarray([num_pos[x] for x in cols])
        sample = pd.DataFrame(ranks[np.ix_(cluster_sample, positions)], columns=cols)
        rho, _ = spearman_pairwise_fast(sample, int(opt["correlation_min_pair_overlap"]))
        labels = cluster_labels(rho, float(opt["cluster_abs_spearman_threshold"]))
        clusters = [np.flatnonzero(labels == key) for key in sorted(set(labels))]
        signatures[prefix] = [sorted(cols[int(i)] for i in members)
                              for members in clusters]
        transition_units[gi] += len(clusters)
        center_sum = np.zeros(n_pair, dtype=np.float64)
        tail_thresholds[prefix] = []
        for members in clusters:
            pos = positions[members]
            member_r = ranks[:, pos]
            valid_member = np.isfinite(member_r).sum(axis=1)
            need = max(1, int(np.ceil(len(members) * float(
                opt["cluster_valid_member_fraction"]))))
            valid_cluster = valid_member >= need
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                center = np.nanmedian(member_r, axis=1)
                tail = np.nanmax(member_r, axis=1)
            center[~valid_cluster], tail[~valid_cluster] = np.nan, np.nan
            good = np.isfinite(center)
            center_sum[good] += center[good]
            valid_by_group[good, gi] += 1
            threshold = quantile(tail[train_pairs], float(opt["shock_tail_quantile"]))
            extreme_by_group[:, gi] += valid_cluster & (tail > threshold)
            tail_thresholds[prefix].append(threshold)
            fp = np.asarray([feature_pos[cols[int(i)]] for i in members])
            transition_by_group[:, gi] += (m2o[:, fp].any(axis=1)
                                            | o2m[:, fp].any(axis=1)
                                            | cat_change[:, fp].any(axis=1))
        needed = int(np.ceil(len(clusters) * float(opt[
            "group_min_valid_cluster_fraction"])))
        good_group = valid_by_group[:, gi] >= needed
        component = np.full(n_pair, np.nan, dtype=np.float32)
        component[good_group] = (center_sum[good_group]
                                 / valid_by_group[good_group, gi])
        group_u[:, gi] = ecdf(component, component[train_pairs])

    # Categorical singleton transition units.
    for name in bundle.categorical:
        gi = bundle.groups.index(name[:2])
        transition_units[gi] += 1
        fp = feature_pos[name]
        transition_by_group[:, gi] += (m2o[:, fp] | o2m[:, fp] | cat_change[:, fp])

    valid_clusters = valid_by_group.sum(axis=1)
    extreme_clusters = extreme_by_group.sum(axis=1)
    ecb = extreme_clusters / np.maximum(valid_clusters, 1)
    ecb[valid_clusters == 0] = np.nan
    u_shock = zero_ecdf(ecb, ecb[train_pairs])
    total_transition_units = int(transition_units.sum())
    tb = transition_by_group.sum(axis=1) / total_transition_units
    u_trans = zero_ecdf(tb, tb[train_pairs])
    complete = np.isfinite(group_u).all(axis=1)
    broad = np.full(n_pair, np.nan, dtype=np.float32)
    broad[complete] = group_u[complete].max(axis=1)
    c_final = np.maximum.reduce([broad, u_shock, u_trans])
    gate = ecdf(c_final, c_final[train_pairs])
    layer = np.full(n_pair, "medium", dtype=object)
    layer[~np.isfinite(gate)] = "insufficient"
    layer[np.isfinite(gate) & (gate <= float(opt["stable_gate_max"]))] = "stable"
    layer[np.isfinite(gate) & (gate >= float(opt["high_gate_min"]))] = "high"
    out = pd.DataFrame({
        "pair_index": np.arange(n_pair, dtype=np.int32),
        "customer_ID": bundle.pair_customer,
        "C_broad": broad, "ECB_p99": ecb, "U_shock": u_shock,
        "TB": tb, "U_trans": u_trans, "C_final": c_final,
        "G": gate, "layer": layer,
    })
    for gi, prefix in enumerate(bundle.groups):
        out["U_" + prefix[0]] = group_u[:, gi]
    cluster_sizes = [len(m) for gs in signatures.values() for m in gs]
    meta = {
        "fit_key": int(fit_key), "n_training_customers": int(len(ids)),
        "n_training_rows": int(len(train_rows)), "n_training_pairs": int(len(train_pairs)),
        "n_cluster_sample_pairs": int(len(cluster_sample)),
        "cluster_members": signatures,
        "continuous_cluster_count": int(len(cluster_sizes)),
        "non_singleton_cluster_count": int(sum(x > 1 for x in cluster_sizes)),
        "transition_unit_count": total_transition_units,
        "transition_units_by_group": dict(zip(bundle.groups, transition_units.tolist())),
        "iqr_by_variable": scales, "cluster_tail_q99": tail_thresholds,
        "training_gate_quantiles": {str(x): quantile(gate[train_pairs], x)
                                    for x in [0, .25, .5, .9, .95, 1]},
        "training_layer_counts": pd.Series(layer[train_pairs]).value_counts().to_dict(),
        "insufficient_gate_for_weighting": float(opt["insufficient_gate_for_weighting"]),
    }
    return out, meta


def lgb_params(bundle: DataBundle) -> dict:
    legacy = yaml.safe_load((ROOT / "02_配置" / "config.yaml").read_text(encoding="utf-8"))
    p = dict(legacy["lgbm"])
    p.pop("n_estimators", None)
    p.update(bagging_freq=0, seed=42, feature_fraction_seed=42,
             bagging_seed=42, data_random_seed=42, num_threads=0,
             verbose=-1, deterministic=True, force_row_wise=True,
             metric="None")
    return p


def make_objective(prev_pos: np.ndarray, next_pos: np.ndarray,
                   prev_weight: np.ndarray, next_weight: np.ndarray,
                   lam: float):
    def objective(preds, dataset):
        y = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        grad = p - y
        hess = p * (1.0 - p)
        hp, hn = prev_pos >= 0, next_pos >= 0
        if lam:
            grad[hp] += 2 * lam * prev_weight[hp] * (preds[hp] - preds[prev_pos[hp]])
            grad[hn] += 2 * lam * next_weight[hn] * (preds[hn] - preds[next_pos[hn]])
            hess += 2 * lam * (prev_weight * hp + next_weight * hn)
        return grad, hess
    return objective


def adjacency_with_gate(bundle: DataBundle, train_rows: np.ndarray,
                        train_pair_indices: np.ndarray, gate: np.ndarray,
                        gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.full(len(bundle.panel), -1, dtype=np.int32)
    pos[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp = bundle.pairs.row_prev.to_numpy(int)[train_pair_indices]
    pc = bundle.pairs.row_curr.to_numpy(int)[train_pair_indices]
    a, b = pos[pp], pos[pc]
    if np.any((a < 0) | (b < 0)):
        raise AssertionError("A training edge crossed the customer split")
    g = gate[train_pair_indices].astype(float)
    g[~np.isfinite(g)] = float(bundle.cfg["state"]["insufficient_gate_for_weighting"])
    w = np.exp(-float(gamma) * g)
    prev_pos = np.full(len(train_rows), -1, dtype=np.int32)
    next_pos = np.full(len(train_rows), -1, dtype=np.int32)
    prev_weight = np.zeros(len(train_rows), dtype=np.float64)
    next_weight = np.zeros(len(train_rows), dtype=np.float64)
    prev_pos[b], next_pos[a] = a, b
    prev_weight[b], next_weight[a] = w, w
    return prev_pos, next_pos, prev_weight, next_weight


def fit_model(bundle: DataBundle, train_customers: np.ndarray, state: pd.DataFrame,
              lam: float, gamma: float, model_path: Path | None = None):
    import lightgbm as lgb
    train_set = set(train_customers.tolist())
    train_rows = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.panel_customer), dtype=bool,
        count=len(bundle.panel_customer)))
    train_pairs = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.pair_customer), dtype=bool,
        count=len(bundle.pair_customer)))
    prev, nxt, wp, wn = adjacency_with_gate(
        bundle, train_rows, train_pairs, state.G.to_numpy(float), gamma)
    X = bundle.panel.loc[train_rows, bundle.features]
    y = bundle.panel.target.to_numpy(float)[train_rows]
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    intercept = float(np.log(p0 / (1 - p0)))
    ds = lgb.Dataset(X, label=y, categorical_feature=bundle.categorical,
                     init_score=np.full(len(y), intercept), free_raw_data=False)
    params = lgb_params(bundle)
    params["objective"] = make_objective(prev, nxt, wp, wn, float(lam))
    booster = lgb.train(params, ds, num_boost_round=int(bundle.cfg["model"]["n_estimators"]))
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(booster.model_to_string(), encoding="utf-8")
    diagnostics = {
        "lambda": float(lam), "gamma": float(gamma),
        "n_train_customers": int(len(train_customers)),
        "n_train_rows": int(len(train_rows)), "n_edges": int(len(train_pairs)),
        "init_score_log_odds": intercept,
        "edge_weight_min": float(np.min(wp[prev >= 0])) if len(train_pairs) else None,
        "edge_weight_mean": float(np.mean(wp[prev >= 0])) if len(train_pairs) else None,
        "edge_weight_max": float(np.max(wp[prev >= 0])) if len(train_pairs) else None,
    }
    return booster, intercept, diagnostics


def predict_rows(bundle: DataBundle, booster, intercept: float,
                 customers: np.ndarray, cv_fold: int) -> pd.DataFrame:
    chosen = set(customers.tolist())
    rows = np.flatnonzero(np.fromiter((x in chosen for x in bundle.panel_customer),
                                     dtype=bool, count=len(bundle.panel_customer)))
    raw = booster.predict(bundle.panel.loc[rows, bundle.features], raw_score=True) + intercept
    return pd.DataFrame({
        "row_index": rows, "customer_ID": bundle.panel_customer[rows],
        "t": bundle.panel.t.to_numpy()[rows], "s": raw,
        "p": 1.0 / (1.0 + np.exp(-raw)), "cv_fold": int(cv_fold),
        "target": bundle.panel.target.to_numpy()[rows],
    })


def verify_gamma_zero() -> dict:
    """Numerically verify adaptive gamma=0 equals the unweighted global objective."""
    rng = np.random.default_rng(917)
    n = 13
    prev = np.full(n, -1, int); nxt = np.full(n, -1, int)
    for a, b in [(0, 1), (1, 2), (4, 5), (5, 6), (9, 10), (10, 11), (11, 12)]:
        prev[b], nxt[a] = a, b
    edge_g = rng.uniform(size=n)
    wp0 = np.where(prev >= 0, np.exp(-0.0 * edge_g), 0.0)
    wn0 = np.where(nxt >= 0, np.exp(-0.0 * edge_g), 0.0)
    wpg = (prev >= 0).astype(float); wng = (nxt >= 0).astype(float)
    y, f = rng.integers(0, 2, n).astype(float), rng.normal(size=n)
    class DS:
        def get_label(self):
            return y
    ga, ha = make_objective(prev, nxt, wp0, wn0, .1)(f.copy(), DS())
    gg, hg = make_objective(prev, nxt, wpg, wng, .1)(f.copy(), DS())
    return {
        "gamma_zero_weights_exactly_one": bool(np.all(wp0[prev >= 0] == 1)
                                                and np.all(wn0[nxt >= 0] == 1)),
        "max_abs_gradient_difference": float(np.max(np.abs(ga - gg))),
        "max_abs_hessian_difference": float(np.max(np.abs(ha - hg))),
        "passed": bool(np.array_equal(ga, gg) and np.array_equal(ha, hg)),
    }


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
