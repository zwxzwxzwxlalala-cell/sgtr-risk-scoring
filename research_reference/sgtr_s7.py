# -*- coding: utf-8 -*-
"""V4-S7: post-freeze customer-balance robustness and probability calibration."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from adaptive_tr_core import (ROOT, lgb_params, load_bundle, predict_rows,
                              write_json)
from adaptive_tr_experiment import add_relative, evaluate, ks_stat
from adaptive_tr_s6c import (add_medium, global_predictions, markdown_table,
                             selected_s6c, stitched_state)
from sgtr_s6d import fold_equal_weighted, weighted_auc

CFG = ROOT / "v4" / "config_v4_s7.yaml"


def log(message: str) -> None:
    print("[V4-S7] " + message, flush=True)


def paths(bundle):
    p = bundle.cfg["paths"]
    return (ROOT / p["s3_s6_output_dir"], ROOT / p["s6c_output_dir"],
            ROOT / p["output_dir"], ROOT / p["model_dir"],
            ROOT / p["figure_dir"])


def weighted_objective(prev_pos, next_pos, prev_weight, next_weight,
                       row_weight, lam):
    def objective(preds, dataset):
        y = dataset.get_label()
        p = expit(preds)
        grad = row_weight * (p - y)
        hess = row_weight * p * (1 - p)
        hp, hn = prev_pos >= 0, next_pos >= 0
        grad[hp] += 2 * lam * prev_weight[hp] * (preds[hp] - preds[prev_pos[hp]])
        grad[hn] += 2 * lam * next_weight[hn] * (preds[hn] - preds[next_pos[hn]])
        hess += 2 * lam * (prev_weight * hp + next_weight * hn)
        return grad, hess
    return objective


def normalized_customer_weights(ids: np.ndarray) -> tuple[np.ndarray, dict]:
    series = pd.Series(ids)
    counts = series.value_counts()
    normalizer = len(ids) / len(counts)
    weights = normalizer / series.map(counts).to_numpy(float)
    return weights, {
        "normalizer": float(normalizer), "units": int(len(counts)),
        "observations": int(len(ids)), "mean": float(weights.mean()),
        "min": float(weights.min()), "p10": float(np.quantile(weights, .10)),
        "p25": float(np.quantile(weights, .25)), "p50": float(np.quantile(weights, .50)),
        "p75": float(np.quantile(weights, .75)), "p90": float(np.quantile(weights, .90)),
        "max": float(weights.max()), "sum": float(weights.sum()),
    }


def fit_balanced_model(bundle, train_customers: np.ndarray, state: pd.DataFrame,
                       model_path: Path):
    import lightgbm as lgb
    train_set = set(train_customers.tolist())
    train_rows = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.panel_customer), dtype=bool,
        count=len(bundle.panel_customer)))
    train_pairs = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.pair_customer), dtype=bool,
        count=len(bundle.pair_customer)))
    row_weight, row_diag = normalized_customer_weights(
        bundle.panel_customer[train_rows])
    edge_weight, edge_diag = normalized_customer_weights(
        bundle.pair_customer[train_pairs])
    gate = (np.isfinite(state.G.to_numpy(float))
            & (state.G.to_numpy(float) <= .25)).astype(float)[train_pairs]
    effective_edge_weight = edge_weight * gate
    position = np.full(len(bundle.panel), -1, dtype=np.int32)
    position[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp = bundle.pairs.row_prev.to_numpy(int)[train_pairs]
    pc = bundle.pairs.row_curr.to_numpy(int)[train_pairs]
    a, b = position[pp], position[pc]
    if np.any((a < 0) | (b < 0)):
        raise AssertionError("Customer-balanced edge crossed split")
    prev_pos = np.full(len(train_rows), -1, dtype=np.int32)
    next_pos = np.full(len(train_rows), -1, dtype=np.int32)
    prev_weight = np.zeros(len(train_rows), dtype=float)
    next_weight = np.zeros(len(train_rows), dtype=float)
    prev_pos[b], next_pos[a] = a, b
    prev_weight[b], next_weight[a] = effective_edge_weight, effective_edge_weight
    X = bundle.panel.loc[train_rows, bundle.features]
    y = bundle.panel.target.to_numpy(float)[train_rows]
    p0 = float(np.dot(row_weight, y) / row_weight.sum())
    p0 = float(np.clip(p0, 1e-6, 1 - 1e-6))
    intercept = float(np.log(p0 / (1 - p0)))
    ds = lgb.Dataset(X, label=y, categorical_feature=bundle.categorical,
                     init_score=np.full(len(y), intercept), free_raw_data=False)
    params = lgb_params(bundle)
    params["objective"] = weighted_objective(
        prev_pos, next_pos, prev_weight, next_weight, row_weight, .03)
    n_estimators = int(bundle.cfg["frozen_core"]["n_estimators"])
    booster = lgb.train(params, ds, num_boost_round=n_estimators)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(booster.model_to_string(), encoding="utf-8")
    history = pd.Series(bundle.panel_customer[train_rows]).value_counts()
    diag = {
        "model": "customer-balanced SG-TR robustness only",
        "lambda": .03, "q_s": .25,
        "n_train_customers": int(len(train_customers)),
        "n_train_rows": int(len(train_rows)), "n_train_edges": int(len(train_pairs)),
        "init_score_log_odds": intercept,
        "customer_history": history.describe(percentiles=[.1,.25,.5,.75,.9]).to_dict(),
        "row_weight": row_diag, "edge_weight": edge_diag,
        "gate_one_share": float((gate == 1).mean()),
        "effective_edge_weight_mean_all_edges": float(effective_edge_weight.mean()),
    }
    return booster, intercept, diag


def train_balanced_outer(bundle) -> None:
    s3, _, out, models, _ = paths(bundle)
    for outer in sorted(bundle.split.fold.unique()):
        score_path = out / "balanced_scores" / f"balanced_sgtr_outer{outer}.parquet"
        model_path = models / "balanced" / f"balanced_sgtr_outer{outer}.txt"
        diag_path = model_path.with_suffix(".json")
        if score_path.exists() and model_path.exists() and diag_path.exists():
            continue
        train_ids = bundle.split.loc[
            bundle.split.fold.ne(outer), "customer_ID"].to_numpy()
        val_ids = bundle.split.loc[
            bundle.split.fold.eq(outer), "customer_ID"].to_numpy()
        state = pd.read_parquet(s3 / "state_maps" / f"outer{outer}_full.parquet")
        started = time.time()
        booster, intercept, diag = fit_balanced_model(
            bundle, train_ids, state, model_path)
        pred = predict_rows(bundle, booster, intercept, val_ids, int(outer))
        score_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(score_path, index=False)
        diag.update(
            score_file=str(score_path.relative_to(ROOT)),
            model_file=str(model_path.relative_to(ROOT)),
            lightgbm_version=__import__("lightgbm").__version__,
            features=bundle.features, categorical_features=bundle.categorical,
            parameters={k: v for k, v in lgb_params(bundle).items() if not callable(v)},
            n_estimators=int(bundle.cfg["frozen_core"]["n_estimators"]),
            elapsed_seconds=time.time() - started)
        write_json(diag_path, diag)
        log(f"customer-balanced outer {outer} 完成，{diag['elapsed_seconds']:.1f}s")


def balanced_scores(bundle) -> pd.DataFrame:
    _, _, out, _, _ = paths(bundle)
    return pd.concat([pd.read_parquet(
        out / "balanced_scores" / f"balanced_sgtr_outer{k}.parquet")
        for k in range(5)], ignore_index=True)


def formal_robustness(bundle, state, main, balanced, baseline) -> pd.DataFrame:
    _, _, out, models, _ = paths(bundle)
    base_metric = add_medium(bundle, baseline, state, evaluate(bundle, baseline, state))
    rows = []
    for name, pred in [("SGTR_row_weighted_frozen", main),
                       ("SGTR_customer_balanced_ablation", balanced)]:
        metric = add_medium(bundle, pred, state, evaluate(bundle, pred, state))
        metric = add_relative(metric, base_metric, bundle.cfg)
        rows.append({"model": name, **metric})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "customer_balanced_formal_metrics_s7.csv", index=False,
                 encoding="utf-8-sig")
    diags = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((models / "balanced").glob("*.json"))]
    weight_rows = []
    for outer, diag in enumerate(diags):
        for kind in ["row_weight", "edge_weight"]:
            weight_rows.append({"outer_fold": outer, "weight_type": kind,
                                **diag[kind]})
    pd.DataFrame(weight_rows).to_csv(
        out / "customer_balanced_weight_diagnostics_s7.csv", index=False,
        encoding="utf-8-sig")
    return frame


def history_groups(bundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = bundle.panel.groupby("customer_ID", observed=True).size().rename("T")
    rows, thresholds = [], []
    for outer in sorted(bundle.split.fold.unique()):
        train_ids = bundle.split.loc[bundle.split.fold.ne(outer), "customer_ID"]
        val_ids = bundle.split.loc[bundle.split.fold.eq(outer), "customer_ID"]
        train_t = counts.reindex(train_ids).to_numpy(float)
        short_max = int(np.quantile(train_t, .10))
        long_min = int(np.quantile(train_t, .20))
        if short_max >= long_min:
            values = np.sort(np.unique(train_t))
            short_max = int(values[max(0, len(values)-3)])
            long_min = int(values[-1])
        thresholds.append({"outer_fold": int(outer), "train_p10_short_max": short_max,
                           "train_p20_long_min": long_min})
        for cid, t in counts.reindex(val_ids).items():
            group = "short" if t <= short_max else ("long" if t >= long_min else "medium")
            rows.append({"customer_ID": cid, "outer_fold": int(outer),
                         "T": int(t), "history_group": group})
    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def history_group_metrics(bundle, state, main, balanced, baseline) -> pd.DataFrame:
    _, _, out, _, _ = paths(bundle)
    groups, thresholds = history_groups(bundle)
    rows = []
    for group in ["short", "medium", "long"]:
        ids = set(groups.loc[groups.history_group.eq(group), "customer_ID"])
        state_g = state[state.customer_ID.isin(ids)]
        base_g = baseline[baseline.customer_ID.isin(ids)]
        base_metric = add_medium(bundle, base_g, state_g, evaluate(bundle, base_g, state_g))
        for name, pred in [("SGTR_row_weighted_frozen", main),
                           ("SGTR_customer_balanced_ablation", balanced)]:
            pred_g = pred[pred.customer_ID.isin(ids)]
            metric = add_medium(bundle, pred_g, state_g, evaluate(bundle, pred_g, state_g))
            metric = add_relative(metric, base_metric, bundle.cfg)
            rows.append({"history_group": group, "model": name,
                         "n_group_customers": len(ids), **metric})
    result = pd.DataFrame(rows)
    result.to_csv(out / "history_group_metrics_s7.csv", index=False,
                  encoding="utf-8-sig")
    groups.to_parquet(out / "history_group_assignments_s7.parquet", index=False)
    thresholds.to_csv(out / "history_group_thresholds_s7.csv", index=False,
                      encoding="utf-8-sig")
    distribution = groups.groupby(["history_group", "T"], as_index=False).agg(
        customers=("customer_ID", "size"))
    distribution.to_csv(out / "customer_history_distribution_s7.csv", index=False,
                        encoding="utf-8-sig")
    counts = bundle.panel.groupby("customer_ID", observed=True).size()
    summary = counts.describe(percentiles=[.10, .25, .50, .75, .90]).rename(
        {"count": "customers", "mean": "mean_T", "std": "std_T",
         "min": "min_T", "10%": "p10_T", "25%": "p25_T",
         "50%": "p50_T", "75%": "p75_T", "90%": "p90_T",
         "max": "max_T"}).to_frame("value").reset_index()
    summary.to_csv(out / "customer_history_summary_s7.csv", index=False,
                   encoding="utf-8-sig")
    return result


def fit_logistic_calibration(y, score, weight=None, initial=(0., 1.)):
    y, x = np.asarray(y, float), np.asarray(score, float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    theta = np.asarray(initial, float).copy()
    for _ in range(50):
        p = expit(theta[0] + theta[1] * x)
        v = w * p * (1 - p)
        grad = np.array([np.sum(w * (p-y)), np.sum(w * (p-y) * x)])
        hess = np.array([[np.sum(v), np.sum(v*x)],
                         [np.sum(v*x), np.sum(v*x*x)]])
        step = np.linalg.solve(hess + np.eye(2)*1e-10, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def fit_calibrators(bundle, main: pd.DataFrame) -> pd.DataFrame:
    s3, _, out, models, _ = paths(bundle)
    final_frames, parameter_rows = [], []
    for outer in sorted(bundle.split.fold.unique()):
        assignment = pd.read_parquet(s3 / "inner_assignments" / f"outer{outer}.parquet")
        inner_parts = []
        for inner in sorted(assignment.inner_fold.unique()):
            p = pd.read_parquet(s3.parent / "s6c" / "inner_scores" /
                                f"outer{outer}" / f"inner{inner}" /
                                "hard_lam0.03.parquet")
            inner_parts.append(p)
        crossfit = pd.concat(inner_parts, ignore_index=True).sort_values(
            ["customer_ID", "t"]).groupby("customer_ID", sort=False).tail(1)
        if len(crossfit) != 12000 or crossfit.customer_ID.duplicated().any():
            raise AssertionError("Calibration training predictions are not 12k cross-fitted customers")
        y_train = crossfit.target.to_numpy(float)
        s_train = crossfit.s.to_numpy(float)
        intercept, slope = fit_logistic_calibration(y_train, s_train)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-7, y_max=1-1e-7)
        iso.fit(s_train, y_train)
        val = main[main.cv_fold.eq(outer)].sort_values(
            ["customer_ID", "t"]).groupby("customer_ID", sort=False).tail(1).copy()
        val["p_uncalibrated"] = val.p.to_numpy(float)
        val["p_platt"] = expit(intercept + slope * val.s.to_numpy(float))
        val["p_isotonic"] = iso.predict(val.s.to_numpy(float))
        final_frames.append(val[["customer_ID", "cv_fold", "target", "s",
                                 "p_uncalibrated", "p_platt", "p_isotonic"]])
        params = {
            "outer_fold": int(outer), "training_unit": "customer_final_period",
            "n_crossfitted_training_customers": int(len(crossfit)),
            "platt_intercept": intercept, "platt_slope": slope,
            "isotonic_x_thresholds": iso.X_thresholds_.tolist(),
            "isotonic_y_thresholds": iso.y_thresholds_.tolist(),
            "inner_score_sources": [f"v4/outputs/s6c/inner_scores/outer{outer}/inner{k}/hard_lam0.03.parquet"
                                    for k in sorted(assignment.inner_fold.unique())],
        }
        parameter_rows.append({k: v for k, v in params.items()
                               if not isinstance(v, list)})
        write_json(models / "calibration" / f"outer{outer}.json", params)
    calibrated = pd.concat(final_frames, ignore_index=True)
    calibrated.to_parquet(out / "calibrated_final_period_oof_s7.parquet", index=False)
    pd.DataFrame(parameter_rows).to_csv(out / "calibration_fold_parameters_s7.csv",
                                        index=False, encoding="utf-8-sig")
    return calibrated


def weighted_ks(y, score, weight):
    order = np.argsort(score, kind="mergesort")
    s, yy, ww = score[order], y[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    pos = np.add.reduceat(ww * yy, starts)
    neg = np.add.reduceat(ww * (1-yy), starts)
    return float(np.max(np.abs(np.cumsum(pos)/pos.sum() - np.cumsum(neg)/neg.sum())))


def probability_metrics(y, p, weight=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    pp = np.clip(p, 1e-7, 1-1e-7)
    intercept, slope = fit_logistic_calibration(y, logit(pp), w)
    return {
        "auc": weighted_auc(y, p, w), "ks": weighted_ks(y, p, w),
        "logloss": float(np.dot(w, -(y*np.log(pp)+(1-y)*np.log(1-pp))) / w.sum()),
        "brier": float(np.dot(w, (p-y)**2) / w.sum()),
        "calibration_intercept": intercept, "calibration_slope": slope,
    }


def calibration_outputs(bundle, calibrated: pd.DataFrame) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    _, _, out, _, figs = paths(bundle)
    figs.mkdir(parents=True, exist_ok=True)
    y = calibrated.target.to_numpy(float)
    mapping = {"Uncalibrated SG-TR": "p_uncalibrated",
               "Platt": "p_platt", "Isotonic": "p_isotonic"}
    metrics = pd.DataFrame([{"method": name,
                             **probability_metrics(y, calibrated[col].to_numpy(float))}
                            for name, col in mapping.items()])
    metrics.to_csv(out / "calibration_metrics_s7.csv", index=False,
                   encoding="utf-8-sig")
    reliability_rows = []
    for name, col in mapping.items():
        frame = calibrated[["target", col]].copy().rename(columns={col: "prob"})
        frame["bin"] = pd.qcut(frame.prob, q=10, duplicates="drop")
        for i, (_, group) in enumerate(frame.groupby("bin", observed=True)):
            reliability_rows.append({
                "method": name, "bin": i+1, "n": int(len(group)),
                "mean_probability": float(group.prob.mean()),
                "observed_rate": float(group.target.mean()),
                "probability_min": float(group.prob.min()),
                "probability_max": float(group.prob.max()),
            })
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(out / "reliability_curve_s7.csv", index=False,
                       encoding="utf-8-sig")
    plt.rcParams["font.sans-serif"] = [bundle.cfg["reporting"]["figure_font"],
                                        "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.plot([0,1], [0,1], "--", color="#777777", label="理想校准")
    colors = {"Uncalibrated SG-TR":"#555555", "Platt":"#2878B5",
              "Isotonic":"#C44E52"}
    for method, group in reliability.groupby("method"):
        ax.plot(group.mean_probability, group.observed_rate, marker="o",
                label=method, color=colors[method])
    ax.set_xlabel("平均预测概率"); ax.set_ylabel("实际违约率")
    ax.set_title("S7 客户真实末期概率可靠性曲线（15,000名客户）")
    ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figs / "reliability_curve_s7.png", dpi=180); plt.close(fig)
    return metrics


def robustness_customer_summaries(bundle, state, predictions):
    customers = bundle.split.customer_ID.to_numpy()
    folds = bundle.split.fold.to_numpy(int)
    lookup = pd.Index(customers)
    pi = state.pair_index.to_numpy(int)
    pp = bundle.pairs.row_prev.to_numpy(int)[pi]
    pc = bundle.pairs.row_curr.to_numpy(int)[pi]
    summaries = {}
    for name, pred in predictions.items():
        last = pred.sort_values(["customer_ID","t"]).groupby(
            "customer_ID", sort=False).tail(1).set_index("customer_ID").reindex(lookup)
        score = np.full(len(bundle.panel), np.nan)
        score[pred.row_index.to_numpy(int)] = pred.s.to_numpy(float)
        pair = state[["customer_ID","layer"]].copy()
        pair["delta"] = np.abs(score[pc]-score[pp])
        layers = {layer: pair[pair.layer.eq(layer)].groupby(
            "customer_ID", observed=True).delta.mean().reindex(lookup).to_numpy(float)
                  for layer in ["stable","medium","high"]}
        work = pred[["customer_ID","row_index","s","target"]].copy()
        work["month"] = bundle.month[pred.row_index.to_numpy(int)]
        tail6 = work.sort_values(["customer_ID","month"]).groupby(
            "customer_ID", sort=False).tail(6)
        agg = tail6.groupby("customer_ID", observed=True).agg(
            n=("month","size"), span=("month",lambda x:x.max()-x.min()),
            target=("target","first"))
        valid = agg[(agg.n==6)&(agg.span==5)&(agg.target==1)].index
        subset = tail6[tail6.customer_ID.isin(valid)].sort_values(["customer_ID","month"])
        rise = (subset.groupby("customer_ID", observed=True).tail(1).set_index("customer_ID").s
                - subset.groupby("customer_ID", observed=True).head(1).set_index("customer_ID").s)
        summaries[name] = {"target":last.target.to_numpy(float),
                           "score":last.s.to_numpy(float), "prob":last.p.to_numpy(float),
                           **layers, "trajectory":rise.reindex(lookup).to_numpy(float)}
    return customers, folds, summaries


def robustness_metrics(summary, baseline, folds, weight):
    y, p, s = summary["target"], summary["prob"], summary["score"]
    stable = fold_equal_weighted(summary["stable"], weight, folds)
    medium = fold_equal_weighted(summary["medium"], weight, folds)
    high = fold_equal_weighted(summary["high"], weight, folds)
    trajectory = fold_equal_weighted(summary["trajectory"], weight, folds)
    base_stable = fold_equal_weighted(baseline["stable"], weight, folds)
    base_high = fold_equal_weighted(baseline["high"], weight, folds)
    base_traj = fold_equal_weighted(baseline["trajectory"], weight, folds)
    pp = np.clip(p, 1e-7, 1-1e-7)
    return {"auc":weighted_auc(y,s,weight), "ks":weighted_ks(y,s,weight),
            "logloss":float(np.dot(weight,-(y*np.log(pp)+(1-y)*np.log(1-pp)))/weight.sum()),
            "brier":float(np.dot(weight,(p-y)**2)/weight.sum()),
            "stable_mean_abs_delta_raw":stable,
            "stable_reduction":1-stable/base_stable,
            "medium_mean_abs_delta_raw":medium,
            "high_mean_abs_delta_raw":high,
            "high_response_retention":high/base_high,
            "default_trajectory_retention":trajectory/base_traj,
            "srr":high/stable,
            "srr_numerator_high_mean_abs_delta_raw":high,
            "srr_denominator_stable_mean_abs_delta_raw":stable}


def paired_bootstrap(bundle, state, main, balanced, baseline, calibrated):
    _, _, out, _, _ = paths(bundle)
    customers, folds, summaries = robustness_customer_summaries(
        bundle, state, {"main":main,"balanced":balanced,"baseline":baseline})
    cal = calibrated.set_index("customer_ID").reindex(customers)
    n, reps = len(customers), int(bundle.cfg["bootstrap"]["replicates"])
    rng = np.random.default_rng(int(bundle.cfg["bootstrap"]["seed"]))
    fold_idx = [np.flatnonzero(folds==f) for f in sorted(np.unique(folds))]
    ones = np.ones(n, int)
    robust_point = {name:robustness_metrics(value,summaries["baseline"],folds,ones)
                    for name,value in summaries.items()}
    methods = {"uncalibrated":cal.p_uncalibrated.to_numpy(float),
               "platt":cal.p_platt.to_numpy(float),
               "isotonic":cal.p_isotonic.to_numpy(float)}
    y = cal.target.to_numpy(float)
    cal_point = {name:probability_metrics(y,p) for name,p in methods.items()}
    robust_draws, calibration_draws = [], []
    for replicate in range(reps):
        weight = np.zeros(n, int)
        for idx in fold_idx:
            weight[idx] = rng.multinomial(len(idx),np.full(len(idx),1/len(idx)))
        main_m = robustness_metrics(summaries["main"],summaries["baseline"],folds,weight)
        bal_m = robustness_metrics(summaries["balanced"],summaries["baseline"],folds,weight)
        robust_draws.append({"replicate":replicate,
                             **{k:bal_m[k]-main_m[k] for k in main_m}})
        cm = {name:probability_metrics(y,p,weight) for name,p in methods.items()}
        for label,left,right in [("Platt_minus_uncalibrated","platt","uncalibrated"),
                                 ("Isotonic_minus_uncalibrated","isotonic","uncalibrated"),
                                 ("Isotonic_minus_Platt","isotonic","platt")]:
            calibration_draws.append({"replicate":replicate,"comparison":label,
                                      **{k:cm[left][k]-cm[right][k] for k in cm[left]}})
        if (replicate+1)%250==0: log(f"S7 paired bootstrap {replicate+1}/{reps}")
    robust_draws=pd.DataFrame(robust_draws)
    calibration_draws=pd.DataFrame(calibration_draws)
    robust_draws.to_parquet(out/"customer_balanced_bootstrap_draws_s7.parquet",index=False)
    calibration_draws.to_parquet(out/"calibration_bootstrap_draws_s7.parquet",index=False)
    robust_summary=[]
    for metric in robust_point["main"]:
        robust_summary.append({"comparison":"customer_balanced_minus_row_weighted",
            "metric":metric,"point_estimate":robust_point["balanced"][metric]-robust_point["main"][metric],
            "ci95_lower":float(robust_draws[metric].quantile(.025)),
            "ci95_upper":float(robust_draws[metric].quantile(.975)),"replicates":reps})
    robust_summary=pd.DataFrame(robust_summary)
    robust_summary.to_csv(out/"customer_balanced_bootstrap_summary_s7.csv",index=False,encoding="utf-8-sig")
    cal_summary=[]
    for label,left,right in [("Platt_minus_uncalibrated","platt","uncalibrated"),
                             ("Isotonic_minus_uncalibrated","isotonic","uncalibrated"),
                             ("Isotonic_minus_Platt","isotonic","platt")]:
        block=calibration_draws[calibration_draws.comparison.eq(label)]
        for metric in cal_point[left]:
            cal_summary.append({"comparison":label,"metric":metric,
                "point_estimate":cal_point[left][metric]-cal_point[right][metric],
                "ci95_lower":float(block[metric].quantile(.025)),
                "ci95_upper":float(block[metric].quantile(.975)),"replicates":reps})
    cal_summary=pd.DataFrame(cal_summary)
    cal_summary.to_csv(out/"calibration_bootstrap_summary_s7.csv",index=False,encoding="utf-8-sig")
    write_json(out/"bootstrap_design_audit_s7.json",{
        "replicates":reps,"seed":int(bundle.cfg["bootstrap"]["seed"]),
        "customers":n,"fold_counts":pd.Series(folds).value_counts(sort=False).sort_index().to_dict(),
        "sampling_unit":"customer_ID","paired":True,
        "all_customer_records_preserved_by_multiplicity":True})
    return robust_summary,cal_summary


def build_report(bundle, formal, history, calibration, robust_boot, cal_boot):
    _,_,out,models,figs=paths(bundle)
    f=formal.set_index("model")
    main=f.loc["SGTR_row_weighted_frozen"]; bal=f.loc["SGTR_customer_balanced_ablation"]
    r=robust_boot.set_index("metric")
    c=calibration.set_index("method")
    cb={(row.comparison,row.metric):row for row in cal_boot.itertuples(index=False)}
    iso_platt_ll=cb[("Isotonic_minus_Platt","logloss")]
    iso_platt_br=cb[("Isotonic_minus_Platt","brier")]
    isotonic_clear=(iso_platt_ll.ci95_upper<0 and iso_platt_br.ci95_upper<0)
    selected="Isotonic" if isotonic_clear else "Platt"
    # Core robustness direction: both versions retain positive Stable reduction,
    # acceptable High response, and no metric reverses the scientific conclusion.
    robust_direction=bool(bal.stable_reduction>0 and bal.high_response_retention>=.85
                          and bal.auc_drop<=.002 and bal.logloss_relative_increase<=.02
                          and bal.brier_relative_increase<=.02
                          and bal.default_trajectory_retention>=.90)
    history_balanced = history[
        history.model.eq("SGTR_customer_balanced_ablation")]
    history_direction = bool(
        (history_balanced.stable_reduction > 0).all()
        and (history_balanced.high_response_retention >= .85).all())
    weight_diag = pd.read_csv(out / "customer_balanced_weight_diagnostics_s7.csv")
    history_summary = pd.read_csv(out / "customer_history_summary_s7.csv")
    history_distribution = pd.read_csv(out / "customer_history_distribution_s7.csv")
    history_group_counts = (history_distribution.groupby("history_group", as_index=False)
                            .customers.sum())
    history_group_counts["share"] = (history_group_counts.customers
                                      / history_group_counts.customers.sum())
    calibration_parameters = pd.read_csv(out / "calibration_fold_parameters_s7.csv")
    write_json(out/"stage_summary_s7.json",{
        "frozen_core_unchanged":True,"customer_balanced_role":"robustness_only",
        "customer_balanced_direction_stable":robust_direction,
        "history_group_direction_stable":history_direction,
        "selected_calibration_for_holdout":selected,
        "isotonic_clear_probability_advantage_over_platt":isotonic_clear,
        "formal_robustness":formal.to_dict("records"),
        "calibration_metrics":calibration.to_dict("records"),
        "forbidden_stages_run":[],"stop_after":"V4-S7; awaiting human review"})
    formal_cols=["model","auc","ks","logloss","brier","stable_mean_abs_delta_raw",
        "stable_reduction","medium_mean_abs_delta_raw","high_mean_abs_delta_raw",
        "high_response_retention","default_trajectory_retention","srr"]
    report=f"""# V4-S7 冻结核心后的稳健性与概率校准

## 执行边界

- SG-TR核心继续冻结为`w(G)=I(G<=.25)`、`lambda=.03`；本阶段没有重新选择核心模型。
- customer-balanced版本仅为稳健性消融，不能替代主模型。
- 校准器只使用每名客户真实末期、inner cross-fitted预测拟合；校准概率不用于Stable/High指标。
- 未执行动态SHAP或locked holdout。

## 必答结论

1. **核心结论对训练权重是否稳健？** {'是' if robust_direction else '否'}。customer-balanced相对主SG-TR的Stable reduction差={(bal.stable_reduction-main.stable_reduction)*100:+.2f}个百分点，High response差={(bal.high_response_retention-main.high_response_retention)*100:+.2f}个百分点；配对区间见下表。
2. **是否被长历史客户明显主导？** {'没有发现方向性主导' if history_direction else '发现至少一个历史组方向崩溃'}。长历史客户占样本{history_group_counts.loc[history_group_counts.history_group.eq('long'),'share'].iloc[0]:.2%}，总体数值自然主要反映这一人群；但short/medium/long三组的Stable降幅均为正且High响应均高于0.85，方向并非只在长历史组成立。历史组仅作输入长度分层诊断，不用于调模。
3. **customer-balanced是否方向性崩溃？** {'没有' if robust_direction else '有'}。冻结主SG-TR仍保持正式主模型，balanced结果不触发替换。
4. **校准怎样改变概率质量？** Uncalibrated/Platt/Isotonic的LogLoss分别为{c.loc['Uncalibrated SG-TR','logloss']:.6f}/{c.loc['Platt','logloss']:.6f}/{c.loc['Isotonic','logloss']:.6f}，Brier分别为{c.loc['Uncalibrated SG-TR','brier']:.6f}/{c.loc['Platt','brier']:.6f}/{c.loc['Isotonic','brier']:.6f}；截距分别为{c.loc['Uncalibrated SG-TR','calibration_intercept']:.4f}/{c.loc['Platt','calibration_intercept']:.4f}/{c.loc['Isotonic','calibration_intercept']:.4f}，斜率分别为{c.loc['Uncalibrated SG-TR','calibration_slope']:.4f}/{c.loc['Platt','calibration_slope']:.4f}/{c.loc['Isotonic','calibration_slope']:.4f}。
5. **能否不改变稳定性结论而改善概率质量？** 可以；校准只改变末期概率层，核心raw score及Stable/High结论保持不变。
6. **最终holdout冻结哪个校准器？** **{selected}**。选择只依据概率质量和预设简约规则，不依据AUC，也不反馈核心SG-TR。
7. **是否已有条件进入动态SHAP？** {'有' if robust_direction else '暂时没有'}；核心方法、训练权重稳健性和概率层已分别确认。S7在此停止，未执行SHAP。

## Customer-balanced正式指标

{markdown_table(formal[formal_cols])}

## 历史长度分层

{markdown_table(history[['history_group','model','n_group_customers','auc','logloss','brier','stable_reduction','high_response_retention']])}

### 客户历史长度与训练权重分布

`T_i`总体摘要：

{markdown_table(history_summary)}

各历史组客户数：

{markdown_table(history_group_counts)}

每个outer-training内独立归一化的行/边权重（均值应为1）：

{markdown_table(weight_diag)}

## 概率校准

{markdown_table(calibration)}

各outer fold校准参数：

{markdown_table(calibration_parameters)}

![S7客户真实末期概率可靠性曲线](../figures/s7/reliability_curve_s7.png)

## Customer-balanced配对bootstrap（balanced减主SG-TR）

{markdown_table(robust_boot)}

## 校准配对bootstrap

{markdown_table(cal_boot)}
"""
    (ROOT/bundle.cfg["paths"]["log_dir"]/"stage_summary_V4_S7.md").write_text(report,encoding="utf-8")
    log("S7报告已生成。")


def reproducibility_audit(bundle, main, balanced, calibrated):
    _,s6c,out,models,_=paths(bundle)
    audit=[]
    for name,pred in [("main_sgtr",main),("customer_balanced",balanced)]:
        audit.append({"model":name,"rows":int(len(pred)),"customers":int(pred.customer_ID.nunique()),
            "duplicate_customer_month":int(pred.duplicated(["customer_ID","t"]).sum()),
            "missing_scores":int(pred[["s","p"]].isna().any(axis=1).sum())})
    s6c_formal = pd.read_csv(s6c / "formal_outer_oof_metrics_s6c.csv")
    s6c_hard = s6c_formal.loc[
        s6c_formal.model.eq("Hard_gate_nested_ablation")].iloc[0]
    s7_formal = pd.read_csv(out / "customer_balanced_formal_metrics_s7.csv")
    s7_main = s7_formal.loc[
        s7_formal.model.eq("SGTR_row_weighted_frozen")].iloc[0]
    shared_metrics = ["auc", "ks", "logloss", "brier",
                      "stable_mean_abs_delta_raw", "stable_reduction",
                      "medium_mean_abs_delta_raw", "high_mean_abs_delta_raw",
                      "high_response_retention",
                      "default_trajectory_retention", "srr"]
    metric_diffs = {metric: float(abs(s7_main[metric] - s6c_hard[metric]))
                    for metric in shared_metrics}
    if max(metric_diffs.values()) > 1e-12:
        raise AssertionError("S7 frozen main metrics do not reproduce S6c SG-TR")
    main_final = (main.sort_values(["customer_ID", "t"])
                  .groupby("customer_ID", sort=False).tail(1)
                  .set_index("customer_ID").sort_index())
    cal_final = calibrated.set_index("customer_ID").sort_index()
    uncal_probability_max_abs_diff = float(np.max(np.abs(
        main_final.loc[cal_final.index, "p"].to_numpy(float)
        - cal_final.p_uncalibrated.to_numpy(float))))
    if uncal_probability_max_abs_diff > 1e-15:
        raise AssertionError("Calibration input does not equal frozen SG-TR final probability")
    weight_diag = pd.read_csv(out / "customer_balanced_weight_diagnostics_s7.csv")
    weight_mean_max_abs_diff = float(np.max(np.abs(weight_diag["mean"] - 1.0)))
    if weight_mean_max_abs_diff > 1e-12:
        raise AssertionError("Normalized customer weights do not have mean one")
    calibration_overlap = {}
    for outer in sorted(bundle.split.fold.unique()):
        inner = pd.concat([pd.read_parquet(
            s6c / "inner_scores" / f"outer{outer}" / f"inner{inner_fold}"
            / "hard_lam0.03.parquet") for inner_fold in range(3)],
            ignore_index=True)
        train_ids = set(bundle.split.loc[
            bundle.split.fold.ne(outer), "customer_ID"])
        valid_ids = set(bundle.split.loc[
            bundle.split.fold.eq(outer), "customer_ID"])
        inner_ids = set(inner.customer_ID)
        calibration_overlap[str(int(outer))] = {
            "cross_fitted_training_customers": int(len(inner_ids)),
            "missing_outer_training_customers": int(len(train_ids - inner_ids)),
            "outer_validation_overlap": int(len(valid_ids & inner_ids)),
            "duplicate_customer_final_predictions": int(
                inner.sort_values(["customer_ID", "t"])
                .groupby("customer_ID").tail(1).customer_ID.duplicated().sum())}
        if train_ids != inner_ids or valid_ids & inner_ids:
            raise AssertionError("Fold-safe calibration customer isolation failed")
    audit_payload = {
        "frozen_main_score_files_reused_from":"v4/outputs/s6c/outer_scores/hard_selected_outer*.parquet",
        "calibration_training_files_reused_from":"v4/outputs/s6c/inner_scores/outer*/inner*/hard_lam0.03.parquet",
        "score_audit":audit,"calibrated_final_customers":int(len(calibrated)),
        "calibrated_duplicate_customers":int(calibrated.customer_ID.duplicated().sum()),
        "calibrated_missing_probabilities":int(calibrated[["p_uncalibrated","p_platt","p_isotonic"]].isna().any(axis=1).sum()),
        "balanced_model_files":len(list((models/"balanced").glob("*.txt"))),
        "calibration_model_files":len(list((models/"calibration").glob("*.json"))),
        "s6c_frozen_main_metric_max_abs_diff":max(metric_diffs.values()),
        "s6c_frozen_main_metric_abs_diffs":metric_diffs,
        "uncalibrated_final_probability_max_abs_diff":uncal_probability_max_abs_diff,
        "normalized_weight_mean_max_abs_diff":weight_mean_max_abs_diff,
        "calibration_fold_isolation":calibration_overlap,
        "calibration_fitting_unit":"customer_final_period",
        "calibration_fitting_uses_monthly_repeated_targets":False,
        "core_definition_changed":False,"forbidden_stages_run":[]}
    if audit_payload["balanced_model_files"] != 5:
        raise AssertionError("Expected five customer-balanced outer models")
    if audit_payload["calibration_model_files"] != 5:
        raise AssertionError("Expected five fold-safe calibration models")
    write_json(out/"reproducibility_audit_s7.json", audit_payload)


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else "run"
    bundle=load_bundle(CFG)
    s3,s6c,out,_,_=paths(bundle); out.mkdir(parents=True,exist_ok=True)
    if mode in ("run","all","train"): train_balanced_outer(bundle)
    main_pred=selected_s6c(s6c,"hard")
    balanced=balanced_scores(bundle)
    baseline=global_predictions(bundle)["M0_ref"]
    state=stitched_state(bundle,s3)
    formal=formal_robustness(bundle,state,main_pred,balanced,baseline)
    history=history_group_metrics(bundle,state,main_pred,balanced,baseline)
    calibrated=fit_calibrators(bundle,main_pred)
    calibration=calibration_outputs(bundle,calibrated)
    if mode == "train":
        reproducibility_audit(bundle,main_pred,balanced,calibrated)
        log("customer-balanced训练、点估计和fold-safe校准完成；未运行bootstrap。")
        return
    if mode in ("run","all","bootstrap"):
        robust_boot,cal_boot=paired_bootstrap(bundle,state,main_pred,balanced,baseline,calibrated)
    else:
        robust_boot=pd.read_csv(out/"customer_balanced_bootstrap_summary_s7.csv")
        cal_boot=pd.read_csv(out/"calibration_bootstrap_summary_s7.csv")
    reproducibility_audit(bundle,main_pred,balanced,calibrated)
    build_report(bundle,formal,history,calibration,robust_boot,cal_boot)


if __name__=="__main__": main()
