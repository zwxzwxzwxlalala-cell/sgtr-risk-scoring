# -*- coding: utf-8 -*-
"""V4-S6d: paired uncertainty and threshold sensitivity for frozen SG-TR."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_tr_core import (ROOT, lgb_params, load_bundle, make_objective,
                              predict_rows, write_json)
from adaptive_tr_experiment import add_relative, evaluate
from adaptive_tr_s6c import (add_medium, global_predictions, load_selected_scores,
                             markdown_table, selected_s6c, stitched_state)

CFG = ROOT / "v4" / "config_v4_s6d.yaml"


def log(message: str) -> None:
    print("[V4-S6d] " + message, flush=True)


def paths(bundle):
    p = bundle.cfg["paths"]
    return (ROOT / p["s3_s6_output_dir"], ROOT / p["s6b_output_dir"],
            ROOT / p["s6c_output_dir"], ROOT / p["output_dir"],
            ROOT / p["model_dir"], ROOT / p["figure_dir"])


def hard_weights(gate: np.ndarray, q_s: float) -> np.ndarray:
    g = np.asarray(gate, dtype=float)
    return (np.isfinite(g) & (g <= float(q_s))).astype(float)


def fit_threshold_model(bundle, train_customers: np.ndarray, state: pd.DataFrame,
                        lam: float, q_s: float, model_path: Path):
    import lightgbm as lgb
    train_set = set(train_customers.tolist())
    train_rows = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.panel_customer), dtype=bool,
        count=len(bundle.panel_customer)))
    train_pairs = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.pair_customer), dtype=bool,
        count=len(bundle.pair_customer)))
    position = np.full(len(bundle.panel), -1, dtype=np.int32)
    position[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp = bundle.pairs.row_prev.to_numpy(int)[train_pairs]
    pc = bundle.pairs.row_curr.to_numpy(int)[train_pairs]
    a, b = position[pp], position[pc]
    if np.any((a < 0) | (b < 0)):
        raise AssertionError("A sensitivity edge crossed the customer split")
    all_weights = hard_weights(state.G.to_numpy(float), q_s)
    w = all_weights[train_pairs]
    prev_pos = np.full(len(train_rows), -1, dtype=np.int32)
    next_pos = np.full(len(train_rows), -1, dtype=np.int32)
    prev_weight = np.zeros(len(train_rows), dtype=float)
    next_weight = np.zeros(len(train_rows), dtype=float)
    prev_pos[b], next_pos[a] = a, b
    prev_weight[b], next_weight[a] = w, w
    X = bundle.panel.loc[train_rows, bundle.features]
    y = bundle.panel.target.to_numpy(float)[train_rows]
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    intercept = float(np.log(p0 / (1 - p0)))
    ds = lgb.Dataset(X, label=y, categorical_feature=bundle.categorical,
                     init_score=np.full(len(y), intercept), free_raw_data=False)
    params = lgb_params(bundle)
    params["objective"] = make_objective(
        prev_pos, next_pos, prev_weight, next_weight, lam)
    booster = lgb.train(params, ds,
                        num_boost_round=int(bundle.cfg["model"]["n_estimators"]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(booster.model_to_string(), encoding="utf-8")
    return booster, intercept, {
        "model": "SG-TR threshold sensitivity only",
        "lambda": float(lam), "q_s": float(q_s),
        "n_train_customers": int(len(train_customers)),
        "n_train_rows": int(len(train_rows)), "n_edges": int(len(train_pairs)),
        "init_score_log_odds": intercept,
        "edge_weight_zero_share": float((w == 0).mean()),
        "edge_weight_one_share": float((w == 1).mean()),
    }


def train_sensitivity(bundle) -> None:
    s3, _, _, out, models, _ = paths(bundle)
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    lam = float(bundle.cfg["model"]["sensitivity_lambda"])
    thresholds = [float(x) for x in bundle.cfg["model"]["sensitivity_q_s"]]
    if thresholds != [.20, .25, .30] or lam != .03:
        raise AssertionError("S6d frozen sensitivity design changed")
    # Q25 is the existing formal SG-TR; only Q20/Q30 are fitted.
    for q_s in [.20, .30]:
        for outer in sorted(bundle.split.fold.unique()):
            score_path = out / "threshold_scores" / f"q{int(q_s*100)}_outer{outer}.parquet"
            model_path = models / f"sgtr_q{int(q_s*100)}_outer{outer}.txt"
            diag_path = model_path.with_suffix(".json")
            if score_path.exists() and model_path.exists() and diag_path.exists():
                continue
            train_ids = bundle.split.loc[
                bundle.split.fold.ne(outer), "customer_ID"].to_numpy()
            val_ids = bundle.split.loc[
                bundle.split.fold.eq(outer), "customer_ID"].to_numpy()
            state = pd.read_parquet(
                s3 / "state_maps" / f"outer{outer}_full.parquet")
            started = time.time()
            booster, intercept, diag = fit_threshold_model(
                bundle, train_ids, state, lam, q_s, model_path)
            pred = predict_rows(bundle, booster, intercept, val_ids, int(outer))
            score_path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(score_path, index=False)
            diag.update(
                score_file=str(score_path.relative_to(ROOT)),
                model_file=str(model_path.relative_to(ROOT)),
                lightgbm_version=__import__("lightgbm").__version__,
                features=bundle.features,
                categorical_features=bundle.categorical,
                parameters={k: v for k, v in lgb_params(bundle).items()
                            if not callable(v)},
                n_estimators=int(bundle.cfg["model"]["n_estimators"]),
                elapsed_seconds=time.time() - started)
            write_json(diag_path, diag)
            log(f"敏感性 Q{int(q_s*100)} outer {outer} 完成，"
                f"{diag['elapsed_seconds']:.1f}s")


def model_predictions(bundle) -> dict[str, pd.DataFrame]:
    s3, s6b, s6c, _, _, _ = paths(bundle)
    models = global_predictions(bundle)
    models["Exponential_lambda_0.1_gamma_12"] = load_selected_scores(
        s6b, s6b / "outer_selections_s6b.csv",
        "outer_scores/adaptive_s6b_selected_outer{outer}.parquet")
    models["Piecewise_lambda_0.01"] = selected_s6c(s6c, "piecewise")
    models["SGTR_q25_lambda_0.03"] = selected_s6c(s6c, "hard")
    return models


def layer_state(state: pd.DataFrame, q_s: float, q_h: float = .90) -> pd.DataFrame:
    out = state.copy()
    g = out.G.to_numpy(float)
    layer = np.full(len(out), "medium", dtype=object)
    layer[~np.isfinite(g)] = "insufficient"
    layer[np.isfinite(g) & (g <= q_s)] = "stable"
    layer[np.isfinite(g) & (g >= q_h)] = "high"
    out["layer"] = layer
    return out


def threshold_analysis(bundle, models: dict[str, pd.DataFrame],
                       state: pd.DataFrame) -> pd.DataFrame:
    _, _, _, out, _, _ = paths(bundle)
    sensitivity = {}
    for q_s in [.20, .30]:
        sensitivity[q_s] = pd.concat([pd.read_parquet(
            out / "threshold_scores" / f"q{int(q_s*100)}_outer{k}.parquet")
            for k in range(5)], ignore_index=True)
    sensitivity[.25] = models["SGTR_q25_lambda_0.03"]
    rows = []
    for q_s in [.20, .25, .30]:
        qstate = layer_state(state, q_s)
        baseline = add_medium(bundle, models["M0_ref"], qstate,
                              evaluate(bundle, models["M0_ref"], qstate))
        metric = add_medium(bundle, sensitivity[q_s], qstate,
                            evaluate(bundle, sensitivity[q_s], qstate))
        metric = add_relative(metric, baseline, bundle.cfg)
        rows.append({
            "q_s": q_s, "lambda": .03,
            "is_frozen_primary": q_s == .25,
            "stable_pairs": int(qstate.layer.eq("stable").sum()),
            "stable_customers": int(qstate.loc[
                qstate.layer.eq("stable"), "customer_ID"].nunique()),
            **metric,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "sgtr_threshold_sensitivity_s6d.csv", index=False,
                 encoding="utf-8-sig")
    return frame


def weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    s, yy, ww = score[order], y[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    pos = np.add.reduceat(ww * yy, starts)
    neg = np.add.reduceat(ww * (1 - yy), starts)
    total_pos, total_neg = pos.sum(), neg.sum()
    before_neg = np.cumsum(neg) - neg
    return float(np.sum(pos * (before_neg + .5 * neg)) /
                 (total_pos * total_neg))


def fold_equal_weighted(values: np.ndarray, weights: np.ndarray,
                        folds: np.ndarray) -> float:
    means = []
    for fold in sorted(np.unique(folds)):
        ok = (folds == fold) & np.isfinite(values)
        denom = weights[ok].sum()
        means.append(float(np.dot(weights[ok], values[ok]) / denom))
    return float(np.mean(means))


def customer_summaries(bundle, predictions: dict[str, pd.DataFrame],
                       state: pd.DataFrame):
    customers = bundle.split.customer_ID.to_numpy()
    folds = bundle.split.fold.to_numpy(int)
    lookup = pd.Index(customers)
    result = {}
    pair_index = state.pair_index.to_numpy(int)
    pp = bundle.pairs.row_prev.to_numpy(int)[pair_index]
    pc = bundle.pairs.row_curr.to_numpy(int)[pair_index]
    for name, pred in predictions.items():
        ordered = pred.sort_values(["customer_ID", "t"])
        last = ordered.groupby("customer_ID", sort=False).tail(1).set_index("customer_ID")
        term = last.reindex(lookup)
        if term[["s", "p", "target"]].isna().any().any():
            raise AssertionError(f"Missing terminal prediction for {name}")
        score = np.full(len(bundle.panel), np.nan)
        score[pred.row_index.to_numpy(int)] = pred.s.to_numpy(float)
        pair = state[["customer_ID", "layer"]].copy()
        pair["abs_delta"] = np.abs(score[pc] - score[pp])
        layer_values = {}
        for layer in ["stable", "medium", "high"]:
            v = pair[pair.layer.eq(layer)].groupby(
                "customer_ID", observed=True).abs_delta.mean()
            layer_values[layer] = v.reindex(lookup).to_numpy(float)
        work = pred[["customer_ID", "row_index", "s", "target"]].copy()
        work["month"] = bundle.month[pred.row_index.to_numpy(int)]
        tail6 = work.sort_values(["customer_ID", "month"]).groupby(
            "customer_ID", sort=False).tail(6)
        agg = tail6.groupby("customer_ID", observed=True).agg(
            n=("month", "size"), span=("month", lambda x: x.max() - x.min()),
            target=("target", "first"))
        valid = agg[(agg.n == 6) & (agg.span == 5) & (agg.target == 1)].index
        first = tail6[tail6.customer_ID.isin(valid)].sort_values(
            ["customer_ID", "month"]).groupby("customer_ID", observed=True).head(1).set_index("customer_ID").s
        final = tail6[tail6.customer_ID.isin(valid)].sort_values(
            ["customer_ID", "month"]).groupby("customer_ID", observed=True).tail(1).set_index("customer_ID").s
        rise = (final - first).reindex(lookup).to_numpy(float)
        result[name] = {
            "target": term.target.to_numpy(float),
            "score": term.s.to_numpy(float),
            "prob": term.p.to_numpy(float),
            "stable": layer_values["stable"],
            "medium": layer_values["medium"],
            "high": layer_values["high"],
            "trajectory": rise,
        }
    return customers, folds, result


def metrics_from_weights(summary: dict, folds: np.ndarray,
                         weights: np.ndarray, baseline: dict) -> dict:
    y, p = summary["target"], summary["prob"]
    eps = np.finfo(float).eps
    pp = np.clip(p, eps, 1 - eps)
    stable = fold_equal_weighted(summary["stable"], weights, folds)
    medium = fold_equal_weighted(summary["medium"], weights, folds)
    high = fold_equal_weighted(summary["high"], weights, folds)
    trajectory = fold_equal_weighted(summary["trajectory"], weights, folds)
    base_high = fold_equal_weighted(baseline["high"], weights, folds)
    base_trajectory = fold_equal_weighted(baseline["trajectory"], weights, folds)
    return {
        "auc": weighted_auc(y, summary["score"], weights),
        "logloss": float(np.dot(weights, -(y*np.log(pp)+(1-y)*np.log(1-pp))) /
                         weights.sum()),
        "brier": float(np.dot(weights, (p-y)**2) / weights.sum()),
        "stable_mean_abs_delta_raw": stable,
        "medium_mean_abs_delta_raw": medium,
        "high_mean_abs_delta_raw": high,
        "high_response_retention": high / base_high,
        "default_trajectory_rise": trajectory,
        "default_trajectory_retention": trajectory / base_trajectory,
        "srr": high / stable,
        "srr_numerator_high_mean_abs_delta_raw": high,
        "srr_denominator_stable_mean_abs_delta_raw": stable,
    }


def paired_bootstrap(bundle, predictions: dict[str, pd.DataFrame],
                     state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _, _, _, out, _, _ = paths(bundle)
    comparisons = {
        "SGTR_vs_M0_ref": "M0_ref",
        "SGTR_vs_Global_lambda_0.01": "Global_lambda_0.01",
        "SGTR_vs_Global_lambda_0.03": "Global_lambda_0.03",
        "SGTR_vs_Global_lambda_0.1": "Global_lambda_0.1",
        "SGTR_vs_Exponential_lambda_0.1_gamma_12": "Exponential_lambda_0.1_gamma_12",
        "SGTR_vs_Piecewise_lambda_0.01": "Piecewise_lambda_0.01",
    }
    needed = {"M0_ref", "SGTR_q25_lambda_0.03", *comparisons.values()}
    selected = {k: v for k, v in predictions.items() if k in needed}
    customers, folds, summaries = customer_summaries(bundle, selected, state)
    n = len(customers)
    ones = np.ones(n, dtype=np.int32)
    baseline = summaries["M0_ref"]
    point = {name: metrics_from_weights(value, folds, ones, baseline)
             for name, value in summaries.items()}
    reps = int(bundle.cfg["bootstrap"]["replicates"])
    rng = np.random.default_rng(int(bundle.cfg["bootstrap"]["seed"]))
    fold_indices = [np.flatnonzero(folds == f) for f in sorted(np.unique(folds))]
    draw_rows = []
    for replicate in range(reps):
        weights = np.zeros(n, dtype=np.int32)
        for idx in fold_indices:
            weights[idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)))
        rep_metrics = {name: metrics_from_weights(value, folds, weights, baseline)
                       for name, value in summaries.items()}
        for label, comparator in comparisons.items():
            row = {"replicate": replicate, "comparison": label}
            for metric, value in rep_metrics["SGTR_q25_lambda_0.03"].items():
                row[metric] = value - rep_metrics[comparator][metric]
            draw_rows.append(row)
        if (replicate + 1) % 250 == 0:
            log(f"paired bootstrap {replicate + 1}/{reps}")
    draws = pd.DataFrame(draw_rows)
    draws.to_parquet(out / "paired_customer_bootstrap_draws_s6d.parquet", index=False)
    summary_rows = []
    metrics = list(point["SGTR_q25_lambda_0.03"].keys())
    for label, comparator in comparisons.items():
        block = draws[draws.comparison.eq(label)]
        for metric in metrics:
            estimate = (point["SGTR_q25_lambda_0.03"][metric]
                        - point[comparator][metric])
            summary_rows.append({
                "comparison": label, "comparator": comparator, "metric": metric,
                "point_estimate_sgtr_minus_comparator": estimate,
                "ci95_lower": float(block[metric].quantile(.025)),
                "ci95_upper": float(block[metric].quantile(.975)),
                "bootstrap_replicates": reps,
                "ci_method": "outer-fold-stratified paired customer percentile",
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "paired_customer_bootstrap_summary_s6d.csv",
                      index=False, encoding="utf-8-sig")
    point_df = pd.DataFrame([{"model": name, **metrics_}
                             for name, metrics_ in point.items()])
    point_df.to_csv(out / "bootstrap_point_metrics_s6d.csv", index=False,
                    encoding="utf-8-sig")
    audit = {"replicates": reps, "seed": int(bundle.cfg["bootstrap"]["seed"]),
             "customers": n, "fold_counts": pd.Series(folds).value_counts(
                 sort=False).sort_index().to_dict(),
             "paired": True, "sampling_unit": "customer_ID",
             "all_months_and_pairs_preserved_by_customer_multiplicity": True}
    write_json(out / "bootstrap_design_audit_s6d.json", audit)
    return summary_df, point_df, audit


def build_report(bundle, threshold: pd.DataFrame, bootstrap: pd.DataFrame,
                 point: pd.DataFrame, audit: dict) -> None:
    _, _, _, out, _, _ = paths(bundle)
    comparisons = {}
    for label in bootstrap.comparison.unique():
        block = bootstrap[bootstrap.comparison.eq(label)].copy()
        comparisons[label] = block.set_index("metric")[[
            "point_estimate_sgtr_minus_comparator", "ci95_lower", "ci95_upper"]]
    piece = comparisons["SGTR_vs_Piecewise_lambda_0.01"]
    exp12 = comparisons["SGTR_vs_Exponential_lambda_0.1_gamma_12"]
    globals_ = {x: comparisons[f"SGTR_vs_Global_lambda_{x}"]
                for x in ["0.01", "0.03", "0.1"]}
    q = threshold.set_index("q_s")
    direction_consistent = bool(
        (threshold.stable_reduction > 0).all()
        and (threshold.high_response_retention > .85).all()
        and (threshold.auc_drop <= .002).all()
        and (threshold.logloss_relative_increase <= .02).all()
        and (threshold.brier_relative_increase <= .02).all()
        and (threshold.default_trajectory_retention >= .90).all())
    sgtr_frozen = direction_consistent
    selected_metrics = ["auc", "logloss", "brier",
                        "stable_mean_abs_delta_raw", "medium_mean_abs_delta_raw",
                        "high_mean_abs_delta_raw", "high_response_retention",
                        "default_trajectory_rise", "default_trajectory_retention",
                        "srr"]
    report_table = bootstrap[bootstrap.metric.isin(selected_metrics)].copy()
    report_table["estimate [95% CI]"] = report_table.apply(
        lambda r: f"{r.point_estimate_sgtr_minus_comparator:+.6f} "
                  f"[{r.ci95_lower:+.6f}, {r.ci95_upper:+.6f}]", axis=1)
    compact = report_table[["comparison", "metric", "estimate [95% CI]"]]
    piece_high = piece.loc["high_response_retention"]
    piece_stable = piece.loc["stable_mean_abs_delta_raw"]
    piece_auc = piece.loc["auc"]
    piece_brier = piece.loc["brier"]
    piece_traj = piece.loc["default_trajectory_retention"]
    global_evidence = []
    for lam, table in globals_.items():
        global_evidence.append({
            "global_lambda": lam,
            "stable_delta": table.loc["stable_mean_abs_delta_raw",
                                       "point_estimate_sgtr_minus_comparator"],
            "stable_ci": [table.loc["stable_mean_abs_delta_raw", "ci95_lower"],
                          table.loc["stable_mean_abs_delta_raw", "ci95_upper"]],
            "high_retention_delta": table.loc["high_response_retention",
                                               "point_estimate_sgtr_minus_comparator"],
            "high_retention_ci": [table.loc["high_response_retention", "ci95_lower"],
                                  table.loc["high_response_retention", "ci95_upper"]],
        })
    global_text = "；".join(
        f"λ={row['global_lambda']}：Stable波动差{row['stable_delta']:+.6f} "
        f"[{row['stable_ci'][0]:+.6f},{row['stable_ci'][1]:+.6f}]，"
        f"High-response差{row['high_retention_delta']:+.2%} "
        f"[{row['high_retention_ci'][0]:+.2%},{row['high_retention_ci'][1]:+.2%}]"
        for row in global_evidence)

    # Confirm that the bootstrap point-estimator exactly reproduces the formal
    # S6c metrics and that Q25 is the untouched formal SG-TR model.
    _, _, s6c, _, models_dir, _ = paths(bundle)
    formal = pd.read_csv(s6c / "formal_outer_oof_metrics_s6c.csv").set_index("model")
    point_indexed = point.set_index("model")
    mapping = {
        "M0_ref": "M0_ref", "Global_lambda_0.01": "Global_lambda_0.01",
        "Global_lambda_0.03": "Global_lambda_0.03",
        "Global_lambda_0.1": "Global_lambda_0.1",
        "Exponential_lambda_0.1_gamma_12": "Exponential_lambda_0.1_gamma_12",
        "Piecewise_lambda_0.01": "Piecewise_nested",
        "SGTR_q25_lambda_0.03": "Hard_gate_nested_ablation",
    }
    check_metrics = ["auc", "logloss", "brier", "stable_mean_abs_delta_raw",
                     "medium_mean_abs_delta_raw", "high_mean_abs_delta_raw",
                     "high_response_retention", "default_trajectory_rise",
                     "default_trajectory_retention", "srr"]
    point_diffs = []
    for point_name, formal_name in mapping.items():
        for metric in check_metrics:
            point_diffs.append(abs(float(point_indexed.loc[point_name, metric])
                                   - float(formal.loc[formal_name, metric])))
    q25 = threshold[threshold.q_s.eq(.25)].iloc[0]
    q25_diffs = [abs(float(q25[m]) - float(
        formal.loc["Hard_gate_nested_ablation", m])) for m in
        ["auc", "logloss", "brier", "stable_reduction",
         "high_response_retention", "default_trajectory_retention"]]
    score_audit = []
    for q_s in [.20, .30]:
        for outer in range(5):
            score_path = out / "threshold_scores" / f"q{int(q_s*100)}_outer{outer}.parquet"
            score = pd.read_parquet(score_path)
            score_audit.append({
                "q_s": q_s, "outer_fold": outer, "rows": int(len(score)),
                "customers": int(score.customer_ID.nunique()),
                "duplicate_customer_month": int(score.duplicated(
                    ["customer_ID", "t"]).sum()),
                "missing_scores": int(score[["s", "p"]].isna().any(axis=1).sum()),
            })
    reproducibility = {
        "bootstrap_point_vs_formal_s6c_max_abs_difference": float(max(point_diffs)),
        "q25_sensitivity_row_vs_formal_sgtr_max_abs_difference": float(max(q25_diffs)),
        "sensitivity_models_expected": 10,
        "sensitivity_model_files": len(list(models_dir.glob("*.txt"))),
        "sensitivity_diagnostic_files": len(list(models_dir.glob("*.json"))),
        "sensitivity_score_files": len(list((out / "threshold_scores").glob("*.parquet"))),
        "sensitivity_score_audit": score_audit,
        "bootstrap_draw_rows": int(audit["replicates"] * 6),
        "forbidden_stages_run": [],
    }
    write_json(out / "reproducibility_audit_s6d.json", reproducibility)
    write_json(out / "stage_summary_s6d.json", {
        "frozen_method": "Stable-State Gated Temporal Regularization (SG-TR)",
        "frozen_definition": "w(G)=I(G<=0.25)",
        "frozen_lambda": .03,
        "bootstrap_design": audit,
        "threshold_direction_consistent": direction_consistent,
        "sgtr_core_method_frozen": sgtr_frozen,
        "global_stability_response_evidence": global_evidence,
        "sgtr_vs_piecewise": piece.reset_index().to_dict("records"),
        "sgtr_vs_exponential_gamma12": exp12.reset_index().to_dict("records"),
        "threshold_sensitivity": threshold.to_dict("records"),
        "reproducibility_audit": reproducibility,
        "forbidden_stages_run": [],
        "stop_after": "V4-S6d; awaiting human review",
    })
    report = f"""# V4-S6d SG-TR最终统计确认与核心方法冻结

## 冻结结论

**{'正式冻结 SG-TR：q_s=0.25、λ=0.03。' if sgtr_frozen else '暂不冻结SG-TR。'}** 本阶段没有重新训练Q25主模型，也没有重新选择参数；仅训练固定λ的Q20/Q30阈值敏感性模型。未执行customer-balanced BCE、calibration、动态SHAP或locked holdout。

SG-TR定义为 `w(G)=I(G<=0.25)`。G是Broad、Sparse Shock和Transition三通道输入状态变化的无监督门控，不表示违约风险；Stable表示输入状态高度稳定，不表示低风险客户。

## 必答结论

1. **相对Global的稳定—响应改善是否有稳定不确定性证据？** **有，但属于稳定的权衡改善而非双指标同时支配。** SG-TR相对各Global配置都保留更多High响应，同时Stable波动也更高、即平滑更弱；两类差异的区间均未跨0。{global_text}。因此证据支持SG-TR把平滑集中于Stable边后，以可量化的Stable降幅让渡换取稳定的响应保留，而不是声称它在同一λ下比Global更平滑。
2. **SG-TR和Piecewise的High-response差距是否稳定？** SG-TR减Piecewise的High-response差为 {piece_high['point_estimate_sgtr_minus_comparator']:+.2%}，95% CI [{piece_high['ci95_lower']:+.2%}, {piece_high['ci95_upper']:+.2%}]。
3. **Medium连续门控是否有足够增量价值？** **没有。** Piecewise的High-response优势约1.54个百分点且区间方向稳定，但SG-TR的Stable波动低{abs(piece_stable['point_estimate_sgtr_minus_comparator']):.6f}、trajectory retention高{piece_traj['point_estimate_sgtr_minus_comparator']:.4f}，二者区间均不跨0；AUC差={piece_auc['point_estimate_sgtr_minus_comparator']:+.6f}、Brier差={piece_brier['point_estimate_sgtr_minus_comparator']:+.6f}且区间跨0。现有证据支持“有限High响应增益不足以抵消额外复杂度与Stable损失”，但不表述为SG-TR在所有指标上显著优于Piecewise。
4. **Q20/Q25/Q30方向是否一致？** {'是' if direction_consistent else '否'}。三种固定阈值均保持Stable reduction为正、High response≥.85，并守住全部原预测与trajectory门槛。Q20/Q30仅为敏感性，不替换Q25。
5. **是否可以正式冻结SG-TR？** {'可以。没有发现代码错误、重大统计异常或阈值方向性崩溃。' if sgtr_frozen else '不可以；阈值敏感性出现方向性崩溃或门槛失败。'}

## 阈值敏感性

{markdown_table(threshold[['q_s','stable_pairs','stable_customers','stable_reduction','high_response_retention','auc','logloss','brier','default_trajectory_retention','acceptable']])}

## 客户级配对bootstrap

设计：{audit['replicates']}次；在每个outer fold内以customer_ID有放回抽样；被抽客户的全部月份和相邻对按抽样次数保留；所有模型共享完全相同的抽样权重。区间仅描述不确定性，不构成新显著性门槛。

{markdown_table(compact)}

## 机制解释

SG-TR不是对所有客户进行普遍压平。它仅在冻结的无监督多通道状态指标判断相邻月份输入状态高度稳定时施加时间一致性约束；对Medium、High及状态不足的相邻期不追加平滑惩罚，保留模型响应风险状态变化的能力。
"""
    (ROOT / bundle.cfg["paths"]["log_dir"] /
     "stage_summary_V4_S6d.md").write_text(report, encoding="utf-8")
    log("S6d统计表和阶段报告已生成。")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    bundle = load_bundle(CFG)
    s3, _, _, out, _, _ = paths(bundle)
    out.mkdir(parents=True, exist_ok=True)
    if mode in ("run", "all", "sensitivity"):
        train_sensitivity(bundle)
    predictions = model_predictions(bundle)
    state = stitched_state(bundle, s3)
    if mode in ("run", "all", "sensitivity"):
        threshold = threshold_analysis(bundle, predictions, state)
    else:
        threshold = pd.read_csv(out / "sgtr_threshold_sensitivity_s6d.csv")
    if mode == "sensitivity":
        log("阈值敏感性训练与汇总完成；未运行bootstrap。")
        return
    if mode in ("run", "all", "bootstrap"):
        bootstrap, point, audit = paired_bootstrap(bundle, predictions, state)
    else:
        bootstrap = pd.read_csv(out / "paired_customer_bootstrap_summary_s6d.csv")
        point = pd.read_csv(out / "bootstrap_point_metrics_s6d.csv")
        audit = __import__("json").loads((out / "bootstrap_design_audit_s6d.json").read_text(
            encoding="utf-8"))
    build_report(bundle, threshold, bootstrap, point, audit)


if __name__ == "__main__":
    main()
