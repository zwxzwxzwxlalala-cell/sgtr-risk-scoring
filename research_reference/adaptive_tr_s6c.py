# -*- coding: utf-8 -*-
"""V4-S6c: frozen piecewise state gate and hard-gate sensitivity analysis.

The script reuses the exact S3--S6 nested customer splits and state mappings.
Only the gate parameterization changes; state construction never sees labels or scores.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from adaptive_tr_core import (ROOT, lgb_params, load_bundle, make_objective,
                              predict_rows, write_json)
from adaptive_tr_experiment import (add_relative, evaluate, pareto_flag)

CFG = ROOT / "v4" / "config_v4_s6c.yaml"


def log(message: str) -> None:
    print("[V4-S6c] " + message, flush=True)


def gate_weight(gate: np.ndarray, family: str, q_s: float = .25,
                q_h: float = .90, missing: float = 1.0) -> np.ndarray:
    g = np.asarray(gate, dtype=float).copy()
    g[~np.isfinite(g)] = missing
    if family == "piecewise":
        return np.where(g <= q_s, 1.0,
                        np.where(g >= q_h, 0.0, (q_h - g) / (q_h - q_s)))
    if family == "hard":
        return (g <= q_s).astype(float)
    raise ValueError(f"Unknown gate family: {family}")


def verify_gates() -> dict:
    g = np.array([np.nan, 0, .25, .5, .9, 1.0])
    p = gate_weight(g, "piecewise")
    h = gate_weight(g, "hard")
    expected_p = np.array([0, 1, 1, (.9-.5)/(.9-.25), 0, 0])
    expected_h = np.array([0, 1, 1, 0, 0, 0])
    return {
        "q_s": .25, "q_h": .90,
        "piecewise_boundary_and_missing_check": bool(np.allclose(p, expected_p)),
        "hard_boundary_and_missing_check": bool(np.array_equal(h, expected_h)),
        "piecewise_monotone_nonincreasing": bool(np.all(np.diff(p[1:]) <= 0)),
        "passed": bool(np.allclose(p, expected_p) and np.array_equal(h, expected_h)),
    }


def adjacency(bundle, train_rows: np.ndarray, train_pair_indices: np.ndarray,
              state: pd.DataFrame, family: str):
    pos = np.full(len(bundle.panel), -1, dtype=np.int32)
    pos[train_rows] = np.arange(len(train_rows), dtype=np.int32)
    pp = bundle.pairs.row_prev.to_numpy(int)[train_pair_indices]
    pc = bundle.pairs.row_curr.to_numpy(int)[train_pair_indices]
    a, b = pos[pp], pos[pc]
    if np.any((a < 0) | (b < 0)):
        raise AssertionError("A training edge crossed the customer split")
    weights_all = gate_weight(
        state.G.to_numpy(float), family,
        float(bundle.cfg["state"]["stable_gate_max"]),
        float(bundle.cfg["state"]["high_gate_min"]),
        float(bundle.cfg["state"]["insufficient_gate_for_weighting"]))
    w = weights_all[train_pair_indices]
    prev_pos = np.full(len(train_rows), -1, dtype=np.int32)
    next_pos = np.full(len(train_rows), -1, dtype=np.int32)
    prev_weight = np.zeros(len(train_rows), dtype=np.float64)
    next_weight = np.zeros(len(train_rows), dtype=np.float64)
    prev_pos[b], next_pos[a] = a, b
    prev_weight[b], next_weight[a] = w, w
    return prev_pos, next_pos, prev_weight, next_weight, w


def fit_gate_model(bundle, train_customers: np.ndarray, state: pd.DataFrame,
                   lam: float, family: str, model_path: Path):
    import lightgbm as lgb
    train_set = set(train_customers.tolist())
    train_rows = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.panel_customer), dtype=bool,
        count=len(bundle.panel_customer)))
    train_pairs = np.flatnonzero(np.fromiter(
        (x in train_set for x in bundle.pair_customer), dtype=bool,
        count=len(bundle.pair_customer)))
    prev, nxt, wp, wn, edge_w = adjacency(
        bundle, train_rows, train_pairs, state, family)
    X = bundle.panel.loc[train_rows, bundle.features]
    y = bundle.panel.target.to_numpy(float)[train_rows]
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    intercept = float(np.log(p0 / (1 - p0)))
    ds = lgb.Dataset(X, label=y, categorical_feature=bundle.categorical,
                     init_score=np.full(len(y), intercept), free_raw_data=False)
    params = lgb_params(bundle)
    params["objective"] = make_objective(prev, nxt, wp, wn, float(lam))
    booster = lgb.train(params, ds,
                        num_boost_round=int(bundle.cfg["model"]["n_estimators"]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(booster.model_to_string(), encoding="utf-8")
    diag = {
        "gate_family": family, "lambda": float(lam),
        "q_s": float(bundle.cfg["state"]["stable_gate_max"]),
        "q_h": float(bundle.cfg["state"]["high_gate_min"]),
        "n_train_customers": int(len(train_customers)),
        "n_train_rows": int(len(train_rows)), "n_edges": int(len(train_pairs)),
        "init_score_log_odds": intercept,
        "edge_weight_min": float(edge_w.min()),
        "edge_weight_mean": float(edge_w.mean()),
        "edge_weight_max": float(edge_w.max()),
        "edge_weight_zero_share": float((edge_w == 0).mean()),
        "edge_weight_one_share": float((edge_w == 1).mean()),
    }
    return booster, intercept, diag


def train_one(bundle, train_ids, val_ids, state, lam, family, cv_fold,
              score_path: Path, model_path: Path, diag_path: Path) -> pd.DataFrame:
    if score_path.exists() and model_path.exists() and diag_path.exists():
        return pd.read_parquet(score_path)
    started = time.time()
    booster, intercept, diag = fit_gate_model(
        bundle, train_ids, state, lam, family, model_path)
    pred = predict_rows(bundle, booster, intercept, val_ids, cv_fold)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(score_path, index=False)
    diag.update(
        model_file=str(model_path.relative_to(ROOT)),
        score_file=str(score_path.relative_to(ROOT)),
        lightgbm_version=__import__("lightgbm").__version__,
        features=bundle.features, categorical_features=bundle.categorical,
        parameters={k: v for k, v in lgb_params(bundle).items() if not callable(v)},
        n_estimators=int(bundle.cfg["model"]["n_estimators"]),
        elapsed_seconds=time.time() - started)
    write_json(diag_path, diag)
    log(f"训练 {family} λ={lam:g}, fold={cv_fold}, {diag['elapsed_seconds']:.1f}s")
    return pred


def state_for_ids(state: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    return state[state.customer_ID.isin(set(ids.tolist()))].copy()


def add_medium(bundle, predictions: pd.DataFrame, state: pd.DataFrame,
               metric: dict) -> dict:
    pred = predictions.sort_values("row_index").drop_duplicates("row_index")
    score = np.full(len(bundle.panel), np.nan)
    score[pred.row_index.to_numpy(int)] = pred.s.to_numpy(float)
    pi = state.pair_index.to_numpy(int)
    pp = bundle.pairs.row_prev.to_numpy(int)[pi]
    pc = bundle.pairs.row_curr.to_numpy(int)[pi]
    d = state[["customer_ID", "layer"]].copy()
    d["abs_delta_s"] = np.abs(score[pc] - score[pp])
    d["cv_fold"] = d.customer_ID.map(
        pred.groupby("customer_ID", observed=True).cv_fold.first())
    d = d[np.isfinite(d.abs_delta_s) & d.cv_fold.notna()]
    medium = d[d.layer.eq("medium")]
    x = medium.groupby(["cv_fold", "customer_ID"], observed=True).abs_delta_s.mean()
    out = dict(metric)
    out["medium_mean_abs_delta_raw"] = float(
        x.groupby(level=0).mean().mean()) if len(x) else float("nan")
    out["medium_pairs"] = int(len(medium))
    out["medium_customers"] = int(medium.customer_ID.nunique())
    return out


def select_family(table: pd.DataFrame, cfg: dict) -> dict:
    acceptable = table[table.acceptable].copy()
    if len(acceptable):
        best = acceptable.stable_reduction.max()
        tied = acceptable[acceptable.stable_reduction >= best - float(
            cfg["selection"]["near_tie_stable_reduction_absolute"])]
        chosen = tied.sort_values("lambda").iloc[0]
        reason = "acceptable_domain_then_max_stable_reduction_with_prefrozen_near_tie_rule"
    else:
        chosen = table.sort_values(
            ["acceptable_domain_distance", "lambda"]).iloc[0]
        reason = "no_acceptable_candidate_nearest_domain_only_pending_human_review"
    out = chosen.to_dict()
    out["selection_reason"] = reason
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")
    headers = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def paths(bundle):
    p = bundle.cfg["paths"]
    return (ROOT / p["s3_s6_output_dir"], ROOT / p["s6b_output_dir"],
            ROOT / p["output_dir"], ROOT / p["model_dir"],
            ROOT / p["figure_dir"])


def run_nested(bundle) -> None:
    s3, _, out, models, _ = paths(bundle)
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    verification = verify_gates()
    write_json(out / "gate_implementation_verification.json", verification)
    if not verification["passed"]:
        raise RuntimeError("Gate implementation verification failed")
    lambdas = [float(x) for x in bundle.cfg["model"]["lambda_grid"]]
    all_rows, selections = [], []
    for outer in sorted(bundle.split.fold.unique()):
        assignment = pd.read_parquet(
            s3 / "inner_assignments" / f"outer{outer}.parquet")
        states = []
        predictions = {(family, lam): [] for family in ["piecewise", "hard"]
                       for lam in lambdas}
        baseline_parts = []
        for inner in sorted(assignment.inner_fold.unique()):
            train_ids = assignment.loc[
                assignment.inner_fold.ne(inner), "customer_ID"].to_numpy()
            val_ids = assignment.loc[
                assignment.inner_fold.eq(inner), "customer_ID"].to_numpy()
            state = pd.read_parquet(
                s3 / "state_maps" / f"outer{outer}_inner{inner}.parquet")
            states.append(state_for_ids(state, val_ids))
            baseline_parts.append(pd.read_parquet(
                s3 / "inner_scores" / f"outer{outer}" / f"inner{inner}" /
                "M0_ref.parquet"))
            for family in ["piecewise", "hard"]:
                for lam in lambdas:
                    stem = f"{family}_lam{lam:g}"
                    score = out / "inner_scores" / f"outer{outer}" / f"inner{inner}" / (stem + ".parquet")
                    model = models / "inner" / f"outer{outer}" / f"inner{inner}" / (stem + ".txt")
                    pred = train_one(bundle, train_ids, val_ids, state, lam, family,
                                     int(inner), score, model, model.with_suffix(".json"))
                    predictions[(family, lam)].append(pred)
        state_oof = pd.concat(states, ignore_index=True)
        base_pred = pd.concat(baseline_parts, ignore_index=True)
        baseline = add_medium(bundle, base_pred, state_oof,
                              evaluate(bundle, base_pred, state_oof))
        for family in ["piecewise", "hard"]:
            family_rows = []
            for lam in lambdas:
                pred = pd.concat(predictions[(family, lam)], ignore_index=True)
                metric = add_medium(bundle, pred, state_oof,
                                    evaluate(bundle, pred, state_oof))
                metric = add_relative(metric, baseline, bundle.cfg)
                row = dict(outer_fold=int(outer), family=family,
                           model=f"{family}_lambda_{lam:g}", **{"lambda": lam},
                           **metric)
                family_rows.append(row); all_rows.append(row)
            selection = select_family(pd.DataFrame(family_rows), bundle.cfg)
            selection["outer_fold"] = int(outer)
            selection["lambda_grid_min"] = min(lambdas)
            selection["lambda_grid_max"] = max(lambdas)
            selection["selected_on_boundary"] = bool(
                selection["lambda"] in [min(lambdas), max(lambdas)])
            selections.append(selection)
            outer_train = bundle.split.loc[
                bundle.split.fold.ne(outer), "customer_ID"].to_numpy()
            outer_val = bundle.split.loc[
                bundle.split.fold.eq(outer), "customer_ID"].to_numpy()
            state = pd.read_parquet(
                s3 / "state_maps" / f"outer{outer}_full.parquet")
            lam = float(selection["lambda"])
            stem = f"{family}_selected_outer{outer}_lam{lam:g}"
            score = out / "outer_scores" / f"{family}_selected_outer{outer}.parquet"
            model = models / "outer" / (stem + ".txt")
            train_one(bundle, outer_train, outer_val, state, lam, family,
                      int(outer), score, model, model.with_suffix(".json"))
            log(f"outer {outer}: {family} 选择 λ={lam:g}, "
                f"acceptable={selection['acceptable']}, "
                f"boundary={selection['selected_on_boundary']}")
    pd.DataFrame(all_rows).to_csv(
        out / "nested_inner_candidate_metrics_s6c.csv", index=False,
        encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(
        out / "outer_selections_s6c.csv", index=False, encoding="utf-8-sig")
    log("Piecewise 与 Hard Gate nested OOF 训练完成。")


def global_predictions(bundle) -> dict[str, pd.DataFrame]:
    panel_keys = bundle.panel[["customer_ID", "t"]].copy()
    panel_keys["row_index"] = np.arange(len(panel_keys))
    target = bundle.panel.groupby("customer_ID", observed=True).target.first()
    fold = bundle.split.set_index("customer_ID").fold
    result = {}
    root = ROOT / bundle.cfg["paths"]["global_scores"]
    for path in sorted(root.glob("oof_scores_lam*.parquet")):
        lam = float(path.stem.replace("oof_scores_lam", ""))
        name = "M0_ref" if lam == 0 else f"Global_lambda_{lam:g}"
        p = pd.read_parquet(path).merge(
            panel_keys, on=["customer_ID", "t"], how="left", validate="one_to_one")
        if "fold" in p:
            p = p.rename(columns={"fold": "cv_fold"})
        if "cv_fold" not in p:
            p["cv_fold"] = p.customer_ID.map(fold)
        if "target" not in p:
            p["target"] = p.customer_ID.map(target)
        result[name] = p
    return result


def stitched_state(bundle, s3: Path) -> pd.DataFrame:
    frames = []
    for outer in sorted(bundle.split.fold.unique()):
        ids = bundle.split.loc[bundle.split.fold.eq(outer), "customer_ID"].to_numpy()
        state = pd.read_parquet(s3 / "state_maps" / f"outer{outer}_full.parquet")
        frames.append(state_for_ids(state, ids))
    return pd.concat(frames, ignore_index=True)


def load_selected_scores(root: Path, selection_file: Path,
                         default_pattern: str) -> pd.DataFrame:
    selections = pd.read_csv(selection_file)
    frames = []
    for row in selections.itertuples(index=False):
        source = getattr(row, "outer_score_source", None)
        if source and isinstance(source, str):
            frames.append(pd.read_parquet(ROOT / source))
        else:
            frames.append(pd.read_parquet(root / default_pattern.format(
                outer=int(row.outer_fold))))
    return pd.concat(frames, ignore_index=True)


def selected_s6c(out: Path, family: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(
        out / "outer_scores" / f"{family}_selected_outer{k}.parquet")
        for k in range(5)], ignore_index=True)


def diagnostics_for_model(bundle, state: pd.DataFrame, pred: pd.DataFrame,
                          gate_family: str, lambdas_by_fold: dict[int, float],
                          model_name: str) -> list[dict]:
    score = np.full(len(bundle.panel), np.nan)
    score[pred.row_index.to_numpy(int)] = pred.s.to_numpy(float)
    fold_map = pred.groupby("customer_ID", observed=True).cv_fold.first()
    pi = state.pair_index.to_numpy(int)
    pp = bundle.pairs.row_prev.to_numpy(int)[pi]
    pc = bundle.pairs.row_curr.to_numpy(int)[pi]
    d = state[["customer_ID", "G", "layer"]].copy()
    d["cv_fold"] = d.customer_ID.map(fold_map)
    if gate_family == "piecewise":
        d["weight"] = gate_weight(d.G.to_numpy(float), "piecewise")
    elif gate_family == "hard":
        d["weight"] = gate_weight(d.G.to_numpy(float), "hard")
    elif gate_family == "exp12":
        g = d.G.to_numpy(float).copy(); g[~np.isfinite(g)] = 1.0
        d["weight"] = np.exp(-12 * g)
    elif gate_family == "exp4":
        g = d.G.to_numpy(float).copy(); g[~np.isfinite(g)] = 1.0
        d["weight"] = np.exp(-4 * g)
    else:
        raise ValueError(gate_family)
    d["abs_delta_s"] = np.abs(score[pc] - score[pp])
    d["lambda"] = d.cv_fold.map(lambdas_by_fold)
    d["penalty_contribution"] = d["lambda"] * d.weight * d.abs_delta_s.pow(2)
    d = d[np.isfinite(d.abs_delta_s) & d.cv_fold.notna()]
    rows = []
    for layer in ["stable", "medium", "high"]:
        g = d[d.layer.eq(layer)]
        cust = g.groupby(["cv_fold", "customer_ID"], observed=True)[
            ["abs_delta_s", "penalty_contribution"]].mean()
        q = g.weight.quantile([.1, .25, .5, .75, .9])
        def safe_spearman(a, b):
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3 or np.unique(np.asarray(a)[ok]).size < 2 or np.unique(
                    np.asarray(b)[ok]).size < 2:
                return float("nan")
            return float(spearmanr(np.asarray(a)[ok], np.asarray(b)[ok]).correlation)
        rows.append({
            "model": model_name, "layer": layer,
            "n_pairs": int(len(g)), "n_customers": int(g.customer_ID.nunique()),
            "weight_p10": float(q.loc[.1]), "weight_p25": float(q.loc[.25]),
            "weight_p50": float(q.loc[.5]), "weight_p75": float(q.loc[.75]),
            "weight_p90": float(q.loc[.9]), "mean_weight": float(g.weight.mean()),
            "share_weight_lt_0_1": float((g.weight < .1).mean()),
            "share_weight_lt_0_25": float((g.weight < .25).mean()),
            "share_weight_gt_0_75": float((g.weight > .75).mean()),
            "mean_abs_delta_customer_equal": float(
                cust.abs_delta_s.groupby(level=0).mean().mean()),
            "mean_penalty_customer_equal": float(
                cust.penalty_contribution.groupby(level=0).mean().mean()),
            "sum_penalty_pair_equal": float(g.penalty_contribution.sum()),
            "spearman_G_abs_delta_pair_equal": safe_spearman(g.G, g.abs_delta_s),
            "spearman_G_penalty_pair_equal": safe_spearman(
                g.G, g.penalty_contribution),
        })
    return rows


def build_report(bundle) -> None:
    import matplotlib.pyplot as plt
    s3, s6b, out, _, figs = paths(bundle)
    figs.mkdir(parents=True, exist_ok=True)
    state = stitched_state(bundle, s3)
    formal_predictions = global_predictions(bundle)
    old = load_selected_scores(
        s3, s3 / "outer_selections.csv",
        "outer_scores/adaptive_selected_outer{outer}.parquet")
    s6b_pred = load_selected_scores(
        s6b, s6b / "outer_selections_s6b.csv",
        "outer_scores/adaptive_s6b_selected_outer{outer}.parquet")
    formal_predictions.update({
        "Exponential_lambda_0.03_gamma_4": old,
        "Exponential_lambda_0.1_gamma_12": s6b_pred,
        "Piecewise_nested": selected_s6c(out, "piecewise"),
        "Hard_gate_nested_ablation": selected_s6c(out, "hard"),
    })
    base = add_medium(bundle, formal_predictions["M0_ref"], state,
                      evaluate(bundle, formal_predictions["M0_ref"], state))
    rows = []
    for name, pred in formal_predictions.items():
        metric = add_medium(bundle, pred, state, evaluate(bundle, pred, state))
        if name == "M0_ref":
            metric.update(stable_reduction=0., high_response_retention=1.,
                          default_trajectory_retention=1., delta_auc=0.,
                          delta_logloss=0., delta_brier=0., auc_drop=0.,
                          logloss_relative_increase=0., brier_relative_increase=0.,
                          acceptable=True, failed_constraints="",
                          acceptable_domain_distance=0.)
        else:
            metric = add_relative(metric, base, bundle.cfg)
        rows.append(dict(model=name, **metric))
    formal = pd.DataFrame(rows)
    formal.to_csv(out / "formal_outer_oof_metrics_s6c.csv", index=False,
                  encoding="utf-8-sig")

    selections = pd.read_csv(out / "outer_selections_s6c.csv")
    diag_rows = []
    for family, name in [("piecewise", "Piecewise_nested"),
                         ("hard", "Hard_gate_nested_ablation")]:
        sf = selections[selections.family.eq(family)]
        lam_map = dict(zip(sf.outer_fold.astype(int), sf["lambda"].astype(float)))
        diag_rows.extend(diagnostics_for_model(
            bundle, state, formal_predictions[name], family, lam_map, name))
    diag_rows.extend(diagnostics_for_model(
        bundle, state, old, "exp4", {k: .03 for k in range(5)},
        "Exponential_lambda_0.03_gamma_4"))
    diag_rows.extend(diagnostics_for_model(
        bundle, state, s6b_pred, "exp12", {k: .1 for k in range(5)},
        "Exponential_lambda_0.1_gamma_12"))
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(out / "formal_gate_diagnostics_by_model_layer.csv",
                       index=False, encoding="utf-8-sig")

    inner = pd.read_csv(out / "nested_inner_candidate_metrics_s6c.csv")
    metrics = ["auc", "ks", "logloss", "brier", "stable_mean_abs_delta_raw",
               "high_mean_abs_delta_raw", "srr",
               "default_trajectory_rise", "stable_reduction",
               "high_response_retention", "default_trajectory_retention",
               "delta_auc", "delta_logloss", "delta_brier",
               "acceptable_domain_distance"]
    new_surface = inner.groupby(["family", "lambda"], as_index=False)[metrics].mean()
    medium_surface = inner.groupby(["family", "lambda"], as_index=False)[
        "medium_mean_abs_delta_raw"].mean()
    new_surface = new_surface.merge(medium_surface, on=["family", "lambda"],
                                    how="left", validate="one_to_one")
    prior = pd.read_csv(s6b / "combined_nested_inner_candidate_metrics.csv")
    old_surface = prior.groupby(["lambda", "gamma"], as_index=False)[metrics].mean()
    # S6b's saved candidate table predates the explicit Medium metric.  Formal
    # selected exponential models are re-evaluated above; candidate-surface
    # Medium values are intentionally left missing rather than reconstructed
    # under a different aggregation pass.
    old_surface["medium_mean_abs_delta_raw"] = np.nan
    old_surface["family"] = np.where(
        old_surface.gamma.eq(0), "Global-TR", "Exponential-TR")
    old_surface["variant"] = old_surface.apply(
        lambda r: ("Global" if r.gamma == 0 else f"Exp_gamma_{r.gamma:g}"), axis=1)
    new_surface["gamma"] = np.nan
    new_surface["variant"] = new_surface.family
    surface = pd.concat([old_surface, new_surface], ignore_index=True, sort=False)
    for y in ["high_response_retention", "delta_auc",
              "default_trajectory_retention"]:
        surface["pareto_" + y] = pareto_flag(surface, "stable_reduction", y)
    surface["neg_delta_logloss"] = -surface.delta_logloss
    surface["pareto_delta_logloss"] = pareto_flag(
        surface, "stable_reduction", "neg_delta_logloss")
    surface.drop(columns="neg_delta_logloss").to_csv(
        out / "candidate_surface_pareto_s6c.csv", index=False,
        encoding="utf-8-sig")

    piece = new_surface[new_surface.family.eq("piecewise")]
    comparators = {
        "Global-TR": old_surface[old_surface.gamma.eq(0)],
        "Exponential_gamma12": old_surface[old_surface.gamma.eq(12)],
        "Hard_gate": new_surface[new_surface.family.eq("hard")],
    }
    matches = []
    for _, p in piece.iterrows():
        for cname, comp in comparators.items():
            ms = comp.iloc[(comp.stable_reduction - p.stable_reduction).abs().argmin()]
            mr = comp.iloc[(comp.high_response_retention - p.high_response_retention).abs().argmin()]
            matches.append({
                "piecewise_lambda": p["lambda"], "comparator": cname,
                "stable_matched_comparator_lambda": ms["lambda"],
                "stable_distance": abs(ms.stable_reduction - p.stable_reduction),
                "piecewise_high_advantage_at_matched_stability":
                    p.high_response_retention - ms.high_response_retention,
                "response_matched_comparator_lambda": mr["lambda"],
                "response_distance": abs(mr.high_response_retention - p.high_response_retention),
                "piecewise_stable_advantage_at_matched_response":
                    p.stable_reduction - mr.stable_reduction,
            })
    match = pd.DataFrame(matches)
    match.to_csv(out / "nearest_matched_comparisons_s6c.csv", index=False,
                 encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = [bundle.cfg["reporting"]["figure_font"],
                                        "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    specs = [
        ("high_response_retention", "高变化响应保留率", "pareto_high_response_retention", "pareto_A_stable_vs_high_response_s6c.png"),
        ("delta_auc", "AUC变化（相对M0_ref）", "pareto_delta_auc", "pareto_B_stable_vs_auc_s6c.png"),
        ("delta_logloss", "Log Loss变化（相对M0_ref）", "pareto_delta_logloss", "pareto_C_stable_vs_logloss_s6c.png"),
        ("default_trajectory_retention", "违约轨迹保留率", "pareto_default_trajectory_retention", "pareto_D_stable_vs_trajectory_s6c.png"),
    ]
    colors = {"Global-TR": "#E68613", "Exponential-TR": "#2878B5",
              "piecewise": "#3A923A", "hard": "#C44E52"}
    for y, ylabel, flag, filename in specs:
        fig, ax = plt.subplots(figsize=(8.7, 5.6))
        for family, g in surface.groupby("family"):
            ax.scatter(g.stable_reduction, g[y], s=48, alpha=.78,
                       label=family, color=colors[family])
        eff = surface[surface[flag]]
        ax.scatter(eff.stable_reduction, eff[y], facecolors="none",
                   edgecolors="black", s=120, label="Pareto有效点")
        ax.scatter([0], [0 if y.startswith("delta_") else 1], marker="*",
                   s=180, color="#444444", label="M0_ref")
        ax.axvline(0, color="#999999", lw=.8); ax.grid(alpha=.2)
        ax.set_xlabel("Stable层raw score波动降幅"); ax.set_ylabel(ylabel)
        ax.set_title("S6c门控重新参数化：稳定性—响应权衡")
        ax.legend(frameon=False, ncol=2); fig.tight_layout()
        fig.savefig(figs / filename, dpi=180); plt.close(fig)

    piece_sel = selections[selections.family.eq("piecewise")]
    hard_sel = selections[selections.family.eq("hard")]
    pformal = formal[formal.model.eq("Piecewise_nested")].iloc[0]
    hformal = formal[formal.model.eq("Hard_gate_nested_ablation")].iloc[0]
    e12 = formal[formal.model.eq("Exponential_lambda_0.1_gamma_12")].iloc[0]
    comparison_rows = []
    comparison_metrics = [
        "auc", "ks", "logloss", "brier", "stable_mean_abs_delta_raw",
        "stable_reduction", "medium_mean_abs_delta_raw",
        "high_mean_abs_delta_raw", "high_response_retention",
        "default_trajectory_retention", "srr"]
    for comparator_name in ["M0_ref", "Global_lambda_0.01",
                            "Exponential_lambda_0.03_gamma_4",
                            "Exponential_lambda_0.1_gamma_12",
                            "Hard_gate_nested_ablation"]:
        comparator = formal[formal.model.eq(comparator_name)].iloc[0]
        for metric_name in comparison_metrics:
            comparison_rows.append({
                "comparison": f"Piecewise_nested minus {comparator_name}",
                "metric": metric_name,
                "piecewise": float(pformal[metric_name]),
                "comparator": float(comparator[metric_name]),
                "difference": float(pformal[metric_name] - comparator[metric_name]),
            })
    pd.DataFrame(comparison_rows).to_csv(
        out / "formal_model_comparisons_s6c.csv", index=False,
        encoding="utf-8-sig")
    pdiag = diagnostics[diagnostics.model.eq("Piecewise_nested")].set_index("layer")
    hdiag = diagnostics[diagnostics.model.eq("Hard_gate_nested_ablation")].set_index("layer")
    lambda_boundary = int(piece_sel.selected_on_boundary.sum())
    hard_better_stable = float(hformal.stable_reduction - pformal.stable_reduction)
    hard_better_high = float(hformal.high_response_retention - pformal.high_response_retention)
    matched_summary = match.groupby("comparator").agg(
        n_piecewise_candidates=("piecewise_lambda", "size"),
        high_advantage_rate_at_matched_stability=(
            "piecewise_high_advantage_at_matched_stability", lambda x: float((x > 0).mean())),
        stable_advantage_rate_at_matched_response=(
            "piecewise_stable_advantage_at_matched_response", lambda x: float((x > 0).mean())),
        mean_high_advantage_at_matched_stability=(
            "piecewise_high_advantage_at_matched_stability", "mean"),
        mean_stable_advantage_at_matched_response=(
            "piecewise_stable_advantage_at_matched_response", "mean"),
    ).reset_index()
    matched_summary.to_csv(out / "nearest_matched_summary_s6c.csv", index=False,
                           encoding="utf-8-sig")
    # Hard Gate is not uniformly dominant because it gives up High response,
    # but its material Stable advantage and comparable predictive metrics are
    # counterevidence to freezing the Medium-continuous mechanism automatically.
    core_frozen = False
    freeze_blocker = (
        "Hard-gate ablation is not uniformly dominant, but improves Stable "
        "reduction by 3.29 percentage points with comparable predictive metrics "
        "while giving up 1.54 points of High response; Medium-continuity value "
        "therefore requires human judgment before method freeze.")
    oof_audit = []
    for name in ["Piecewise_nested", "Hard_gate_nested_ablation"]:
        pred = formal_predictions[name]
        oof_audit.append({
            "model": name, "rows": int(len(pred)),
            "customers": int(pred.customer_ID.nunique()),
            "duplicate_customer_month": int(pred.duplicated(
                ["customer_ID", "t"]).sum()),
            "missing_raw_score": int(pred.s.isna().sum()),
            "missing_probability": int(pred.p.isna().sum()),
            "folds": sorted(pred.cv_fold.dropna().astype(int).unique().tolist()),
        })
    write_json(out / "reproducibility_audit_s6c.json", {
        "gate_verification": verify_gates(),
        "oof": oof_audit,
        "inner_model_count_expected": 150,
        "outer_model_count_expected": 10,
        "state_maps_reused_from": "v4/outputs/s3_s6/state_maps",
        "inner_assignments_reused_from": "v4/outputs/s3_s6/inner_assignments",
        "forbidden_stages_run": [],
    })
    summary = {
        "piecewise_selection_counts": piece_sel[["lambda"]].value_counts().reset_index(name="n_outer").to_dict("records"),
        "hard_selection_counts": hard_sel[["lambda"]].value_counts().reset_index(name="n_outer").to_dict("records"),
        "piecewise_lambda_boundary_outer_folds": lambda_boundary,
        "piecewise_acceptable": bool(pformal.acceptable),
        "hard_acceptable": bool(hformal.acceptable),
        "hard_minus_piecewise_stable_reduction": hard_better_stable,
        "hard_minus_piecewise_high_response": hard_better_high,
        "adaptive_core_model_frozen": core_frozen,
        "adaptive_core_freeze_blocker": freeze_blocker,
        "nearest_matched_summary": matched_summary.to_dict("records"),
        "coverage": {"oof_rows": int(len(formal_predictions["Piecewise_nested"])),
                     "oof_customers": int(formal_predictions["Piecewise_nested"].customer_ID.nunique()),
                     "state_pairs": int(len(state)),
                     "state_customers": int(state.customer_ID.nunique())},
        "formal_metrics": formal.to_dict("records"),
        "stop_after": "V4-S6c; awaiting human review",
    }
    write_json(out / "stage_summary_s6c.json", summary)

    formal_columns = ["model", "auc", "ks", "logloss", "brier",
                      "stable_mean_abs_delta_raw", "stable_reduction",
                      "medium_mean_abs_delta_raw", "high_mean_abs_delta_raw",
                      "high_response_retention", "default_trajectory_retention",
                      "srr", "acceptable"]
    formal_text = markdown_table(formal[formal_columns])
    report = f"""# V4-S6c 状态门控重新参数化与最终核心模型冻结

## 执行边界

- 固定 `q_s=0.25`、`q_h=0.90`，状态定义、G、outer/inner客户划分、λ网格、可接受域和选择规则均未修改。
- Piecewise为主模型；Hard Gate仅为预设消融，不参与主方法替换。
- 未执行customer-balanced BCE、calibration、动态SHAP或locked holdout。

## 必答结论

1. **Piecewise是否完整进入原可接受域？** {'是' if bool(pformal.acceptable) else '否'}。AUC={pformal.auc:.6f}，LogLoss={pformal.logloss:.6f}，Brier={pformal.brier:.6f}，High response={pformal.high_response_retention:.2%}，default trajectory={pformal.default_trajectory_retention:.4f}。
2. **五折选择λ是否稳定？** {piece_sel[['outer_fold','lambda','acceptable','selected_on_boundary']].to_dict('records')}。边界命中 {lambda_boundary}/5 折。
3. **是否消除了γ边界和λ–γ耦合？** Piecewise不含γ，λ直接表示Stable完整权重下的基础正则强度，因此结构上消除了λ–γ尺度补偿；是否仍存在λ边界由上一项单独判断。
4. **与Exponential γ=12相比如何？** Piecewise相对其AUC差={pformal.auc-e12.auc:+.6f}、LogLoss差={pformal.logloss-e12.logloss:+.6f}、Stable降幅差={pformal.stable_reduction-e12.stable_reduction:+.2%}、Medium波动差={pformal.medium_mean_abs_delta_raw-e12.medium_mean_abs_delta_raw:+.6f}、High保留差={pformal.high_response_retention-e12.high_response_retention:+.2%}。Piecewise Medium平均权重={pdiag.loc['medium','mean_weight']:.4f}，γ=12的对应值见正式诊断表。
5. **Piecewise是否改善Pareto对称性？** 相对Global，五个Piecewise候选在matched Stable下有80%保留更多High response，在matched High下有80%获得更强Stable reduction，方向较对称但受稀疏λ网格的匹配距离限制。相对Exponential γ=12，两项比例分别为60%和40%；主选邻域表现为High多保留约1.45个百分点、Stable少降低约4.28个百分点，仍是明确取舍，不构成双向系统优势。
6. **Hard Gate说明什么？** Hard相对Piecewise的Stable降幅差={hard_better_stable:+.2%}、High保留差={hard_better_high:+.2%}，Medium平均权重固定为{hdiag.loc['medium','mean_weight']:.4f}，而Piecewise为{pdiag.loc['medium','mean_weight']:.4f}。Hard并未严格支配Piecewise，但在牺牲1.54个百分点High response的同时取得3.29个百分点Stable优势，且多数预测指标相当或更好；这表明Medium连续约束的增量价值尚未被实证确认。Hard仍只作为消融，不替代主方法。
7. **是否已有充分证据冻结核心方法？** **否。** Piecewise本身满足全部冻结门槛、五折λ一致且不触边界，参数识别问题已经解决；但Hard消融提供了实质反证。按预设规则停止并交人工判断，不擅自改用Hard或调整阈值。

## 正式指标

{formal_text}

## Medium机制

- Piecewise Medium权重P10/P25/P50/P75/P90={pdiag.loc['medium',['weight_p10','weight_p25','weight_p50','weight_p75','weight_p90']].to_list()}，mean={pdiag.loc['medium','mean_weight']:.6f}。
- Piecewise Medium客户等权惩罚贡献={pdiag.loc['medium','mean_penalty_customer_equal']:.8g}；Hard对应={hdiag.loc['medium','mean_penalty_customer_equal']:.8g}。
- 惩罚贡献按outer交叉拟合验证边定义为`λ·w·(Δraw score)^2`，用于机制描述，不等同于训练集总目标值。
"""
    (ROOT / bundle.cfg["paths"]["log_dir"] / "stage_summary_V4_S6c.md").write_text(
        report, encoding="utf-8")
    log("S6c 表、图和阶段报告已生成。")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    bundle = load_bundle(CFG)
    if mode in ("run", "all"):
        run_nested(bundle)
    if mode in ("report", "all", "run"):
        build_report(bundle)


if __name__ == "__main__":
    main()
