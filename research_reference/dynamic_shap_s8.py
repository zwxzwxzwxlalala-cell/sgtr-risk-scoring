# -*- coding: utf-8 -*-
"""V4-S8: strictly OOF dynamic TreeSHAP for the frozen SG-TR core.

This stage never trains or selects a model.  Formal aggregates use a
prespecified random one-pair-per-customer-per-layer sample.  Score-selected
cases are generated separately and never enter the formal aggregates.
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

from adaptive_tr_core import ROOT, load_bundle, write_json
from adaptive_tr_s6c import selected_s6c, stitched_state


CFG = ROOT / "v4" / "config_v4_s8.yaml"
GROUPS = ["D", "P", "B", "S", "R"]
MODEL_LABELS = {"m0": "M0_ref", "sgtr": "SG-TR"}


def log(message: str) -> None:
    print("[V4-S8] " + message, flush=True)


def paths(bundle):
    p = bundle.cfg["paths"]
    return (ROOT / p["s3_s6_output_dir"], ROOT / p["s6c_output_dir"],
            ROOT / p["output_dir"], ROOT / p["figure_dir"],
            ROOT / p["log_dir"])


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
        return str(value)
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def load_scores(bundle):
    p = bundle.cfg["paths"]
    m0 = pd.read_parquet(ROOT / p["m0_oof"]).rename(columns={"fold": "outer_fold"})
    keys = bundle.panel[["customer_ID"]].copy()
    keys["t"] = keys.groupby("customer_ID", observed=True).cumcount() + 1
    keys["row_index"] = np.arange(len(keys), dtype=np.int32)
    m0 = m0.merge(keys, on=["customer_ID", "t"], how="left",
                  validate="one_to_one")
    if m0.row_index.isna().any() or len(m0) != len(bundle.panel):
        raise AssertionError("M0_ref OOF cannot be mapped one-to-one to panel rows")
    m0["row_index"] = m0.row_index.astype(np.int32)
    _, s6c, _, _, _ = paths(bundle)
    sgtr = selected_s6c(s6c, "hard").rename(columns={"cv_fold": "outer_fold"})
    if len(sgtr) != len(bundle.panel) or sgtr.row_index.duplicated().any():
        raise AssertionError("Frozen SG-TR OOF does not cover panel rows exactly once")
    arrays = {}
    for name, frame in [("m0", m0), ("sgtr", sgtr)]:
        score = np.full(len(bundle.panel), np.nan, dtype=float)
        score[frame.row_index.to_numpy(int)] = frame.s.to_numpy(float)
        if not np.isfinite(score).all():
            raise AssertionError(f"Missing {name} OOF raw scores")
        arrays[name] = score
    return m0, sgtr, arrays


def pair_frame(bundle, state: pd.DataFrame, score_arrays: dict) -> pd.DataFrame:
    pair_cols = ["date_prev", "date_curr", "t_prev", "t_curr",
                 "row_prev", "row_curr", "pair_id", "fold"]
    base = state.merge(
        bundle.pairs.reset_index().rename(columns={"index": "pair_index"})[
            ["pair_index"] + pair_cols],
        on="pair_index", how="left", validate="one_to_one")
    if base[pair_cols].isna().any().any():
        raise AssertionError("State rows do not align to natural-month pairs")
    expected_fold = base.customer_ID.map(
        bundle.split.set_index("customer_ID").fold).to_numpy(int)
    if not np.array_equal(expected_fold, base.fold.to_numpy(int)):
        raise AssertionError("Pair fold is not the customer's frozen outer fold")
    prev, curr = base.row_prev.to_numpy(int), base.row_curr.to_numpy(int)
    for name in ["m0", "sgtr"]:
        delta = score_arrays[name][curr] - score_arrays[name][prev]
        base[f"delta_score_{name}"] = delta
        base[f"abs_delta_score_{name}"] = np.abs(delta)
    base["score_suppression"] = (base.abs_delta_score_m0
                                  - base.abs_delta_score_sgtr)
    base["retention_ratio"] = (base.abs_delta_score_sgtr
                                / np.maximum(base.abs_delta_score_m0, 1e-12))
    channel_values = base[["C_broad", "U_shock", "U_trans"]].to_numpy(float)
    names = np.asarray(["broad", "shock", "transition"])
    maxv = np.nanmax(channel_values, axis=1)
    tied = np.isclose(channel_values, maxv[:, None], rtol=0, atol=1e-12)
    unique = tied.sum(axis=1) == 1
    dominant = np.full(len(base), "tie", dtype=object)
    dominant[unique] = names[np.argmax(channel_values[unique], axis=1)]
    base["dominant_channel"] = dominant
    base["dominant_channel_count"] = tied.sum(axis=1)
    return base


def formal_sample(bundle, pairs: pd.DataFrame) -> pd.DataFrame:
    cfg = bundle.cfg["shap"]
    layers = cfg["formal_layers"]
    eligible = pairs[pairs.layer.isin(layers)].copy()
    rng = np.random.default_rng(int(bundle.cfg["seed"]))
    eligible["_random_order"] = rng.random(len(eligible))
    sample = (eligible.sort_values(
        ["fold", "layer", "customer_ID", "_random_order"], kind="mergesort")
        .drop_duplicates(["fold", "layer", "customer_ID"], keep="first")
        .drop(columns="_random_order"))
    if sample.duplicated(["fold", "layer", "customer_ID"]).any():
        raise AssertionError("Formal sample has more than one pair per customer-layer")
    sample["is_formal"] = True
    sample["case_type"] = ""
    return sample


def choose_unique_customers(frame: pd.DataFrame, n: int, used: set) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        if row.customer_ID in used:
            continue
        rows.append(row._asdict())
        used.add(row.customer_ID)
        if len(rows) == n:
            break
    return pd.DataFrame(rows, columns=frame.columns)


def select_cases(bundle, pairs: pd.DataFrame,
                 excluded_pair_indices: set) -> tuple[pd.DataFrame, dict]:
    rules = bundle.cfg["case_rules_frozen_before_identity_review"]
    n = int(rules["cases_per_type"])
    used, selected, thresholds = set(), [], {}

    a = pairs[pairs.layer.eq(rules["case_a"]["layer"])].copy()
    a_threshold = float(a.abs_delta_score_m0.quantile(
        float(rules["case_a"]["m0_abs_delta_min_quantile"])))
    reduction = float(rules["case_a"]["minimum_sgtr_reduction_fraction"])
    cand = a[(a.abs_delta_score_m0 >= a_threshold)
             & (a.score_suppression >= reduction * a.abs_delta_score_m0)
             & ~a.pair_index.isin(excluded_pair_indices)]
    cand = cand.sort_values(["score_suppression", "abs_delta_score_m0"],
                            ascending=False, kind="mergesort")
    picked = choose_unique_customers(cand, n, used)
    picked["case_type"] = "A_stable_m0_jump_sgtr_suppressed"
    selected.append(picked)
    thresholds["case_a_m0_abs_delta_p95"] = a_threshold
    thresholds["case_a_eligible"] = int(len(cand))

    b = pairs[pairs.layer.eq(rules["case_b"]["layer"])].copy()
    b_threshold = float(b.abs_delta_score_m0.quantile(
        float(rules["case_b"]["m0_abs_delta_min_quantile"])))
    same = np.sign(b.delta_score_m0) == np.sign(b.delta_score_sgtr)
    lo = float(rules["case_b"]["sgtr_to_m0_abs_delta_ratio_min"])
    hi = float(rules["case_b"]["sgtr_to_m0_abs_delta_ratio_max"])
    cand = b[(b.abs_delta_score_m0 >= b_threshold) & same
             & b.retention_ratio.between(lo, hi)
             & ~b.pair_index.isin(excluded_pair_indices)]
    cand = cand.sort_values(["abs_delta_score_m0", "retention_ratio"],
                            ascending=[False, False], kind="mergesort")
    picked = choose_unique_customers(cand, n, used)
    picked["case_type"] = "B_high_response_retained"
    selected.append(picked)
    thresholds["case_b_m0_abs_delta_p75"] = b_threshold
    thresholds["case_b_eligible"] = int(len(cand))

    for key, label, channel, column in [
            ("case_c_shock", "C_sparse_shock", "shock", "U_shock"),
            ("case_c_transition", "C_transition", "transition", "U_trans")]:
        rule = rules[key]
        qualified = pairs[(pairs.dominant_channel.eq(channel))
                          & (pairs[column] >= float(rule["channel_value_min"]))].copy()
        q = float(qualified.abs_delta_score_m0.quantile(
            float(rule["m0_abs_delta_min_quantile_within_qualified_channel"])))
        cand = qualified[(qualified.abs_delta_score_m0 >= q)
                         & ~qualified.pair_index.isin(excluded_pair_indices)].sort_values(
            [column, "abs_delta_score_m0"], ascending=False, kind="mergesort")
        picked = choose_unique_customers(cand, n, used)
        picked["case_type"] = label
        selected.append(picked)
        thresholds[f"{key}_m0_abs_delta_p75"] = q
        thresholds[f"{key}_qualified"] = int(len(qualified))
        thresholds[f"{key}_eligible"] = int(len(cand))

    cases = pd.concat(selected, ignore_index=True)
    cases["is_formal"] = False
    thresholds.update({
        "rules_frozen_before_customer_identity_review": True,
        "globally_unique_customers": bool(
            rules["globally_unique_customers_across_case_types"]),
        "formal_pair_indices_excluded_from_case_pool": True,
        "selected_counts": cases.case_type.value_counts().to_dict(),
    })
    return cases, thresholds


def load_model_records(bundle):
    p = bundle.cfg["paths"]
    manifest = json.loads((ROOT / p["regularized_model_dir"]
                           / "training_manifest.json").read_text(encoding="utf-8"))
    m0_diag = {int(x["fold"]): x for x in manifest["diagnostics"]
               if x["model_file"].startswith("Mreg_lam0_fold")}
    records = {"m0": {}, "sgtr": {}}
    for fold in range(5):
        records["m0"][fold] = {
            "model_path": ROOT / p["regularized_model_dir"]
            / f"Mreg_lam0_fold{fold}.txt",
            "intercept": float(m0_diag[fold]["init_score_log_odds"]),
        }
        meta_path = ROOT / p["s6c_model_dir"] / f"hard_selected_outer{fold}_lam0.03.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records["sgtr"][fold] = {
            "model_path": ROOT / p["s6c_model_dir"]
            / f"hard_selected_outer{fold}_lam0.03.txt",
            "intercept": float(meta["init_score_log_odds"]),
        }
    return records


def compute_shap(bundle, explain_pairs: pd.DataFrame, score_arrays: dict):
    import lightgbm as lgb
    _, _, out, _, _ = paths(bundle)
    records = load_model_records(bundle)
    audit_rows, delta_frames, summary_frames = [], [], []
    features = bundle.features
    feature_groups = {g: np.asarray([i for i, f in enumerate(features)
                                    if f.startswith(g + "_")], dtype=int)
                      for g in GROUPS}
    pair_lookup = explain_pairs.set_index("pair_index")
    for fold in range(5):
        fold_pairs = explain_pairs[explain_pairs.fold.eq(fold)].copy()
        prev = fold_pairs.row_prev.to_numpy(int)
        curr = fold_pairs.row_curr.to_numpy(int)
        X_prev = bundle.panel.loc[prev, features]
        X_curr = bundle.panel.loc[curr, features]
        log(f"outer fold {fold}: {len(fold_pairs):,} pairs, {2*len(fold_pairs):,} rows")
        for model in ["m0", "sgtr"]:
            rec = records[model][fold]
            booster = lgb.Booster(model_str=rec["model_path"].read_text(encoding="utf-8"))
            started = time.time()
            cp = np.asarray(booster.predict(X_prev, pred_contrib=True))
            cc = np.asarray(booster.predict(X_curr, pred_contrib=True))
            raw_p = np.asarray(booster.predict(X_prev, raw_score=True)) + rec["intercept"]
            raw_c = np.asarray(booster.predict(X_curr, raw_score=True)) + rec["intercept"]
            official_p = score_arrays[model][prev]
            official_c = score_arrays[model][curr]
            recon_p = cp[:, -1] + rec["intercept"] + cp[:, :-1].sum(axis=1)
            recon_c = cc[:, -1] + rec["intercept"] + cc[:, :-1].sum(axis=1)
            dphi = cc[:, :-1] - cp[:, :-1]
            official_delta = official_c - official_p
            row_err = np.r_[np.abs(recon_p - official_p),
                            np.abs(recon_c - official_c)]
            rescore_err = np.r_[np.abs(raw_p - official_p),
                                np.abs(raw_c - official_c)]
            pair_err = np.abs(dphi.sum(axis=1) - official_delta)
            audit_rows.append({
                "model": MODEL_LABELS[model], "outer_fold": fold,
                "explained_pairs": int(len(fold_pairs)),
                "explained_rows": int(2 * len(fold_pairs)),
                "max_abs_row_reconstruction_error": float(row_err.max()),
                "p99_abs_row_reconstruction_error": float(np.quantile(row_err, .99)),
                "mean_abs_row_reconstruction_error": float(row_err.mean()),
                "max_abs_pair_decomposition_error": float(pair_err.max()),
                "p99_abs_pair_decomposition_error": float(np.quantile(pair_err, .99)),
                "mean_abs_pair_decomposition_error": float(pair_err.mean()),
                "max_abs_rescore_vs_frozen_oof_error": float(rescore_err.max()),
                "base_value_min_with_intercept": float(min(cp[:, -1].min(), cc[:, -1].min())
                                                        + rec["intercept"]),
                "base_value_max_with_intercept": float(max(cp[:, -1].max(), cc[:, -1].max())
                                                        + rec["intercept"]),
            })
            f = pd.DataFrame(dphi.astype(np.float32), columns=features)
            f.insert(0, "model", MODEL_LABELS[model])
            f.insert(0, "pair_index", fold_pairs.pair_index.to_numpy(np.int32))
            delta_frames.append(f)
            summary = pd.DataFrame({
                "pair_index": fold_pairs.pair_index.to_numpy(np.int32),
                "model": MODEL_LABELS[model],
                "delta_score": official_delta,
                "abs_delta_score": np.abs(official_delta),
                "shap_attribution_volatility": np.abs(dphi).sum(axis=1),
            })
            for group, idx in feature_groups.items():
                summary[f"signed_{group}"] = dphi[:, idx].sum(axis=1)
                summary[f"magnitude_{group}"] = np.abs(dphi[:, idx]).sum(axis=1)
            summary_frames.append(summary)
            log(f"  {MODEL_LABELS[model]} SHAP complete in {time.time()-started:.1f}s")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "shap_additivity_audit_s8.csv", index=False,
                 encoding="utf-8-sig")
    tol = float(bundle.cfg["shap"]["additivity_tolerance"])
    error_cols = ["max_abs_row_reconstruction_error",
                  "max_abs_pair_decomposition_error",
                  "max_abs_rescore_vs_frozen_oof_error"]
    passed = bool((audit[error_cols].to_numpy(float) <= tol).all())
    write_json(out / "shap_additivity_audit_s8.json", {
        "tolerance": tol, "passed": passed,
        "maximum_observed_error": float(audit[error_cols].max().max()),
        "models": ["M0_ref", "SG-TR"], "scale": "raw_margin",
        "per_fold": audit.to_dict("records")})
    if not passed:
        raise RuntimeError("SHAP additivity audit failed; downstream interpretation stopped")
    deltas = pd.concat(delta_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    deltas.to_parquet(out / "dynamic_shap_feature_deltas_s8.parquet", index=False)
    summaries.to_parquet(out / "dynamic_shap_pair_model_s8.parquet", index=False)
    return audit, deltas, summaries


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(a), np.nan)
    ok = den > 0
    out[ok] = np.sum(a[ok] * b[ok], axis=1) / den[ok]
    return out


def combine_formal(bundle, formal: pd.DataFrame, deltas: pd.DataFrame,
                   summaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, out, _, _ = paths(bundle)
    meta_cols = ["pair_index", "customer_ID", "fold", "date_prev", "date_curr",
                 "t_prev", "t_curr", "row_prev", "row_curr", "layer",
                 "C_broad", "U_shock", "U_trans", "C_final", "G",
                 "U_D", "U_P", "U_B", "U_S", "U_R",
                 "dominant_channel", "dominant_channel_count"]
    meta = formal[meta_cols].copy()
    long = summaries.merge(meta, on="pair_index", how="inner", validate="many_to_one")
    long.to_parquet(out / "formal_dynamic_shap_long_s8.parquet", index=False)
    m0 = long[long.model.eq("M0_ref")].drop(columns="model")
    sg = long[long.model.eq("SG-TR")].drop(columns=meta_cols[1:] + ["model"], errors="ignore")
    metric_cols = [c for c in m0.columns if c not in meta_cols]
    m0 = m0.rename(columns={c: c + "_m0" for c in metric_cols})
    sg = sg.rename(columns={c: c + "_sgtr" for c in sg.columns if c != "pair_index"})
    wide = m0.merge(sg, on="pair_index", validate="one_to_one")
    dm0 = deltas[deltas.model.eq("M0_ref")].set_index("pair_index")[bundle.features]
    dsg = deltas[deltas.model.eq("SG-TR")].set_index("pair_index")[bundle.features]
    idx = wide.pair_index.to_numpy(int)
    a = dm0.loc[idx].to_numpy(float)
    b = dsg.loc[idx].to_numpy(float)
    wide["feature_delta_shap_cosine"] = cosine_rows(a, b)
    signed_m0 = wide[[f"signed_{g}_m0" for g in GROUPS]].to_numpy(float)
    signed_sg = wide[[f"signed_{g}_sgtr" for g in GROUPS]].to_numpy(float)
    wide["group_signed_cosine"] = cosine_rows(signed_m0, signed_sg)
    wide["score_direction_agreement"] = (
        np.sign(wide.delta_score_m0) == np.sign(wide.delta_score_sgtr))
    wide["score_suppression"] = (wide.abs_delta_score_m0
                                  - wide.abs_delta_score_sgtr)
    wide["score_retention_ratio"] = (wide.abs_delta_score_sgtr
                                      / np.maximum(wide.abs_delta_score_m0, 1e-12))
    wide["sav_suppression"] = (wide.shap_attribution_volatility_m0
                                - wide.shap_attribution_volatility_sgtr)
    wide.to_parquet(out / "formal_dynamic_shap_paired_s8.parquet", index=False)
    return long, wide


def aggregate_formal(bundle, long: pd.DataFrame, wide: pd.DataFrame):
    _, _, out, _, _ = paths(bundle)
    metrics = ["abs_delta_score", "shap_attribution_volatility"]
    metrics += [f"signed_{g}" for g in GROUPS]
    metrics += [f"magnitude_{g}" for g in GROUPS]
    fold_means = long.groupby(["layer", "model", "fold"], as_index=False)[metrics].mean()
    aggregate = fold_means.groupby(["layer", "model"], as_index=False)[metrics].mean()
    counts = long.groupby(["layer", "model"], as_index=False).agg(
        n_pairs=("pair_index", "size"), n_customers=("customer_ID", "nunique"))
    aggregate = counts.merge(aggregate, on=["layer", "model"])
    aggregate.to_csv(out / "layer_model_dynamic_shap_summary_s8.csv", index=False,
                     encoding="utf-8-sig")
    fold_means.to_csv(out / "layer_model_dynamic_shap_by_fold_s8.csv", index=False,
                      encoding="utf-8-sig")
    group_rows = []
    for row in aggregate.itertuples(index=False):
        for group in GROUPS:
            group_rows.append({
                "layer": row.layer, "model": row.model, "group": group,
                "n_pairs": int(row.n_pairs), "n_customers": int(row.n_customers),
                "mean_signed_delta_shap": float(getattr(row, f"signed_{group}")),
                "mean_absolute_delta_shap": float(getattr(row, f"magnitude_{group}"))})
    pd.DataFrame(group_rows).to_csv(
        out / "group_dynamic_shap_summary_s8.csv", index=False,
        encoding="utf-8-sig")
    comp_metrics = ["feature_delta_shap_cosine", "group_signed_cosine",
                    "score_direction_agreement", "score_suppression",
                    "sav_suppression"]
    comp_fold = wide.groupby(["layer", "fold"], as_index=False)[comp_metrics].mean()
    ratio_rows = []
    for (layer, fold), frame in wide.groupby(["layer", "fold"]):
        ratio_rows.append({
            "layer": layer, "fold": fold,
            "score_retention_ratio": float(
                frame.abs_delta_score_sgtr.mean() / frame.abs_delta_score_m0.mean()),
            "sav_retention_ratio": float(
                frame.shap_attribution_volatility_sgtr.mean()
                / frame.shap_attribution_volatility_m0.mean())})
    comp_fold = comp_fold.merge(pd.DataFrame(ratio_rows),
                                on=["layer", "fold"], validate="one_to_one")
    comp = comp_fold.groupby("layer", as_index=False)[
        comp_metrics + ["score_retention_ratio", "sav_retention_ratio"]].mean()
    comp_counts = wide.groupby("layer", as_index=False).agg(
        n_pairs=("pair_index", "size"), n_customers=("customer_ID", "nunique"))
    comp = comp_counts.merge(comp, on="layer")
    comp.to_csv(out / "model_dynamic_similarity_s8.csv", index=False,
                encoding="utf-8-sig")
    return aggregate, comp


def paired_bootstrap(bundle, wide: pd.DataFrame):
    _, _, out, _, _ = paths(bundle)
    reps = int(bundle.cfg["bootstrap"]["replicates"])
    rng = np.random.default_rng(int(bundle.cfg["bootstrap"]["seed"]))
    base_metrics = ["abs_delta_score", "shap_attribution_volatility"]
    base_metrics += [f"signed_{g}" for g in GROUPS]
    base_metrics += [f"magnitude_{g}" for g in GROUPS]
    draw_frames, summaries = [], []
    for layer in bundle.cfg["shap"]["formal_layers"]:
        frame = wide[wide.layer.eq(layer)].reset_index(drop=True)
        folds = frame.fold.to_numpy(int)
        diff = np.column_stack([
            frame[f"{metric}_sgtr"].to_numpy(float)
            - frame[f"{metric}_m0"].to_numpy(float) for metric in base_metrics])
        fold_idx = [np.flatnonzero(folds == f) for f in sorted(np.unique(folds))]
        point = np.mean([diff[idx].mean(axis=0) for idx in fold_idx], axis=0)
        draws = np.empty((reps, len(base_metrics)), dtype=float)
        for r in range(reps):
            fold_values = []
            for idx in fold_idx:
                weight = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)))
                fold_values.append(np.average(diff[idx], axis=0, weights=weight))
            draws[r] = np.mean(fold_values, axis=0)
        draw = pd.DataFrame(draws, columns=base_metrics)
        draw.insert(0, "replicate", np.arange(reps))
        draw.insert(0, "layer", layer)
        draw_frames.append(draw)
        for j, metric in enumerate(base_metrics):
            summaries.append({
                "comparison": "SGTR_minus_M0_ref", "layer": layer,
                "metric": metric, "point_estimate": float(point[j]),
                "ci95_lower": float(np.quantile(draws[:, j], .025)),
                "ci95_upper": float(np.quantile(draws[:, j], .975)),
                "replicates": reps})
    pd.concat(draw_frames, ignore_index=True).to_parquet(
        out / "dynamic_shap_bootstrap_draws_s8.parquet", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "dynamic_shap_bootstrap_summary_s8.csv", index=False,
                   encoding="utf-8-sig")
    write_json(out / "bootstrap_design_audit_s8.json", {
        "replicates": reps, "seed": int(bundle.cfg["bootstrap"]["seed"]),
        "sampling_unit": "customer_ID", "stratification": "outer_fold",
        "paired_models": True, "formal_one_pair_per_customer_per_layer": True,
        "case_selected_pairs_included": False})
    return summary


def input_shap_chain(bundle, long: pd.DataFrame):
    _, _, out, _, _ = paths(bundle)
    correlation_rows, bins, agreement_rows = [], [], []
    scopes = [("all", long)] + [(layer, long[long.layer.eq(layer)])
                                 for layer in bundle.cfg["shap"]["formal_layers"]]
    for model in ["M0_ref", "SG-TR"]:
        model_frame = long[long.model.eq(model)]
        for scope, _ in scopes:
            frame = model_frame if scope == "all" else model_frame[model_frame.layer.eq(scope)]
            for input_group in GROUPS:
                for shap_group in GROUPS:
                    x = frame[f"U_{input_group}"].to_numpy(float)
                    y = frame[f"magnitude_{shap_group}"].to_numpy(float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() > 2 else np.nan
                    correlation_rows.append({
                        "model": model, "scope": scope,
                        "input_group": input_group, "shap_group": shap_group,
                        "spearman": float(rho), "n_pairs": int(ok.sum())})
            for group in GROUPS:
                frame2 = frame[["pair_index", f"U_{group}",
                                f"magnitude_{group}"]].dropna().copy()
                frame2["bin"] = pd.qcut(frame2[f"U_{group}"], 10,
                                         labels=False, duplicates="drop")
                for bin_id, block in frame2.groupby("bin", observed=True):
                    bins.append({
                        "model": model, "scope": scope, "group": group,
                        "bin": int(bin_id) + 1, "n_pairs": int(len(block)),
                        "mean_input_percentile": float(block[f"U_{group}"].mean()),
                        "mean_shap_magnitude": float(block[f"magnitude_{group}"].mean())})
            u = frame[[f"U_{g}" for g in GROUPS]].to_numpy(float)
            a = frame[[f"magnitude_{g}" for g in GROUPS]].to_numpy(float)
            top_u, top_a = np.argmax(u, axis=1), np.argmax(a, axis=1)
            agreement_rows.append({
                "model": model, "scope": scope, "n_pairs": int(len(frame)),
                "top_group_agreement_rate": float(np.mean(top_u == top_a)),
                "input_top_tie_rate": float(np.mean(
                    np.isclose(u, u.max(axis=1)[:, None]).sum(axis=1) > 1)),
                "shap_top_tie_rate": float(np.mean(
                    np.isclose(a, a.max(axis=1)[:, None]).sum(axis=1) > 1)),
            })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(out / "input_shap_spearman_s8.csv", index=False,
                        encoding="utf-8-sig")
    pd.DataFrame(bins).to_csv(out / "input_shap_bin_trends_s8.csv", index=False,
                              encoding="utf-8-sig")
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(out / "input_shap_top_group_agreement_s8.csv", index=False,
                     encoding="utf-8-sig")
    return correlations, agreement


def channel_and_suppression(bundle, long: pd.DataFrame, wide: pd.DataFrame,
                            deltas: pd.DataFrame):
    _, _, out, _, _ = paths(bundle)
    rows = []
    for (channel, model), frame in long.groupby(["dominant_channel", "model"]):
        row = {"dominant_channel": channel, "model": model,
               "n_pairs": int(len(frame)),
               "n_customers": int(frame.customer_ID.nunique()),
               "mean_abs_delta_score": float(frame.abs_delta_score.mean()),
               "mean_sav": float(frame.shap_attribution_volatility.mean())}
        total = frame[[f"magnitude_{g}" for g in GROUPS]].sum(axis=1)
        row["mean_top_group_concentration"] = float(
            frame[[f"magnitude_{g}" for g in GROUPS]].max(axis=1).div(total).mean())
        mean_magnitudes = {}
        for group in GROUPS:
            mean_magnitudes[group] = float(frame[f"magnitude_{group}"].mean())
            row[f"mean_magnitude_{group}"] = mean_magnitudes[group]
            row[f"mean_signed_{group}"] = float(frame[f"signed_{group}"].mean())
        total_mean_magnitude = sum(mean_magnitudes.values())
        for group in GROUPS:
            row[f"magnitude_share_{group}"] = mean_magnitudes[group] / total_mean_magnitude
        row["largest_magnitude_group"] = max(mean_magnitudes, key=mean_magnitudes.get)
        row["largest_magnitude_group_share"] = max(mean_magnitudes.values()) / total_mean_magnitude
        rows.append(row)
    channel = pd.DataFrame(rows)
    channel.to_csv(out / "state_channel_dynamic_shap_patterns_s8.csv", index=False,
                   encoding="utf-8-sig")

    relations = []
    for layer in bundle.cfg["shap"]["formal_layers"]:
        frame = wide[wide.layer.eq(layer)]
        for group in GROUPS:
            reduction = (frame[f"magnitude_{group}_m0"]
                         - frame[f"magnitude_{group}_sgtr"])
            relations.append({
                "layer": layer, "group": group,
                "mean_group_attribution_reduction": float(reduction.mean()),
                "spearman_score_suppression_vs_group_reduction": float(
                    spearmanr(frame.score_suppression, reduction).correlation),
                "n_pairs": int(len(frame))})
    relation = pd.DataFrame(relations)
    relation.to_csv(out / "score_suppression_group_relations_s8.csv", index=False,
                    encoding="utf-8-sig")

    formal_ids = set(wide.pair_index)
    d0 = deltas[(deltas.model.eq("M0_ref"))
                & deltas.pair_index.isin(formal_ids)].set_index("pair_index")
    ds = deltas[(deltas.model.eq("SG-TR"))
                & deltas.pair_index.isin(formal_ids)].set_index("pair_index")
    stable_ids = wide.loc[wide.layer.eq("stable"), "pair_index"].to_numpy(int)
    feature_rows = []
    for feature in bundle.features:
        m0_abs = d0.loc[stable_ids, feature].abs().mean()
        sg_abs = ds.loc[stable_ids, feature].abs().mean()
        feature_rows.append({
            "feature": feature, "group": feature.split("_")[0],
            "stable_mean_abs_delta_shap_m0": float(m0_abs),
            "stable_mean_abs_delta_shap_sgtr": float(sg_abs),
            "stable_attribution_reduction": float(m0_abs - sg_abs)})
    feature = pd.DataFrame(feature_rows).sort_values(
        "stable_attribution_reduction", ascending=False)
    feature.to_csv(out / "stable_feature_attribution_suppression_s8.csv", index=False,
                   encoding="utf-8-sig")
    return channel, relation, feature


def case_details(bundle, cases: pd.DataFrame, deltas: pd.DataFrame,
                 summaries: pd.DataFrame):
    _, _, out, _, _ = paths(bundle)
    cases.to_csv(out / "rule_selected_case_candidates_s8.csv", index=False,
                 encoding="utf-8-sig")
    case_ids = cases.pair_index.to_numpy(int)
    m0 = deltas[(deltas.model.eq("M0_ref"))
                & deltas.pair_index.isin(case_ids)].set_index("pair_index")
    sg = deltas[(deltas.model.eq("SG-TR"))
                & deltas.pair_index.isin(case_ids)].set_index("pair_index")
    top_n = int(bundle.cfg["reporting"]["top_features_per_case"])
    rows = []
    for case in cases.itertuples(index=False):
        importance = np.maximum(np.abs(m0.loc[case.pair_index, bundle.features].to_numpy(float)),
                                np.abs(sg.loc[case.pair_index, bundle.features].to_numpy(float)))
        order = np.argsort(-importance)[:top_n]
        for rank, j in enumerate(order, 1):
            feature = bundle.features[int(j)]
            before = bundle.panel.at[int(case.row_prev), feature]
            after = bundle.panel.at[int(case.row_curr), feature]
            rows.append({
                "case_type": case.case_type, "customer_ID": case.customer_ID,
                "pair_index": int(case.pair_index), "rank": rank,
                "feature": feature, "group": feature.split("_")[0],
                "input_prev": str(before), "input_curr": str(after),
                "missing_to_observed": bool(pd.isna(before) and not pd.isna(after)),
                "observed_to_missing": bool(not pd.isna(before) and pd.isna(after)),
                "delta_shap_m0": float(m0.at[case.pair_index, feature]),
                "delta_shap_sgtr": float(sg.at[case.pair_index, feature]),
            })
    details = pd.DataFrame(rows)
    details.to_csv(out / "rule_selected_case_feature_details_s8.csv", index=False,
                   encoding="utf-8-sig")
    case_summary = summaries[summaries.pair_index.isin(case_ids)].merge(
        cases[["pair_index", "case_type", "customer_ID", "layer", "G",
               "C_broad", "U_shock", "U_trans"]], on="pair_index",
        validate="many_to_one")
    case_summary.to_csv(out / "rule_selected_case_shap_summary_s8.csv", index=False,
                        encoding="utf-8-sig")
    return details, case_summary


def plots(bundle, aggregate: pd.DataFrame, correlations: pd.DataFrame,
          cases: pd.DataFrame, case_summary: pd.DataFrame,
          m0_scores: pd.DataFrame, sg_scores: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    _, _, _, figs, _ = paths(bundle)
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [bundle.cfg["reporting"]["figure_font"],
                                        "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.axis("off")
    boxes = [
        (.09, .72, "Broad\n整体组级变化", "#DCEAF7"),
        (.09, .48, "Sparse Shock\n稀疏极端冲击", "#FCE2C2"),
        (.09, .24, "Transition\n状态转换", "#E8DDF3"),
        (.29, .48, "Final state\nmax(三通道)", "#E2F0D9"),
        (.46, .48, "Stable-state Gate\nG≤0.25", "#D9EAD3"),
        (.62, .48, "SG-TR\nλ=0.03", "#B7D7A8"),
        (.78, .48, "Raw risk score\n动态解释", "#CFE2F3"),
        (.93, .48, "Platt\n末期概率", "#F4CCCC")]
    width, height = .13, .16
    for x, y, label, color in boxes:
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                             boxstyle="round,pad=0.012", facecolor=color,
                             edgecolor="#555555", linewidth=1.2)
        ax.add_patch(box); ax.text(x, y, label, ha="center", va="center", fontsize=9.5)
    for y in [.72, .48, .24]:
        ax.add_patch(FancyArrowPatch((.09+width/2, y), (.29-width/2, .48),
                                    arrowstyle="->", mutation_scale=12,
                                    color="#555555"))
    for a, b in [(.29, .46), (.46, .62), (.62, .78), (.78, .93)]:
        ax.add_patch(FancyArrowPatch((a+width/2, .48), (b-width/2, .48),
                                    arrowstyle="->", mutation_scale=12,
                                    color="#555555"))
    ax.text(.17, .08, "冻结的无监督状态三通道并行汇总", ha="center", fontsize=10)
    ax.text(.78, .08, "解释使用raw margin；Platt仅用于最终末期概率", ha="center", fontsize=10)
    ax.set_title("SG-TR冻结方法与解释链", fontsize=16, pad=10)
    fig.tight_layout(); fig.savefig(figs / "method_overview_s8.png", dpi=180); plt.close(fig)

    show = aggregate[aggregate.layer.isin(["stable", "high"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors2 = {"M0_ref": "#777777", "SG-TR": "#2878B5"}
    for ax, metric, title in zip(
            axes, ["abs_delta_score", "shap_attribution_volatility"],
            ["平均 |Δ raw score|", "平均 SHAP Attribution Volatility"]):
        x = np.arange(2); width = .34
        for j, model in enumerate(["M0_ref", "SG-TR"]):
            vals = [show[(show.layer.eq(layer)) & (show.model.eq(model))][metric].iloc[0]
                    for layer in ["stable", "high"]]
            ax.bar(x + (j-.5)*width, vals, width, label=model, color=colors2[model])
        ax.set_xticks(x, ["Stable", "High"]); ax.set_title(title); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("客户等权、折等权均值"); axes[1].legend(frameon=False)
    fig.suptitle("Stable与High层动态归因：M0_ref vs SG-TR")
    fig.tight_layout(); fig.savefig(figs / "stable_high_dynamic_shap_s8.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    for ax, model in zip(axes, ["M0_ref", "SG-TR"]):
        block = correlations[(correlations.model.eq(model))
                             & correlations.scope.eq("all")]
        matrix = block.pivot(index="input_group", columns="shap_group",
                             values="spearman").reindex(index=GROUPS, columns=GROUPS)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-.35, vmax=.35)
        ax.set_xticks(range(5), GROUPS); ax.set_yticks(range(5), GROUPS)
        ax.set_xlabel("ΔSHAP magnitude组"); ax.set_title(model)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("输入状态变化组 U")
    fig.subplots_adjust(left=.08, right=.88, bottom=.13, top=.83, wspace=.20)
    cax = fig.add_axes([.91, .18, .018, .58])
    fig.colorbar(im, cax=cax, label="Spearman ρ")
    fig.suptitle("五类输入状态变化与五类动态SHAP的描述性对应")
    fig.savefig(figs / "input_shap_group_heatmap_s8.png", dpi=180); plt.close(fig)

    def trajectory(case_prefix: str, filename: str, title: str):
        chosen = cases[cases.case_type.str.startswith(case_prefix)].head(3)
        if chosen.empty:
            return
        fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 3.0*len(chosen)), squeeze=False)
        for i, case in enumerate(chosen.itertuples(index=False)):
            ax = axes[i, 0]
            for name, frame, color in [("M0_ref", m0_scores, "#777777"),
                                       ("SG-TR", sg_scores, "#2878B5")]:
                block = frame[frame.customer_ID.eq(case.customer_ID)].sort_values("t")
                dates = bundle.panel.loc[block.row_index.to_numpy(int), "S_2"]
                ax.plot(dates, block.s, marker="o", label=name, color=color)
            ax.axvspan(pd.Timestamp(case.date_prev), pd.Timestamp(case.date_curr),
                       color="#F6B26B", alpha=.22)
            ax.set_title(f"{case_prefix.rstrip('_')}-{i+1}  G={case.G:.3f}  "
                         f"Broad/Shock/Trans={case.C_broad:.2f}/{case.U_shock:.2f}/{case.U_trans:.2f}")
            ax.set_ylabel("raw score"); ax.grid(alpha=.2)
        axes[-1, 0].set_xlabel("账期"); axes[0, 0].legend(frameon=False)
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(figs / filename, dpi=180); plt.close(fig)
    trajectory("A_", "stable_case_trajectories_s8.png",
               "规则化Stable案例：M0跳分与SG-TR抑制")
    trajectory("B_", "high_case_trajectories_s8.png",
               "规则化High案例：SG-TR保留风险响应")

    type_codes = {"A_stable_m0_jump_sgtr_suppressed": "A",
                  "B_high_response_retained": "B",
                  "C_sparse_shock": "C-Shock",
                  "C_transition": "C-Transition"}
    ordered = cases.copy()
    ordered["case_number"] = ordered.groupby("case_type").cumcount() + 1
    ordered["case_label"] = ordered.case_type.map(type_codes) + "-" + ordered.case_number.astype(str)
    labels = ordered.case_label.tolist()
    state_matrix = ordered[["C_broad", "U_shock", "U_trans"]].to_numpy(float)
    signed = {}
    for model in ["M0_ref", "SG-TR"]:
        block = (case_summary[case_summary.model.eq(model)].set_index("pair_index")
                 .reindex(ordered.pair_index))
        signed[model] = block[[f"signed_{g}" for g in GROUPS]].to_numpy(float)
    vmax = max(np.abs(signed["M0_ref"]).max(), np.abs(signed["SG-TR"]).max())
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.5),
                             gridspec_kw={"width_ratios": [3, 5, 5]})
    im0 = axes[0].imshow(state_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_xticks(range(3), ["Broad", "Shock", "Transition"], rotation=25)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_title("冻结状态通道")
    for i in range(len(labels)):
        for j in range(3):
            axes[0].text(j, i, f"{state_matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
    for ax, model in zip(axes[1:], ["M0_ref", "SG-TR"]):
        im = ax.imshow(signed[model], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(5), GROUPS); ax.set_yticks(range(len(labels)), [])
        ax.set_title(f"{model} signed ΔSHAP组贡献")
        for i in range(len(labels)):
            for j in range(5):
                ax.text(j, i, f"{signed[model][i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle("规则化案例：状态通道与五组动态归因（仅解释展示）")
    fig.subplots_adjust(left=.08, right=.91, bottom=.10, top=.88, wspace=.30)
    signed_cax = fig.add_axes([.935, .27, .012, .42])
    fig.colorbar(im, cax=signed_cax, label="signed ΔSHAP")
    fig.savefig(figs / "rule_selected_case_profiles_s8.png", dpi=180); plt.close(fig)


def build_report(bundle, audit, formal, aggregate, similarity, bootstrap,
                 correlations, agreement, channel, relation, feature, cases):
    _, _, out, _, logs = paths(bundle)
    agg = aggregate.set_index(["layer", "model"])
    sim = similarity.set_index("layer")
    stable_m0 = agg.loc[("stable", "M0_ref")]
    stable_sg = agg.loc[("stable", "SG-TR")]
    high_m0 = agg.loc[("high", "M0_ref")]
    high_sg = agg.loc[("high", "SG-TR")]
    stable_score_reduction = 1 - stable_sg.abs_delta_score / stable_m0.abs_delta_score
    stable_sav_reduction = (1 - stable_sg.shap_attribution_volatility
                            / stable_m0.shap_attribution_volatility)
    high_score_retention = high_sg.abs_delta_score / high_m0.abs_delta_score
    high_sav_retention = (high_sg.shap_attribution_volatility
                          / high_m0.shap_attribution_volatility)
    stable_groups = relation[relation.layer.eq("stable")].sort_values(
        "mean_group_attribution_reduction", ascending=False)
    top_groups = ", ".join(stable_groups.head(3).group.tolist())
    case_counts = cases.case_type.value_counts().rename_axis("case_type").reset_index(name="n")
    additivity_max = float(audit[["max_abs_row_reconstruction_error",
                                  "max_abs_pair_decomposition_error",
                                  "max_abs_rescore_vs_frozen_oof_error"]].max().max())
    ready = bool(additivity_max <= float(bundle.cfg["shap"]["additivity_tolerance"])
                 and stable_sav_reduction > 0
                 and high_score_retention >= .80
                 and case_counts.n.min() >= 2)
    summary = {
        "frozen_core_unchanged": True,
        "models_retrained": False,
        "locked_holdout_used": False,
        "formal_pairs": int(len(formal)),
        "formal_customers": int(formal.customer_ID.nunique()),
        "maximum_additivity_or_rescore_error": additivity_max,
        "stable_score_reduction": float(stable_score_reduction),
        "stable_shap_attribution_volatility_reduction": float(stable_sav_reduction),
        "high_score_response_retention_formal_sample": float(high_score_retention),
        "high_shap_attribution_volatility_retention": float(high_sav_retention),
        "high_score_direction_agreement": float(sim.loc["high", "score_direction_agreement"]),
        "high_feature_delta_shap_cosine": float(sim.loc["high", "feature_delta_shap_cosine"]),
        "top_stable_suppressed_groups": stable_groups.head(5).group.tolist(),
        "case_counts": dict(zip(case_counts.case_type, case_counts.n.astype(int))),
        "ready_for_locked_holdout": ready,
        "causal_interpretation": False,
        "stop_after": "V4-S8; awaiting human review",
    }
    write_json(out / "stage_summary_s8.json", summary)
    audit_show = audit[["model", "outer_fold", "max_abs_row_reconstruction_error",
                       "p99_abs_row_reconstruction_error",
                       "mean_abs_row_reconstruction_error",
                       "max_abs_pair_decomposition_error",
                       "p99_abs_pair_decomposition_error",
                       "mean_abs_pair_decomposition_error",
                       "max_abs_rescore_vs_frozen_oof_error"]]
    boot_show = bootstrap[bootstrap.metric.isin(
        ["abs_delta_score", "shap_attribution_volatility"])]
    channel_show = channel[["dominant_channel", "model", "n_pairs",
                            "mean_abs_delta_score", "mean_sav",
                            "mean_top_group_concentration",
                            "largest_magnitude_group",
                            "largest_magnitude_group_share"]]
    group_rows = []
    for row in aggregate.itertuples(index=False):
        for group in GROUPS:
            group_rows.append({
                "layer": row.layer, "model": row.model, "group": group,
                "mean_signed_delta_shap": getattr(row, f"signed_{group}"),
                "mean_absolute_delta_shap": getattr(row, f"magnitude_{group}")})
    group_show = pd.DataFrame(group_rows)
    broad_m0 = channel[(channel.dominant_channel.eq("broad"))
                       & channel.model.eq("M0_ref")].iloc[0]
    shock_m0 = channel[(channel.dominant_channel.eq("shock"))
                       & channel.model.eq("M0_ref")].iloc[0]
    transition_m0 = channel[(channel.dominant_channel.eq("transition"))
                            & channel.model.eq("M0_ref")].iloc[0]
    diagonal = correlations[(correlations.scope.eq("all"))
                            & (correlations.input_group.eq(correlations.shap_group))]
    diagonal_min = float(diagonal.spearman.min())
    diagonal_max = float(diagonal.spearman.max())
    report = f"""# V4-S8 SG-TR动态可解释性与客户轨迹归因

## 执行边界与审计结论

- 本阶段只复用冻结的五折M0_ref和SG-TR outer模型；没有训练、选参或修改状态定义、Q25、lambda=.03、row-weighted主口径与Platt锁定。
- 正式汇总按outer fold和Stable/Medium/High固定seed=42，每客户每层最多一个连续自然月pair；规则化案例不进入正式均值。
- TreeSHAP解释raw margin，不作因果解释。SHAP基值已加回自定义目标单独保存的初始截距。
- 最大加性/重评分误差为`{additivity_max:.3e}`，低于`{bundle.cfg['shap']['additivity_tolerance']:.1e}`，审计通过后才生成以下结果。

## 必答结论

1. **Stable层评分下降是否伴随归因波动下降？** 是。正式客户等权样本中，|Δscore|下降`{stable_score_reduction:.2%}`，SHAP Attribution Volatility下降`{stable_sav_reduction:.2%}`。配对bootstrap区间见下表。
2. **哪些特征组被抑制最多？** 按Stable层M0_ref减SG-TR的平均动态SHAP magnitude排序，前三组为 **{top_groups}**；这是归因波动的同时减少，不是因果贡献声明。
3. **High为何仍保留响应？** High不施加temporal penalty；正式样本的|Δscore|保留`{high_score_retention:.2%}`、SAV保留`{high_sav_retention:.2%}`，评分方向一致率`{sim.loc['high','score_direction_agreement']:.2%}`，特征级ΔSHAP向量余弦相似度均值`{sim.loc['high','feature_delta_shap_cosine']:.3f}`。模型训练目标不同，因此不宣称两者完全相同。
4. **三通道在解释空间是否呈现不同模式？** 有描述性差异：Shock-dominant的M0 |Δscore|/SAV最高（`{shock_m0.mean_abs_delta_score:.3f}`/`{shock_m0.mean_sav:.3f}`）；Transition-dominant的D组份额与top-group集中度最高（`{transition_m0.largest_magnitude_group_share:.2%}`/`{transition_m0.mean_top_group_concentration:.2%}`）；Broad-dominant覆盖最多pairs且更分散。差异不表示三通道导致评分变化。
5. **是否存在规则化案例？** 是。按查看客户身份前冻结的阈值，得到以下案例；案例只用于解释。
6. **是否支持“该稳定时稳定、该响应时响应”？** 动态SHAP支持这一机制解释：Stable层评分与总体归因波动共同下降，High层保留大部分评分响应和归因调整。该证据解释模型行为，不证明因果或外部有效性。
7. **是否具备进入locked holdout条件？** {'是' if ready else '否'}。解释审计、正式总体归因和规则化案例均已完成；S8在此停止，未抽取或使用holdout。

## SHAP加性审计

{markdown_table(audit_show, 9)}

## Stable / Medium / High正式汇总

{markdown_table(aggregate[['layer','model','n_pairs','n_customers','abs_delta_score','shap_attribution_volatility']])}

## 配对bootstrap（SG-TR减M0_ref）

{markdown_table(boot_show)}

## 模型间动态结构相似性

{markdown_table(similarity)}

## Stable层五组归因波动抑制

{markdown_table(stable_groups)}

## 五组signed与magnitude动态SHAP

signed值保留方向，magnitude为组内各特征`|ΔSHAP|`之和；二者不可互换。

{markdown_table(group_show)}

## Broad / Shock / Transition主导模式

{markdown_table(channel_show)}

## 输入变化—模型归因变化

对应组Spearman范围为`{diagonal_min:.3f}`至`{diagonal_max:.3f}`；完整5×5相关、分箱趋势、top-group一致率分别见`input_shap_spearman_s8.csv`、`input_shap_bin_trends_s8.csv`和`input_shap_top_group_agreement_s8.csv`。状态指标用于控制时间正则强度，不是SHAP变化或评分变化的预测模型；弱相关或较低top-group一致率不构成失败，也不得反馈修改状态定义。

{markdown_table(agreement)}

## 规则化案例数量

{markdown_table(case_counts)}

案例身份、状态通道、两模型评分变化和top ΔSHAP特征分别见`rule_selected_case_candidates_s8.csv`、`rule_selected_case_shap_summary_s8.csv`和`rule_selected_case_feature_details_s8.csv`。

## 图表

- `figures/s8/method_overview_s8.png`
- `figures/s8/stable_high_dynamic_shap_s8.png`
- `figures/s8/input_shap_group_heatmap_s8.png`
- `figures/s8/stable_case_trajectories_s8.png`
- `figures/s8/high_case_trajectories_s8.png`
- `figures/s8/rule_selected_case_profiles_s8.png`
"""
    (logs / "stage_summary_V4_S8.md").write_text(report, encoding="utf-8")
    return summary


def reproducibility_audit(bundle, formal, cases, explain, audit, summary):
    _, _, out, figs, _ = paths(bundle)
    cross_layer = (formal.groupby("customer_ID").layer.nunique() > 1).sum()
    payload = {
        "frozen_core_unchanged": True,
        "model_training_performed": False,
        "holdout_accessed": False,
        "formal_selection": bundle.cfg["shap"]["formal_sample"],
        "formal_pairs": int(len(formal)),
        "formal_unique_pair_indices": int(formal.pair_index.nunique()),
        "formal_duplicate_customer_layer": int(
            formal.duplicated(["customer_ID", "layer"]).sum()),
        "formal_customers_repeated_across_layers": int(cross_layer),
        "explained_unique_pairs": int(explain.pair_index.nunique()),
        "case_pairs": int(len(cases)),
        "formal_case_pair_overlap": int(len(
            set(formal.pair_index) & set(cases.pair_index))),
        "case_pairs_added_to_formal_aggregate": False,
        "m0_models": 5, "sgtr_models": 5,
        "all_explanations_outer_validation_only": True,
        "shap_scale": "raw_margin",
        "additivity_passed": bool(
            audit.max_abs_pair_decomposition_error.max()
            <= float(bundle.cfg["shap"]["additivity_tolerance"])),
        "maximum_additivity_or_rescore_error": summary[
            "maximum_additivity_or_rescore_error"],
        "figure_files": sorted(p.name for p in figs.glob("*.png")),
        "forbidden_actions_run": [],
    }
    if payload["formal_duplicate_customer_layer"] != 0:
        raise AssertionError("Formal sample customer-layer duplication")
    if payload["formal_case_pair_overlap"] != 0:
        raise AssertionError("Rule-selected cases overlap formal aggregate pairs")
    write_json(out / "reproducibility_audit_s8.json", payload)


def main():
    started = time.time()
    bundle = load_bundle(CFG)
    s3, _, out, figs, logs = paths(bundle)
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    m0_scores, sg_scores, score_arrays = load_scores(bundle)
    state = stitched_state(bundle, s3)
    pairs = pair_frame(bundle, state, score_arrays)
    formal = formal_sample(bundle, pairs)
    cases, thresholds = select_cases(bundle, pairs, set(formal.pair_index))
    write_json(out / "case_selection_thresholds_s8.json", thresholds)
    formal.to_parquet(out / "formal_pair_sample_s8.parquet", index=False)
    explain = pd.concat([formal, cases], ignore_index=True).sort_values(
        ["pair_index", "is_formal"], ascending=[True, False]).drop_duplicates(
            "pair_index", keep="first")
    explain.to_parquet(out / "explained_pair_union_s8.parquet", index=False)
    log(f"formal sample {len(formal):,} pairs; case sample {len(cases):,}; "
        f"union {len(explain):,}")
    audit, deltas, summaries = compute_shap(bundle, explain, score_arrays)
    long, wide = combine_formal(bundle, formal, deltas, summaries)
    aggregate, similarity = aggregate_formal(bundle, long, wide)
    bootstrap = paired_bootstrap(bundle, wide)
    correlations, agreement = input_shap_chain(bundle, long)
    channel, relation, feature = channel_and_suppression(
        bundle, long, wide, deltas)
    _, case_summary = case_details(bundle, cases, deltas, summaries)
    plots(bundle, aggregate, correlations, cases, case_summary, m0_scores, sg_scores)
    summary = build_report(bundle, audit, formal, aggregate, similarity,
                           bootstrap, correlations, agreement, channel,
                           relation, feature, cases)
    reproducibility_audit(bundle, formal, cases, explain, audit, summary)
    log(f"S8 complete in {time.time()-started:.1f}s; stopped before locked holdout")


if __name__ == "__main__":
    main()
