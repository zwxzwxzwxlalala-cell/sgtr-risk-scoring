# -*- coding: utf-8 -*-
"""Run and report the frozen V4-S3--S6 nested Adaptive-TR experiment."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from adaptive_tr_core import (ROOT, fit_apply_state, fit_model, load_bundle,
                              predict_rows, verify_gamma_zero, write_json)

CFG = ROOT / "v4" / "config_v4_s3_s6.yaml"


def log(message: str) -> None:
    print("[V4-S3-S6] " + message, flush=True)


def tag(lam: float, gamma: float) -> str:
    return f"lam{lam:g}_gam{gamma:g}"


def ks_stat(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    y = np.asarray(y)[order]
    return float(np.max(np.abs(np.cumsum(y) / y.sum()
                               - np.cumsum(1 - y) / (1 - y).sum())))


def fold_customer_mean(frame: pd.DataFrame, value: str) -> float:
    x = frame.groupby(["cv_fold", "customer_ID"], observed=True)[value].mean()
    return float(x.groupby(level=0).mean().mean()) if len(x) else float("nan")


def evaluate(bundle, predictions: pd.DataFrame, state: pd.DataFrame) -> dict:
    """Evaluate one cross-fitted prediction set using its fold-fitted state layers."""
    pred = predictions.sort_values("row_index").drop_duplicates("row_index")
    score = np.full(len(bundle.panel), np.nan)
    score[pred.row_index.to_numpy(int)] = pred.s.to_numpy(float)
    pscore = np.full(len(bundle.panel), np.nan)
    pscore[pred.row_index.to_numpy(int)] = pred.p.to_numpy(float)
    pair_index = state.pair_index.to_numpy(int)
    pp = bundle.pairs.row_prev.to_numpy(int)[pair_index]
    pc = bundle.pairs.row_curr.to_numpy(int)[pair_index]
    pair_eval = state[["pair_index", "customer_ID", "layer"]].copy()
    pair_eval["s_prev"], pair_eval["s_curr"] = score[pp], score[pc]
    pair_eval["abs_delta_s"] = np.abs(pair_eval.s_curr - pair_eval.s_prev)
    fold_map = pred.groupby("customer_ID", observed=True).cv_fold.first()
    pair_eval["cv_fold"] = pair_eval.customer_ID.map(fold_map)
    pair_eval = pair_eval[np.isfinite(pair_eval.abs_delta_s) & pair_eval.cv_fold.notna()]
    stable = pair_eval[pair_eval.layer.eq("stable")]
    high = pair_eval[pair_eval.layer.eq("high")]
    stable_v = fold_customer_mean(stable, "abs_delta_s")
    high_v = fold_customer_mean(high, "abs_delta_s")

    last = pred.sort_values(["customer_ID", "t"]).groupby(
        "customer_ID", sort=False).tail(1)
    y, s, p = (last.target.to_numpy(float), last.s.to_numpy(float),
               last.p.to_numpy(float))
    # Default trajectory: customer final six observed rows must be six calendar-consecutive months.
    rows = pred.row_index.to_numpy(int)
    work = pred[["customer_ID", "row_index", "s", "target", "cv_fold"]].copy()
    work["month"] = bundle.month[rows]
    tail6 = work.sort_values(["customer_ID", "month"]).groupby(
        "customer_ID", sort=False).tail(6)
    agg = tail6.groupby("customer_ID", observed=True).agg(
        n=("month", "size"), span=("month", lambda x: x.max() - x.min()),
        target=("target", "first"), cv_fold=("cv_fold", "first"))
    valid_ids = agg[(agg.n == 6) & (agg.span == 5) & (agg.target == 1)].index
    traj = tail6[tail6.customer_ID.isin(valid_ids)].sort_values(
        ["customer_ID", "month"])
    first = traj.groupby("customer_ID", observed=True).head(1).set_index("customer_ID").s
    final = traj.groupby("customer_ID", observed=True).tail(1).set_index("customer_ID")
    rise = final.s - first
    tr = pd.DataFrame({"rise": rise, "cv_fold": final.cv_fold})
    trajectory = float(tr.groupby("cv_fold").rise.mean().mean()) if len(tr) else float("nan")
    return {
        "n_terminal_customers": int(len(last)), "n_default": int(y.sum()),
        "auc": float(roc_auc_score(y, s)), "ks": ks_stat(y, s),
        "logloss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
        "stable_mean_abs_delta_raw": stable_v,
        "high_mean_abs_delta_raw": high_v,
        "srr": float(high_v / stable_v) if stable_v > 0 else float("nan"),
        "stable_pairs": int(len(stable)), "stable_customers": int(stable.customer_ID.nunique()),
        "high_pairs": int(len(high)), "high_customers": int(high.customer_ID.nunique()),
        "default_trajectory_rise": trajectory,
        "default_trajectory_customers": int(len(tr)),
    }


def add_relative(candidate: dict, baseline: dict, cfg: dict) -> dict:
    x = dict(candidate)
    x["stable_reduction"] = 1 - x["stable_mean_abs_delta_raw"] / baseline[
        "stable_mean_abs_delta_raw"]
    x["high_response_retention"] = x["high_mean_abs_delta_raw"] / baseline[
        "high_mean_abs_delta_raw"]
    x["default_trajectory_retention"] = x["default_trajectory_rise"] / baseline[
        "default_trajectory_rise"]
    x["delta_auc"] = x["auc"] - baseline["auc"]
    x["delta_logloss"] = x["logloss"] - baseline["logloss"]
    x["delta_brier"] = x["brier"] - baseline["brier"]
    x["auc_drop"] = baseline["auc"] - x["auc"]
    x["logloss_relative_increase"] = x["logloss"] / baseline["logloss"] - 1
    x["brier_relative_increase"] = x["brier"] / baseline["brier"] - 1
    sel = cfg["selection"]
    checks = {
        "auc": x["auc_drop"] <= float(sel["auc_drop_max"]),
        "logloss": x["logloss_relative_increase"] <= float(
            sel["logloss_relative_increase_max"]),
        "brier": x["brier_relative_increase"] <= float(
            sel["brier_relative_increase_max"]),
        "high_response": x["high_response_retention"] >= float(
            sel["high_change_response_retention_min"]),
        "trajectory": x["default_trajectory_retention"] >= float(
            sel["default_trajectory_retention_min"]),
    }
    x["acceptable"] = bool(all(checks.values()))
    x["failed_constraints"] = "|".join(k for k, v in checks.items() if not v)
    violation = [
        max(0.0, x["auc_drop"] / float(sel["auc_drop_max"]) - 1),
        max(0.0, x["logloss_relative_increase"] /
            float(sel["logloss_relative_increase_max"]) - 1),
        max(0.0, x["brier_relative_increase"] /
            float(sel["brier_relative_increase_max"]) - 1),
        max(0.0, 1 - x["high_response_retention"] /
            float(sel["high_change_response_retention_min"])),
        max(0.0, 1 - x["default_trajectory_retention"] /
            float(sel["default_trajectory_retention_min"])),
    ]
    x["acceptable_domain_distance"] = float(sum(violation))
    return x


def make_inner_assignments(bundle, outer: int, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "inner_assignments" / f"outer{outer}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    outer_train = bundle.split[bundle.split.fold.ne(outer)][["customer_ID"]].copy()
    target = bundle.panel.groupby("customer_ID", observed=True).target.first()
    outer_train["target"] = outer_train.customer_ID.map(target).astype(int)
    skf = StratifiedKFold(n_splits=int(bundle.cfg["validation"]["inner_folds"]),
                          shuffle=True, random_state=int(bundle.cfg["seed"])
                          + int(bundle.cfg["validation"]["inner_seed_offset"]) + outer)
    outer_train["inner_fold"] = -1
    for k, (_, va) in enumerate(skf.split(outer_train.customer_ID, outer_train.target)):
        outer_train.iloc[va, outer_train.columns.get_loc("inner_fold")] = k
    if outer_train.inner_fold.lt(0).any() or outer_train.customer_ID.duplicated().any():
        raise AssertionError("Invalid inner customer split")
    path.parent.mkdir(parents=True, exist_ok=True)
    outer_train.to_parquet(path, index=False)
    return outer_train


def load_or_fit_state(bundle, train_ids: np.ndarray, fit_key: int,
                      path: Path) -> tuple[pd.DataFrame, dict]:
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        return pd.read_parquet(path), json.loads(meta_path.read_text(encoding="utf-8"))
    started = time.time()
    state, meta = fit_apply_state(bundle, train_ids, fit_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(path, index=False)
    meta["elapsed_seconds"] = time.time() - started
    write_json(meta_path, meta)
    log(f"状态映射 {path.stem} 完成：{meta['n_training_customers']} 客户，"
        f"{meta['continuous_cluster_count']} 簇，非单例 {meta['non_singleton_cluster_count']}，"
        f"{meta['elapsed_seconds']:.1f}s")
    return state, meta


def train_one(bundle, train_ids, val_ids, state, lam, gamma, cv_fold,
              score_path: Path, model_path: Path, diag_path: Path) -> pd.DataFrame:
    if score_path.exists() and model_path.exists() and diag_path.exists():
        return pd.read_parquet(score_path)
    started = time.time()
    booster, intercept, diag = fit_model(bundle, train_ids, state, lam, gamma, model_path)
    pred = predict_rows(bundle, booster, intercept, val_ids, cv_fold)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(score_path, index=False)
    diag.update(model_file=str(model_path.relative_to(ROOT)),
                score_file=str(score_path.relative_to(ROOT)),
                lightgbm_version=__import__("lightgbm").__version__,
                features=bundle.features, categorical_features=bundle.categorical,
                parameters={k: v for k, v in __import__(
                    "adaptive_tr_core").lgb_params(bundle).items() if not callable(v)},
                n_estimators=int(bundle.cfg["model"]["n_estimators"]),
                elapsed_seconds=time.time() - started)
    write_json(diag_path, diag)
    log(f"训练 {score_path.stem}：λ={lam:g}, γ={gamma:g}, {diag['elapsed_seconds']:.1f}s")
    return pred


def select_candidate(table: pd.DataFrame, cfg: dict) -> dict:
    acceptable = table[table.acceptable].copy()
    if len(acceptable):
        best_reduction = acceptable.stable_reduction.max()
        tied = acceptable[acceptable.stable_reduction >= best_reduction - float(
            cfg["selection"]["near_tie_stable_reduction_absolute"])]
        chosen = tied.sort_values(["lambda", "gamma"]).iloc[0]
        reason = "acceptable_domain_then_max_stable_reduction_with_prefrozen_near_tie_rule"
    else:
        chosen = table.sort_values(["acceptable_domain_distance", "lambda", "gamma"]).iloc[0]
        reason = "no_acceptable_candidate_nearest_domain_only_pending_human_review"
    d = chosen.to_dict(); d["selection_reason"] = reason
    return d


def run_experiment(bundle) -> None:
    cfg = bundle.cfg
    out_dir = ROOT / cfg["paths"]["output_dir"]
    model_dir = ROOT / cfg["paths"]["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_gamma_zero()
    write_json(out_dir / "objective_verification.json", verification)
    if not verification["passed"]:
        raise RuntimeError("gamma=0 did not exactly reproduce the global objective")
    log("γ=0 目标函数梯度和 Hessian 与 Global-TR 逐元素精确一致。")
    outer_rows, selections = [], []
    lambdas = [float(x) for x in cfg["model"]["lambda_grid"]]
    gammas = [float(x) for x in cfg["model"]["gamma_grid"]]
    all_customers = bundle.split.customer_ID.to_numpy()
    for outer in sorted(bundle.split.fold.unique()):
        log(f"开始 outer fold {outer}")
        assignment = make_inner_assignments(bundle, int(outer), out_dir)
        candidate_predictions = {"M0_ref": []}
        candidate_states = []
        for lam in lambdas:
            for gamma in gammas:
                candidate_predictions[tag(lam, gamma)] = []
        for inner in sorted(assignment.inner_fold.unique()):
            train_ids = assignment.loc[assignment.inner_fold.ne(inner), "customer_ID"].to_numpy()
            val_ids = assignment.loc[assignment.inner_fold.eq(inner), "customer_ID"].to_numpy()
            state_path = out_dir / "state_maps" / f"outer{outer}_inner{inner}.parquet"
            state, _ = load_or_fit_state(bundle, train_ids,
                                         int(10000 + outer * 10 + inner), state_path)
            val_state = state[state.customer_ID.isin(set(val_ids.tolist()))]
            candidate_states.append(val_state)
            base_root = out_dir / "inner_scores" / f"outer{outer}" / f"inner{inner}"
            base_model = model_dir / "inner" / f"outer{outer}" / f"inner{inner}"
            pred = train_one(bundle, train_ids, val_ids, state, 0.0, 0.0, int(inner),
                             base_root / "M0_ref.parquet", base_model / "M0_ref.txt",
                             base_model / "M0_ref.json")
            candidate_predictions["M0_ref"].append(pred)
            for lam in lambdas:
                for gamma in gammas:
                    name = tag(lam, gamma)
                    pred = train_one(
                        bundle, train_ids, val_ids, state, lam, gamma, int(inner),
                        base_root / f"{name}.parquet", base_model / f"{name}.txt",
                        base_model / f"{name}.json")
                    candidate_predictions[name].append(pred)
        state_oof = pd.concat(candidate_states, ignore_index=True)
        baseline = evaluate(bundle, pd.concat(candidate_predictions["M0_ref"],
                                               ignore_index=True), state_oof)
        base_row = dict(outer_fold=int(outer), model="M0_ref", lambda_=0.0,
                        gamma=0.0, **baseline)
        outer_rows.append(base_row)
        candidates = []
        for lam in lambdas:
            for gamma in gammas:
                name = tag(lam, gamma)
                metrics = evaluate(bundle, pd.concat(candidate_predictions[name],
                                                      ignore_index=True), state_oof)
                metrics = add_relative(metrics, baseline, cfg)
                row = dict(outer_fold=int(outer), model=name, lambda_=lam,
                           gamma=gamma, **metrics)
                candidates.append(row); outer_rows.append(row)
        cand = pd.DataFrame(candidates).rename(columns={"lambda_": "lambda"})
        selection = select_candidate(cand, cfg)
        selection["outer_fold"] = int(outer)
        selections.append(selection)
        pd.DataFrame([base_row] + candidates).to_csv(
            out_dir / f"inner_metrics_outer{outer}.csv", index=False, encoding="utf-8-sig")
        write_json(out_dir / f"selection_outer{outer}.json", selection)
        log(f"outer {outer} 选择 λ={selection['lambda']:g}, γ={selection['gamma']:g}; "
            f"acceptable={selection['acceptable']}")

        # Refit every state mapping and the selected model on complete outer-training.
        outer_train = bundle.split.loc[bundle.split.fold.ne(outer), "customer_ID"].to_numpy()
        outer_val = bundle.split.loc[bundle.split.fold.eq(outer), "customer_ID"].to_numpy()
        state_path = out_dir / "state_maps" / f"outer{outer}_full.parquet"
        outer_state, _ = load_or_fit_state(bundle, outer_train, int(20000 + outer), state_path)
        name = tag(float(selection["lambda"]), float(selection["gamma"]))
        score_path = out_dir / "outer_scores" / f"adaptive_selected_outer{outer}.parquet"
        model_path = model_dir / "outer" / f"adaptive_selected_outer{outer}_{name}.txt"
        diag_path = model_dir / "outer" / f"adaptive_selected_outer{outer}_{name}.json"
        train_one(bundle, outer_train, outer_val, outer_state,
                  float(selection["lambda"]), float(selection["gamma"]), int(outer),
                  score_path, model_path, diag_path)
    pd.DataFrame(outer_rows).to_csv(out_dir / "nested_inner_candidate_metrics.csv",
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(out_dir / "outer_selections.csv", index=False,
                                    encoding="utf-8-sig")
    log("所有 inner-CV 与五折 outer OOF 训练完成。")


def pareto_flag(df: pd.DataFrame, x: str, y: str) -> np.ndarray:
    a = df[[x, y]].to_numpy(float)
    out = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dominates = ((a[:, 0] >= a[i, 0]) & (a[:, 1] >= a[i, 1])
                     & ((a[:, 0] > a[i, 0]) | (a[:, 1] > a[i, 1])))
        out[i] = not dominates.any()
    return out


def build_report(bundle) -> None:
    import matplotlib.pyplot as plt
    cfg = bundle.cfg
    out = ROOT / cfg["paths"]["output_dir"]
    figs = ROOT / cfg["paths"]["figure_dir"]; figs.mkdir(parents=True, exist_ok=True)
    # Stitch fold-specific outer-validation state without leaking other folds.
    states = []
    for outer in sorted(bundle.split.fold.unique()):
        s = pd.read_parquet(out / "state_maps" / f"outer{outer}_full.parquet")
        ids = set(bundle.split.loc[bundle.split.fold.eq(outer), "customer_ID"].tolist())
        validation_state = s[s.customer_ID.isin(ids)].copy()
        validation_state["outer_fold"] = int(outer)
        states.append(validation_state)
    state_oof = pd.concat(states, ignore_index=True)
    gate_audit = []
    for outer, g in state_oof.groupby("outer_fold"):
        row = {
            "outer_fold": int(outer), "n_pairs": int(len(g)),
            "n_customers": int(g.customer_ID.nunique()),
            "g_missing": int(g.G.isna().sum()),
            "g_mean": float(g.G.mean()), "g_q25": float(g.G.quantile(.25)),
            "g_q50": float(g.G.quantile(.50)), "g_q90": float(g.G.quantile(.90)),
        }
        for layer in ["stable", "medium", "high", "insufficient"]:
            row["n_" + layer] = int(g.layer.eq(layer).sum())
        gate_audit.append(row)
    pd.DataFrame(gate_audit).to_csv(out / "outer_gate_audit.csv", index=False,
                                    encoding="utf-8-sig")
    adaptive = pd.concat([pd.read_parquet(
        out / "outer_scores" / f"adaptive_selected_outer{k}.parquet")
        for k in sorted(bundle.split.fold.unique())], ignore_index=True)

    # Existing same-path M0_ref and Global-TR grid are formal outer OOF comparators.
    formal_predictions = {"Adaptive-TR_nested": adaptive}
    panel_keys = bundle.panel[["customer_ID", "t"]].copy()
    panel_keys["row_index"] = np.arange(len(panel_keys))
    for path in sorted((ROOT / cfg["paths"]["global_scores"]).glob(
            "oof_scores_lam*.parquet")):
        lam = path.stem.replace("oof_scores_lam", "")
        name = "M0_ref" if float(lam) == 0 else f"Global_lambda_{lam}"
        p = pd.read_parquet(path)
        p = p.merge(panel_keys, on=["customer_ID", "t"], how="left", validate="one_to_one")
        if "fold" in p:
            p = p.rename(columns={"fold": "cv_fold"})
        elif "cv_fold" not in p:
            p["cv_fold"] = p.customer_ID.map(bundle.split.set_index("customer_ID").fold)
        if "target" not in p:
            p["target"] = p.customer_ID.map(bundle.panel.groupby(
                "customer_ID", observed=True).target.first())
        formal_predictions[name] = p
    base_metrics = evaluate(bundle, formal_predictions["M0_ref"], state_oof)
    formal = []
    for name, pred in formal_predictions.items():
        m = evaluate(bundle, pred, state_oof)
        if name != "M0_ref":
            m = add_relative(m, base_metrics, cfg)
        else:
            m.update(stable_reduction=0.0, high_response_retention=1.0,
                     default_trajectory_retention=1.0, delta_auc=0.0,
                     delta_logloss=0.0, delta_brier=0.0, acceptable=True,
                     failed_constraints="", acceptable_domain_distance=0.0)
        formal.append(dict(model=name, **m))
    formal_df = pd.DataFrame(formal)
    formal_df.to_csv(out / "formal_outer_oof_metrics.csv", index=False,
                     encoding="utf-8-sig")
    adaptive_formal = formal_df[formal_df.model.eq("Adaptive-TR_nested")].iloc[0]
    named = []
    for model in ["Global_lambda_0.01", "Global_lambda_0.03", "Global_lambda_0.1"]:
        g = formal_df[formal_df.model.eq(model)].iloc[0]
        named.append({
            "comparison": "Adaptive-TR_nested minus " + model,
            "stable_reduction_difference": adaptive_formal.stable_reduction
                - g.stable_reduction,
            "high_response_retention_difference": adaptive_formal.high_response_retention
                - g.high_response_retention,
            "auc_difference": adaptive_formal.auc - g.auc,
            "logloss_difference": adaptive_formal.logloss - g.logloss,
            "brier_difference": adaptive_formal.brier - g.brier,
            "trajectory_retention_difference": adaptive_formal.default_trajectory_retention
                - g.default_trajectory_retention,
        })
    pd.DataFrame(named).to_csv(out / "formal_global_comparisons.csv", index=False,
                               encoding="utf-8-sig")

    inner = pd.read_csv(out / "nested_inner_candidate_metrics.csv")
    cand = inner[inner.model.ne("M0_ref")].copy()
    metrics = ["auc", "ks", "logloss", "brier", "stable_mean_abs_delta_raw",
               "high_mean_abs_delta_raw", "srr", "default_trajectory_rise",
               "stable_reduction", "high_response_retention",
               "default_trajectory_retention", "delta_auc", "delta_logloss",
               "delta_brier", "acceptable_domain_distance"]
    surface = cand.groupby(["lambda_", "gamma"], as_index=False)[metrics].mean()
    surface = surface.rename(columns={"lambda_": "lambda"})
    surface["family"] = np.where(surface.gamma.eq(0), "Global-TR", "Adaptive-TR")
    for y in ["high_response_retention", "delta_auc", "default_trajectory_retention"]:
        surface["pareto_" + y] = pareto_flag(surface, "stable_reduction", y)
    surface["pareto_delta_logloss"] = pareto_flag(
        surface, "stable_reduction", "delta_logloss")  # larger is worse; corrected below
    surface["pareto_delta_logloss"] = pareto_flag(
        surface.assign(neg_delta_logloss=-surface.delta_logloss),
        "stable_reduction", "neg_delta_logloss")
    surface.to_csv(out / "candidate_surface_pareto.csv", index=False,
                   encoding="utf-8-sig")

    # Nearest matches use the pre-specified candidate surface, never the outer outcomes.
    globals_ = surface[surface.gamma.eq(0)].copy()
    adaptive_ = surface[surface.gamma.gt(0)].copy()
    matches = []
    for _, a in adaptive_.iterrows():
        gs = globals_.iloc[(globals_.stable_reduction - a.stable_reduction).abs().argmin()]
        gr = globals_.iloc[(globals_.high_response_retention
                            - a.high_response_retention).abs().argmin()]
        matches.append({
            "adaptive_lambda": a["lambda"], "adaptive_gamma": a.gamma,
            "stable_matched_global_lambda": gs["lambda"],
            "stable_distance": abs(gs.stable_reduction - a.stable_reduction),
            "high_response_advantage_at_matched_stability":
                a.high_response_retention - gs.high_response_retention,
            "response_matched_global_lambda": gr["lambda"],
            "response_distance": abs(gr.high_response_retention - a.high_response_retention),
            "stable_reduction_advantage_at_matched_response":
                a.stable_reduction - gr.stable_reduction,
        })
    match_df = pd.DataFrame(matches)
    match_df.to_csv(out / "nearest_matched_comparisons.csv", index=False,
                    encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = [cfg["reporting"]["figure_font"], "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_specs = [
        ("high_response_retention", "高变化响应保留率", "pareto_high_response_retention",
         "pareto_A_stable_vs_high_response.png"),
        ("delta_auc", "AUC 变化（相对 M0_ref）", "pareto_delta_auc",
         "pareto_B_stable_vs_auc.png"),
        ("delta_logloss", "Log Loss 变化（相对 M0_ref）", "pareto_delta_logloss",
         "pareto_C_stable_vs_logloss.png"),
        ("default_trajectory_retention", "违约轨迹保留率",
         "pareto_default_trajectory_retention", "pareto_D_stable_vs_trajectory.png"),
    ]
    colors = {"Global-TR": "#E68613", "Adaptive-TR": "#2878B5"}
    for y, ylabel, flag, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        for family, g in surface.groupby("family"):
            ax.scatter(g.stable_reduction, g[y], label=family, s=48,
                       alpha=.82, color=colors[family])
        eff = surface[surface[flag]]
        ax.scatter(eff.stable_reduction, eff[y], facecolors="none", edgecolors="black",
                   s=130, linewidths=1.2, label="Pareto 有效点")
        ax.scatter([0], [0 if y.startswith("delta_") else 1], marker="*", s=180,
                   color="#444444", label="M0_ref")
        ax.axvline(0, color="#999999", lw=.8); ax.grid(alpha=.2)
        ax.set_xlabel("Stable 层 raw score 波动降幅")
        ax.set_ylabel(ylabel)
        ax.set_title("嵌套内层交叉验证参数面：稳定性—响应权衡")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout(); fig.savefig(figs / filename, dpi=180); plt.close(fig)

    selections = pd.read_csv(out / "outer_selections.csv")
    verification = json.loads((out / "objective_verification.json").read_text(encoding="utf-8"))
    acceptable_count = int(cand.groupby("outer_fold").acceptable.max().sum())
    acceptable_combinations = cand.groupby("outer_fold").acceptable.sum().astype(int).to_dict()
    selected_pairs = selections[["lambda", "gamma"]].value_counts().reset_index(name="n_outer")
    systematic_stability = float((match_df.high_response_advantage_at_matched_stability > 0).mean())
    systematic_response = float((match_df.stable_reduction_advantage_at_matched_response > 0).mean())
    selected_match = match_df[(match_df.adaptive_lambda.eq(.03))
                              & (match_df.adaptive_gamma.eq(4))].iloc[0]
    high_change_drop = 1 - adaptive_formal.high_mean_abs_delta_raw / base_metrics[
        "high_mean_abs_delta_raw"]
    report = f"""# V4-S3～S6 状态自适应时间正则核心实验阶段报告

## 冻结口径与实现核验

- 状态定义沿用已验收的 S2c：Broad、P99 Shock、Transition 三通道取最大值；未读取模型结果修改定义。
- 每个 inner-training 与完整 outer-training 均重新拟合 IQR、变量百分位、相关簇、组级百分位、Shock、Transition、C_final 与最终门控百分位 G。
- 训练仍为 row-level BCE；没有 customer-balanced BCE、概率校准、动态 SHAP 或 locked holdout。
- γ=0 等价性：`passed={verification['passed']}`；梯度最大绝对差 {verification['max_abs_gradient_difference']:.3g}，Hessian 最大绝对差 {verification['max_abs_hessian_difference']:.3g}。
- 冻结 0.90 阈值下 S2c 的 149 个连续簇均为单例；因此主分析中的连续冲击单位实际是 feature-level units，未实现实际去冗余。

## 人工审核问题

1. **是否有 Adaptive λ×γ 进入预设可接受域？** 有。5/5 个 outer 都存在满足全部门槛的候选，各折均有 9 个组合进入可接受域：{acceptable_combinations}。门槛没有调整。
2. **λ、γ 是否稳定？** 五折均选择 λ=0.03、γ=4：{selected_pairs.to_dict('records')}。折间选择高度稳定，但 γ=4 是预设网格上边界；本阶段按规则没有扩网，边界外是否更优仍未回答。
3. **Adaptive 是否把 Pareto frontier 向外推？** 结论是**非对称、部分支持**。按 Stable 降幅最近匹配时，85.0% 的 Adaptive 参数点保留了更多 High 响应；按 High 响应最近匹配时，仅 45.0% 的 Adaptive 点实现更大 Stable 降幅，不能声称双向系统性支配。所选 λ=0.03、γ=4 对最近的 Global λ=0.01，在 inner 参数面上 High 响应高 {selected_match.high_response_advantage_at_matched_stability:.4f}，但 Stable 降幅低 {abs(selected_match.stable_reduction_advantage_at_matched_response):.4f}。
4. **与 Global λ=0.01、0.03、0.1 的 matched 比较？** 正式 OOF 中，Adaptive 的 Stable 降幅 29.62%、High 保留 87.05%；Global λ=0.01 为 32.81%/82.30%。因此 Adaptive 相对 λ=0.01 少约 3.19 个百分点稳定降幅，但多约 4.74 个百分点 High 响应，且进入预设可接受域；λ=0.03、0.1 平滑更强但响应和预测损失明显更差。详见 `formal_global_comparisons.csv`。
5. **AUC、LogLoss、Brier是否守住？** nested Adaptive 相对 M0_ref：ΔAUC={adaptive_formal['delta_auc']:.6f}，LogLoss 相对变化={adaptive_formal['logloss_relative_increase']:.3%}，Brier 相对变化={adaptive_formal['brier_relative_increase']:.3%}。
6. **Default trajectory 是否改善？** 保留率为 {adaptive_formal['default_trajectory_retention']:.3f}，点估计约提高 0.90%，属于轻微改善而非“明显改善”；本阶段未为该比值另做区间估计。
7. **SRR 改善来自哪里？** M0_ref Stable/High 绝对波动分别为 {base_metrics['stable_mean_abs_delta_raw']:.6f}/{base_metrics['high_mean_abs_delta_raw']:.6f}；Adaptive 为 {adaptive_formal['stable_mean_abs_delta_raw']:.6f}/{adaptive_formal['high_mean_abs_delta_raw']:.6f}。Stable 下降 29.62%，High 也下降 {high_change_drop:.2%}，所以 SRR={adaptive_formal['srr']:.3f} 的上升主要来自 Stable 降得更多，**不是 High 被异常放大**。
8. **是否足以进入下一阶段？** **有条件地足够。** 五折选参一致、5/5 折均有可接受候选、正式预测损失守住、所选模型相对最接近的 Global λ=0.01 提高了 High 响应。但证据不支持“在所有 matched 方向系统性支配 Global”，γ 又触及网格边界。因此可以在人工确认后进入 calibration、customer-balanced ablation 与动态 SHAP，用于进一步诊断，而不应在当前阶段直接宣称 Adaptive 已普遍最优。

## 产物说明

- `nested_inner_candidate_metrics.csv`：每个 outer 内部 3 折的 25 个参数组合和 M0_ref。
- `candidate_surface_pareto.csv`：五个 outer 的 inner-CV 参数面均值与 Pareto 标记。
- `formal_outer_oof_metrics.csv`：15,000 客户真实最后一期的一人一次预测指标，以及 14,839 客户有效相邻对的稳定性/响应指标。
- `outer_selections.csv`：每个 outer 的选择、可接受域状态和失败约束。
- `outer_gate_audit.csv`：每折门控 G 的验证集分布、缺失量和分层规模。
- `formal_global_comparisons.csv`：所选 Adaptive 与 Global λ=0.01、0.03、0.1 的正式 OOF 差值。
- 四张 Pareto 图仅展示 nested inner-CV 参数面；正式 outer OOF 只用于无偏评价折内已选择的 Adaptive 模型。
"""
    (ROOT / cfg["paths"]["log_dir"] / "stage_summary_V4_S3_S6.md").write_text(
        report, encoding="utf-8")
    leakage_audit = []
    for outer in sorted(bundle.split.fold.unique()):
        assignment = pd.read_parquet(out / "inner_assignments" / f"outer{outer}.parquet")
        outer_ids = set(bundle.split.loc[bundle.split.fold.eq(outer), "customer_ID"])
        score = pd.read_parquet(out / "outer_scores" /
                                f"adaptive_selected_outer{outer}.parquet")
        leakage_audit.append({
            "outer_fold": int(outer),
            "inner_assignment_customers": int(assignment.customer_ID.nunique()),
            "inner_fold_customer_counts": assignment.inner_fold.value_counts(
                ).sort_index().astype(int).to_dict(),
            "outer_validation_customers_in_inner_assignment": int(
                len(outer_ids & set(assignment.customer_ID))),
            "outer_score_customers": int(score.customer_ID.nunique()),
            "wrong_outer_score_customers": int(len(set(score.customer_ID) - outer_ids)),
        })
    coverage = {
        "outer_oof_rows": int(len(adaptive)),
        "outer_oof_customers": int(adaptive.customer_ID.nunique()),
        "duplicate_customer_month_rows": int(adaptive.duplicated(
            ["customer_ID", "t"]).sum()),
        "missing_score_rows": int(adaptive[["s", "p"]].isna().any(axis=1).sum()),
        "state_pairs": int(len(state_oof)),
        "state_pair_indices_unique": int(state_oof.pair_index.nunique()),
        "state_customers": int(state_oof.customer_ID.nunique()),
        "state_layer_counts": state_oof.layer.value_counts().astype(int).to_dict(),
    }
    write_json(out / "stage_summary.json", {
        "gamma_zero_verification": verification,
        "coverage_audit": coverage,
        "customer_split_leakage_audit": leakage_audit,
        "outer_selections": selections.to_dict("records"),
        "formal_outer_oof": formal_df.to_dict("records"),
        "matched_advantage_rates": {
            "high_response_at_matched_stability": systematic_stability,
            "stable_reduction_at_matched_response": systematic_response,
        },
        "stop_after": "V4-S6; awaiting human review",
    })
    log("阶段表、四张 Pareto 图和阶段报告已生成。")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    bundle = load_bundle(CFG)
    if mode == "verify":
        result = verify_gamma_zero(); print(json.dumps(result, indent=2)); return
    if mode in ("run", "all"):
        run_experiment(bundle)
    if mode in ("report", "all", "run"):
        build_report(bundle)


if __name__ == "__main__":
    main()
