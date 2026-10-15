# -*- coding: utf-8 -*-
"""V4-S2c fold-safe three-channel state change definition.

Inputs are restricted to features, dates, frozen customer folds, the prior
input-only D/P layer and S2b's input-only comparison layer.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "v4"
OUT = V4 / "outputs" / "s2c"


def ecdf(values: np.ndarray, training: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    ok = np.isfinite(values)
    base = np.sort(training[np.isfinite(training)])
    if base.size:
        out[ok] = np.searchsorted(base, values[ok], side="right") / base.size
    return out


def zero_aware_ecdf(values: np.ndarray, training: np.ndarray) -> np.ndarray:
    """Zero remains zero; positive values rank only among positive training values."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    out[finite & (values == 0)] = 0.0
    positive = finite & (values > 0)
    base = np.sort(training[np.isfinite(training) & (training > 0)])
    if base.size:
        out[positive] = np.searchsorted(base, values[positive], side="right") / base.size
    return out


def q(values: np.ndarray, level: float) -> float:
    x = values[np.isfinite(values)]
    return float(np.quantile(x, level)) if x.size else float("nan")


def classify(score: np.ndarray, low: float, high: float) -> np.ndarray:
    out = np.full(len(score), "medium", dtype=object)
    out[~np.isfinite(score)] = "insufficient"
    out[np.isfinite(score) & (score <= low)] = "stable"
    out[np.isfinite(score) & (score >= high)] = "high"
    return out


def top2(u: np.ndarray) -> np.ndarray:
    out = np.full(len(u), np.nan, dtype=np.float32)
    valid = np.isfinite(u).all(axis=1)
    ordered = np.sort(u[valid], axis=1)
    out[valid] = ordered[:, -2:].mean(axis=1)
    return out


def main() -> None:
    cfg = yaml.safe_load((V4 / "config_v4_s2c.yaml").read_text(encoding="utf-8"))
    opts = cfg["index"]
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = pd.read_csv(ROOT / cfg["paths"]["feature_dictionary"],
                             encoding="utf-8-sig")
    feature_names = dictionary.loc[dictionary.keep, "variable"].tolist()
    assert len(feature_names) == 160 and len(set(feature_names)) == 160
    cat = sorted(set(feature_names) & set(opts["categorical_columns"]))
    nums = [c for c in feature_names if c not in cat]
    groups = opts["groups"]
    panel = pd.read_parquet(ROOT / cfg["paths"]["panel"],
                            columns=["customer_ID", "S_2", "t"] + feature_names)
    split = pd.read_parquet(ROOT / cfg["paths"]["split"])
    pairs = pd.read_parquet(ROOT / cfg["paths"]["adjacent_pairs"])
    old = pd.read_parquet(ROOT / cfg["paths"]["v3_behavior_pairs"],
                          columns=["pair_id", "stratum_main"])
    s2b = pd.read_parquet(ROOT / cfg["paths"]["s2b_pair_level"],
                          columns=["pair_id", "layer_max"])
    pairs = pairs.merge(old, on="pair_id", how="left", validate="one_to_one")
    pairs = pairs.merge(s2b.rename(columns={"layer_max": "s2b_layer_max"}),
                        on="pair_id", how="left", validate="one_to_one")
    fits = {int(x["fold"]): x for x in json.loads(
        (ROOT / cfg["paths"]["s1_fold_fits"]).read_text(encoding="utf-8"))}
    assert len(pairs) == 165493 and pairs.pair_id.is_unique
    prev = pairs.row_prev.to_numpy(dtype=int)
    curr = pairs.row_curr.to_numpy(dtype=int)
    fold_array = pairs.fold.to_numpy(dtype=int)
    numeric = panel[nums].to_numpy(dtype=np.float64)
    num_pos = {name: j for j, name in enumerate(nums)}
    feature_pos = {name: j for j, name in enumerate(feature_names)}
    n = len(pairs)

    # Transition events are observable states even when the value itself is missing.
    m2o = np.zeros((n, len(feature_names)), dtype=bool)
    o2m = np.zeros_like(m2o)
    category_change = np.zeros_like(m2o)
    for name in feature_names:
        j = feature_pos[name]
        values = panel[name].to_numpy()
        pm, cm = pd.isna(values[prev]), pd.isna(values[curr])
        m2o[:, j] = pm & ~cm
        o2m[:, j] = ~pm & cm
        if name in cat:
            category_change[:, j] = ~pm & ~cm & (values[prev] != values[curr])

    pair_parts, fit_records, cluster_records = [], [], []
    for fold in sorted(split.fold.unique()):
        print("[V4-S2c] fold", fold, flush=True)
        fit = fits[int(fold)]
        train = np.flatnonzero(fold_array != fold)
        validation = np.flatnonzero(fold_array == fold)
        z = np.full((n, len(nums)), np.nan, dtype=np.float32)
        r = np.full_like(z, np.nan)
        var_ext95 = np.zeros_like(z, dtype=bool)
        var_ext99 = np.zeros_like(z, dtype=bool)
        for j, name in enumerate(nums):
            values = numeric[:, j]
            scale = float(fit["iqr_by_variable"][name])
            raw = np.abs(values[curr] - values[prev]) / scale
            clipped = np.minimum(raw, float(opts["change_cap"]))
            z[:, j] = raw
            r[:, j] = ecdf(clipped, clipped[train])
            var_ext95[:, j] = np.isfinite(raw) & (raw > q(raw[train], .95))
            var_ext99[:, j] = np.isfinite(raw) & (raw > q(raw[train], .99))

        group_c = np.full((n, len(groups)), np.nan, dtype=np.float32)
        group_u = np.full_like(group_c, np.nan)
        valid_cluster_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        shock95_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        shock99_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        trans_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        m2o_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        o2m_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        cat_by_group = np.zeros((n, len(groups)), dtype=np.int16)
        transition_units_by_group = np.zeros(len(groups), dtype=np.int16)
        cluster_thresholds = {}

        for gi, prefix in enumerate(groups):
            clusters = fit["cluster_members"][prefix]
            transition_units_by_group[gi] += len(clusters)
            cluster_sum = np.zeros(n, dtype=np.float64)
            cluster_thresholds[prefix] = []
            for ci, members in enumerate(clusters):
                pos = np.asarray([num_pos[c] for c in members])
                member_r = r[:, pos]
                valid_members = np.isfinite(member_r).sum(axis=1)
                need = max(1, int(np.ceil(len(members) * float(
                    opts["cluster_valid_member_fraction"]))))
                cluster_valid = valid_members >= need
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cluster_center = np.nanmedian(member_r, axis=1)
                    cluster_tail = np.nanmax(member_r, axis=1)
                cluster_center[~cluster_valid] = np.nan
                cluster_tail[~cluster_valid] = np.nan
                good = np.isfinite(cluster_center)
                cluster_sum[good] += cluster_center[good]
                valid_cluster_by_group[good, gi] += 1
                tail95, tail99 = q(cluster_tail[train], .95), q(cluster_tail[train], .99)
                extreme95 = cluster_valid & (cluster_tail > tail95)
                extreme99 = cluster_valid & (cluster_tail > tail99)
                shock95_by_group[:, gi] += extreme95
                shock99_by_group[:, gi] += extreme99
                fpos = np.asarray([feature_pos[c] for c in members])
                cm2o = m2o[:, fpos].any(axis=1)
                co2m = o2m[:, fpos].any(axis=1)
                ccat = category_change[:, fpos].any(axis=1)
                cany = cm2o | co2m | ccat
                trans_by_group[:, gi] += cany
                m2o_by_group[:, gi] += cm2o
                o2m_by_group[:, gi] += co2m
                cat_by_group[:, gi] += ccat
                cluster_thresholds[prefix].append({
                    "cluster_id": ci, "members": sorted(members),
                    "size": len(members), "tail_q95": tail95, "tail_q99": tail99})
                cluster_records.append({
                    "fold": int(fold), "group": prefix, "cluster_id": ci,
                    "members": "|".join(sorted(members)), "size": len(members),
                    "tail_q95": tail95, "tail_q99": tail99,
                    "validation_valid_rate": float(cluster_valid[validation].mean()),
                    "validation_extreme_p95_rate": float(extreme95[validation].mean()),
                    "validation_extreme_p99_rate": float(extreme99[validation].mean())})
            need_group = int(np.ceil(len(clusters) * .5))
            good_group = valid_cluster_by_group[:, gi] >= need_group
            group_c[good_group, gi] = (cluster_sum[good_group] /
                                        valid_cluster_by_group[good_group, gi])
            group_u[:, gi] = ecdf(group_c[:, gi], group_c[train, gi])

        # Categorical variables are singleton transition units assigned to their prefix.
        for name in cat:
            gi = groups.index(name[:2])
            transition_units_by_group[gi] += 1
            fp = feature_pos[name]
            cm2o, co2m, ccat = m2o[:, fp], o2m[:, fp], category_change[:, fp]
            trans_by_group[:, gi] += cm2o | co2m | ccat
            m2o_by_group[:, gi] += cm2o
            o2m_by_group[:, gi] += co2m
            cat_by_group[:, gi] += ccat

        valid_cluster_count = valid_cluster_by_group.sum(axis=1)
        shock95_count = shock95_by_group.sum(axis=1)
        shock99_count = shock99_by_group.sum(axis=1)
        ecb95 = shock95_count / np.maximum(valid_cluster_count, 1)
        ecb99 = shock99_count / np.maximum(valid_cluster_count, 1)
        ecb95[valid_cluster_count == 0] = np.nan
        ecb99[valid_cluster_count == 0] = np.nan
        u_shock95 = zero_aware_ecdf(ecb95, ecb95[train])
        u_shock99 = zero_aware_ecdf(ecb99, ecb99[train])

        total_transition_units = int(transition_units_by_group.sum())
        trans_count = trans_by_group.sum(axis=1)
        tb = trans_count / total_transition_units
        tb_m2o = m2o_by_group.sum(axis=1) / total_transition_units
        tb_o2m = o2m_by_group.sum(axis=1) / total_transition_units
        tb_cat = cat_by_group.sum(axis=1) / total_transition_units
        u_trans = zero_aware_ecdf(tb, tb[train])

        complete = np.isfinite(group_u).all(axis=1)
        broad = np.full(n, np.nan, dtype=np.float32)
        mean = np.full(n, np.nan, dtype=np.float32)
        broad[complete] = group_u[complete].max(axis=1)
        mean[complete] = group_u[complete].mean(axis=1)
        top_2 = top2(group_u)
        final99 = np.maximum.reduce([broad, u_shock99, u_trans])
        final95 = np.maximum.reduce([broad, u_shock95, u_trans])

        scores = {"broad": broad, "mean": mean, "top2": top_2,
                  "final_p99": final99, "final_p95": final95}
        thresholds, layers = {}, {}
        for name, score in scores.items():
            low, high = q(score[train], .25), q(score[train], .90)
            thresholds[name] = {"q25": low, "q90": high}
            layers[name] = classify(score, low, high)

        loo_scores, loo_layers = {}, {}
        for gi, prefix in enumerate(groups):
            keep = [j for j in range(len(groups)) if j != gi]
            broad_loo = np.full(n, np.nan, dtype=np.float32)
            broad_ok = np.isfinite(group_u[:, keep]).all(axis=1)
            broad_loo[broad_ok] = group_u[broad_ok][:, keep].max(axis=1)
            valid_loo = valid_cluster_count - valid_cluster_by_group[:, gi]
            shock_loo_count = shock99_count - shock99_by_group[:, gi]
            ecb_loo = shock_loo_count / np.maximum(valid_loo, 1)
            ecb_loo[valid_loo == 0] = np.nan
            u_shock_loo = zero_aware_ecdf(ecb_loo, ecb_loo[train])
            units_loo = total_transition_units - transition_units_by_group[gi]
            tb_loo = (trans_count - trans_by_group[:, gi]) / units_loo
            u_trans_loo = zero_aware_ecdf(tb_loo, tb_loo[train])
            final_loo = np.maximum.reduce([broad_loo, u_shock_loo, u_trans_loo])
            low, high = q(final_loo[train], .25), q(final_loo[train], .90)
            key = "final_leave_" + prefix[0] + "_out"
            loo_scores[key] = final_loo
            loo_layers[key] = classify(final_loo, low, high)
            thresholds[key] = {"q25": low, "q90": high}

        sub = pairs.iloc[validation][[
            "pair_id", "customer_ID", "fold", "date_prev", "date_curr",
            "stratum_main", "s2b_layer_max"]].copy()
        for gi, prefix in enumerate(groups):
            sub["U_" + prefix[0]] = group_u[validation, gi]
            sub["valid_clusters_" + prefix[0]] = valid_cluster_by_group[validation, gi]
            sub["shock_clusters_p95_" + prefix[0]] = shock95_by_group[validation, gi]
            sub["shock_clusters_p99_" + prefix[0]] = shock99_by_group[validation, gi]
            sub["transition_clusters_" + prefix[0]] = trans_by_group[validation, gi]
        for name, score in scores.items():
            sub["C_" + name] = score[validation]
            sub["layer_" + name] = layers[name][validation]
        for key, score in loo_scores.items():
            sub["C_" + key] = score[validation]
            sub["layer_" + key] = loo_layers[key][validation]
        sub["valid_continuous_variables"] = np.isfinite(z[validation]).sum(axis=1)
        sub["extreme_variable_count_p95"] = var_ext95[validation].sum(axis=1)
        sub["extreme_variable_count_p99"] = var_ext99[validation].sum(axis=1)
        sub["extreme_variable_share_p95"] = (sub.extreme_variable_count_p95 /
                                                       sub.valid_continuous_variables)
        sub["extreme_variable_share_p99"] = (sub.extreme_variable_count_p99 /
                                                       sub.valid_continuous_variables)
        sub["valid_continuous_clusters"] = valid_cluster_count[validation]
        sub["extreme_cluster_count_p95"] = shock95_count[validation]
        sub["extreme_cluster_count_p99"] = shock99_count[validation]
        sub["ECB_p95"] = ecb95[validation]
        sub["ECB_p99"] = ecb99[validation]
        sub["U_shock_p95"] = u_shock95[validation]
        sub["U_shock_p99"] = u_shock99[validation]
        sub["transition_cluster_count"] = trans_count[validation]
        sub["TB"] = tb[validation]
        sub["U_trans"] = u_trans[validation]
        sub["TB_missing_to_observed"] = tb_m2o[validation]
        sub["TB_observed_to_missing"] = tb_o2m[validation]
        sub["TB_categorical_change"] = tb_cat[validation]
        pair_parts.append(sub)
        fit_records.append({
            "fold": int(fold), "n_training_pairs": len(train),
            "n_validation_pairs": len(validation),
            "continuous_cluster_count": int(sum(
                len(fit["cluster_members"][prefix]) for prefix in groups)),
            "transition_unit_count": total_transition_units,
            "transition_units_by_group": dict(zip(groups,
                                                    transition_units_by_group.tolist())),
            "thresholds": thresholds, "cluster_tail_thresholds": cluster_thresholds})

    result = pd.concat(pair_parts, ignore_index=True).sort_values("pair_id")
    assert len(result) == len(pairs) and result.pair_id.is_unique
    result.to_parquet(OUT / "state_index_s2c_pair_level.parquet", index=False)
    pd.DataFrame(cluster_records).to_csv(OUT / "shock_cluster_calibration.csv",
                                         index=False, encoding="utf-8-sig")
    (OUT / "s2c_fold_fits.json").write_text(json.dumps(
        fit_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[V4-S2c] pairs", len(result), "layers",
          result.layer_final_p99.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
