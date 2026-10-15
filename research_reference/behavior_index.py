# -*- coding: utf-8 -*-
"""
S2d 行为变化指数的唯一实现（v3.1 第 3 节）

本模块是唯一指标实现；主流程、敏感性与日志都调用这里，不各自复制公式。
只读特征、日期、客户 ID 与已固定 fold；不读取评分、target 或 SHAP。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _quantile(v: np.ndarray, qs):
    """固定 linear 分位数算法；兼容 numpy 1.21（interpolation=）与 >=1.22（method=）。"""
    try:
        return np.quantile(v, qs, method="linear")
    except TypeError:
        return np.quantile(v, qs, interpolation="linear")


def adjacent_prev_index(panel: pd.DataFrame) -> np.ndarray:
    """同客户且日历月相差 1 的上一行下标；否则 -1。不要求间隔固定 30 天。"""
    cust = panel["customer_ID"].to_numpy()
    s2 = pd.to_datetime(panel["S_2"])
    month = s2.dt.year.to_numpy() * 12 + s2.dt.month.to_numpy()
    n = len(panel)
    same = np.zeros(n, dtype=bool)
    same[1:] = cust[:-1] == cust[1:]
    contig = np.zeros(n, dtype=bool)
    contig[1:] = (month[1:] - month[:-1]) == 1
    ok = same & contig
    prev = np.full(n, -1, dtype=int)
    idx = np.arange(n)
    prev[ok] = idx[ok] - 1
    return prev


def build_adjacent_pairs(panel: pd.DataFrame, folds: pd.DataFrame) -> tuple:
    """返回 (pairs_df, stats)；pair_id=(customer_ID,date_prev,date_curr)。"""
    prev = adjacent_prev_index(panel)
    r_curr = np.where(prev >= 0)[0]
    r_prev = prev[r_curr]
    s2 = pd.to_datetime(panel["S_2"]).to_numpy()
    cust = panel["customer_ID"].to_numpy()
    pairs = pd.DataFrame({
        "customer_ID": cust[r_curr],
        "date_prev": s2[r_prev],
        "date_curr": s2[r_curr],
        "t_prev": panel["t"].to_numpy()[r_prev],
        "t_curr": panel["t"].to_numpy()[r_curr],
        "row_prev": r_prev,
        "row_curr": r_curr,
    })
    pairs["pair_id"] = (
        pairs["customer_ID"].astype(str) + "|"
        + pd.to_datetime(pairs["date_prev"]).dt.strftime("%Y-%m-%d") + "|"
        + pd.to_datetime(pairs["date_curr"]).dt.strftime("%Y-%m-%d"))
    pairs = pairs.merge(folds, on="customer_ID", how="left")
    pc = pairs.groupby("customer_ID").size()
    stats = dict(
        n_rows=int(len(panel)),
        n_customers=int(panel["customer_ID"].nunique()),
        n_gap_excluded=int(panel["month_gap_before"].sum()),
        n_pairs=int(len(pairs)),
        pairs_per_customer_mean=round(float(pc.mean()), 3),
        pairs_per_customer_p10=int(pc.quantile(0.10)),
        pairs_per_customer_p50=int(pc.quantile(0.50)),
        pairs_per_customer_p90=int(pc.quantile(0.90)),
    )
    return pairs, stats


def fit_scale(x: np.ndarray, eps: float) -> tuple:
    """主尺度 IQR=Q75-Q25（linear）；退化时备用 Q95-Q05。"""
    v = x[np.isfinite(x)]
    if v.size < 10:
        return None, "有效观测<10"
    q25, q75 = _quantile(v, [0.25, 0.75])
    s = float(q75 - q25)
    if np.isfinite(s) and s > eps:
        return s, "iqr"
    q05, q95 = _quantile(v, [0.05, 0.95])
    s2 = float(q95 - q05)
    if np.isfinite(s2) and s2 > eps:
        return s2, "p95_p05"
    return None, "尺度退化"


def pair_z(values: np.ndarray, rows_curr: np.ndarray, rows_prev: np.ndarray,
           scale: float) -> np.ndarray:
    """z = |x_curr - x_prev| / scale；两端任一缺失记 NaN，不填 0。"""
    xc = values[rows_curr]
    xp = values[rows_prev]
    z = np.full(len(rows_curr), np.nan)
    ok = np.isfinite(xc) & np.isfinite(xp)
    z[ok] = np.abs(xc[ok] - xp[ok]) / scale
    return z


def coverage_stats(values: np.ndarray, rows_curr: np.ndarray,
                   rows_prev: np.ndarray, cust_curr: np.ndarray) -> tuple:
    ok = np.isfinite(values[rows_curr]) & np.isfinite(values[rows_prev])
    cov = float(ok.mean()) if len(ok) else 0.0
    n_cust = int(pd.Series(cust_curr[ok]).nunique()) if ok.any() else 0
    return cov, n_cust


def spearman_with_overlap(X: pd.DataFrame, min_overlap: int) -> tuple:
    """两两 Spearman 水平相关；共同非缺失 < min_overlap 记为 0。"""
    k = X.shape[1]
    if k < 2:
        return np.ones((k, k)), np.zeros((k, k), dtype=int)
    notna = X.notna().to_numpy(dtype=np.int32)
    overlap = notna.T @ notna
    rho = np.nan_to_num(X.corr(method="spearman").to_numpy(), nan=0.0)
    rho[overlap < min_overlap] = 0.0
    np.fill_diagonal(rho, 1.0)
    return rho, overlap


def cluster_labels(rho: np.ndarray, abs_rho_threshold: float) -> np.ndarray:
    """距离为 1-|rho| 的 complete-linkage 聚类，切分距离 = 1-阈值。"""
    k = rho.shape[0]
    if k == 1:
        return np.array([1])
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)
    z = linkage(squareform(dist, checks=False), method="complete")
    return fcluster(z, t=1.0 - abs_rho_threshold, criterion="distance")


def spearman_pairwise_fast(X: pd.DataFrame, min_overlap: int) -> tuple:
    """与 spearman_with_overlap 完全相同的判据（两两 Spearman、成对删除、
    共同非缺失 < min_overlap 记 0），但用 BLAS 矩阵乘法实现，
    避免 pandas 逐对 Cython 循环在列数多时变成 O(p^2 * n) 的瓶颈。
    """
    M = np.isfinite(X.to_numpy(dtype=float))
    R = X.rank().to_numpy(dtype=float)
    R0 = np.where(M, R, 0.0)
    Mf = M.astype(float)
    N = Mf.T @ Mf
    S_prod = R0.T @ R0
    S_i = R0.T @ Mf
    S_ii = (R0 ** 2).T @ Mf
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_i = S_i / N
        mean_j = S_i.T / N
        var_i = S_ii / N - mean_i ** 2
        var_j = S_ii.T / N - mean_j ** 2
        rho = (S_prod / N - mean_i * mean_j) / np.sqrt(var_i * var_j)
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    rho[N < min_overlap] = 0.0
    np.fill_diagonal(rho, 1.0)
    return rho, N.astype(int)


def prefix_index(z: pd.DataFrame, labels: np.ndarray, var_cov_min: float,
                 cluster_cov_min: float, cap: float) -> pd.DataFrame:
    """按簇聚合、再按前缀聚合。行=相邻对，列=该前缀合格变量。

    返回 index, n_valid_clusters, n_clusters, max_raw_z, prefix_ok
    """
    n = len(z)
    zc = z.to_numpy(dtype=float)
    finite = np.isfinite(zc)
    clipped = np.where(finite, np.minimum(zc, cap), np.nan)
    raw_max = np.full(n, np.nan)
    any_fin = finite.any(axis=1)
    if any_fin.any():
        raw_max[any_fin] = np.nanmax(np.where(finite, zc, -np.inf)[any_fin],
                                     axis=1)
    cluster_vals = []
    for lab in np.unique(labels):
        cols = np.where(labels == lab)[0]
        m = finite[:, cols]
        cov = m.mean(axis=1)
        mean_v = np.nanmean(np.where(m, clipped[:, cols], np.nan), axis=1)
        cluster_vals.append(np.where(cov >= var_cov_min, mean_v, np.nan))
    if cluster_vals:
        cv = np.column_stack(cluster_vals)
    else:
        cv = np.zeros((n, 0))
    n_clusters = cv.shape[1]
    valid = np.isfinite(cv)
    n_valid = valid.sum(axis=1)
    need = max(1, int(np.ceil(cluster_cov_min * n_clusters))) if n_clusters else 0
    ok = n_valid >= need
    idx = np.where(ok, np.nanmean(cv, axis=1), np.nan)
    return pd.DataFrame({"index": idx, "n_valid_clusters": n_valid,
                         "n_clusters": n_clusters, "max_raw_z": raw_max,
                         "prefix_ok": ok})


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """加权 inverted-CDF：排序后累计归一权重首次达到 q 的取值。"""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    v, w = v[m], w[m]
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    i = int(np.searchsorted(cw, q, side="left"))
    return float(v[min(i, len(v) - 1)])


def customer_equal_weights(customer_ids: np.ndarray) -> np.ndarray:
    """每客户等权：该客户每对权重 = 1/该客户在本集合中的对数。"""
    s = pd.Series(customer_ids)
    return 1.0 / s.map(s.value_counts()).to_numpy(dtype=float)


def classify_pairs(idx_valid: np.ndarray, state_change: np.ndarray,
                   h_d: np.ndarray, h_p: np.ndarray, zmax_d: np.ndarray,
                   zmax_p: np.ndarray, q25_d: float, q25_p: float,
                   q90_d: float, q90_p: float, z_guard: float,
                   enforce_z_guard: bool = True,
                   low_label: str = "dp_low") -> np.ndarray:
    """按 v3.1 第 3.5 节顺序分层。

    enforce_z_guard=True  ：v3.1 原定义（逐列联合 max z 守卫），归档口径。
    enforce_z_guard=False ：v3.1.1 主层"聚合低变化"，逐列 max z 只作诊断变量。
    """
    out = np.empty(len(h_d), dtype=object)
    out[:] = "dp_middle"
    low = (h_d <= q25_d) & (h_p <= q25_p)
    if enforce_z_guard:
        low = low & (zmax_d <= z_guard) & (zmax_p <= z_guard)
    high = (h_d > q90_d) | (h_p > q90_p)
    out[high & ~low] = "dp_high"
    out[low] = low_label
    out[state_change] = "state_change"
    out[~idx_valid] = "insufficient"
    return out
