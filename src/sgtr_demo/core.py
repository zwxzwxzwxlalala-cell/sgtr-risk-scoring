"""SG-TR objective on supplied margins and frozen state percentiles.

The research mapper is fitted on training customers only. This demo accepts G;
it does not estimate G or claim to reproduce the full three-channel mapper.
"""
import numpy as np


def sigmoid(margin):
    x = np.asarray(margin, dtype=float)
    return np.exp(-np.logaddexp(0.0, -x))


def stable_gate(g, threshold=0.25):
    g = np.asarray(g, dtype=float)
    finite = np.isfinite(g)
    if np.any((g[finite] < 0) | (g[finite] > 1)):
        raise ValueError('Finite G values must be in [0, 1].')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1].')
    return (finite & (g <= threshold)).astype(float)


def adjacency(customer, month):
    """Return original-row index pairs for consecutive natural months.

    Rows may be unordered. Duplicate customer-month keys are rejected. The
    caller supplies only rows from a single training/validation partition.
    """
    customer = np.asarray(customer).astype(str)
    dates = np.asarray(month, dtype='datetime64[M]')
    if customer.ndim != 1 or dates.ndim != 1 or len(customer) != len(dates):
        raise ValueError('customer and month must be aligned one-dimensional arrays.')
    if np.isnat(dates).any():
        raise ValueError('month contains NaT.')
    lookup, pairs = {}, []
    months = dates.astype(np.int64)
    for i, (c, m) in enumerate(zip(customer, months)):
        key = (c, int(m))
        if key in lookup:
            raise ValueError('Duplicate customer-month key.')
        lookup[key] = i
    for (c, m), b in sorted(lookup.items()):
        a = lookup.get((c, m - 1))
        if a is not None:
            pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _validate(margin, y, edges, weights, lam):
    f, y = np.asarray(margin, float), np.asarray(y, float)
    raw_edges = np.asarray(edges)
    if raw_edges.size == 0:
        raw_edges = np.empty((0, 2), dtype=np.int64)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError('edges must have shape (n_edges, 2).')
    if not np.issubdtype(raw_edges.dtype, np.integer):
        raise ValueError('edge positions must be integers.')
    e, w = raw_edges.astype(np.int64), np.asarray(weights, float)
    if f.ndim != 1 or y.shape != f.shape or not len(f):
        raise ValueError('margin and y must be aligned nonempty vectors.')
    if not np.isfinite(f).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Margins must be finite and labels binary.')
    if w.shape != (len(e),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Edge weights must be finite, nonnegative and aligned.')
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be finite and nonnegative.')
    if (e < 0).any() or (e >= len(f)).any() or (e[:, 0] == e[:, 1]).any():
        raise ValueError('Invalid edge positions.')
    if len(np.unique(np.sort(e, axis=1), axis=0)) != len(e):
        raise ValueError('Each undirected temporal edge must occur once.')
    return f, y, e, w


def loss(margin, y, edges, weights, lam=0.03):
    """Sum row-level BCE plus lambda * sum w * (f_b-f_a)^2."""
    f, y, e, w = _validate(margin, y, edges, weights, lam)
    bce = np.sum(np.logaddexp(0.0, f) - y * f)
    delta = f[e[:, 1]] - f[e[:, 0]]
    return float(bce + lam * np.sum(w * delta**2))


def derivatives(margin, y, edges, weights, lam=0.03):
    """Analytic gradient and exact Hessian diagonal, used as a diagonal approximation.

    The full Hessian also has -2*lambda*w off-diagonal entries per edge.
    Like the research custom objective, this API returns only the diagonal.
    """
    f, y, e, w = _validate(margin, y, edges, weights, lam)
    p = sigmoid(f)
    grad, hess = p - y, p * (1.0 - p)
    a, b = e[:, 0], e[:, 1]
    delta = f[b] - f[a]
    np.add.at(grad, a, -2 * lam * w * delta)
    np.add.at(grad, b, 2 * lam * w * delta)
    np.add.at(hess, a, 2 * lam * w)
    np.add.at(hess, b, 2 * lam * w)
    return grad, hess


def customer_mean_abs_change(margin, edges, pair_customers, mask):
    """Within-customer average, then equal average of eligible customers.

    Outer-OOF reporting additionally averages its five fold-level estimates.
    This primitive handles one partition, not an entire nested evaluation.
    """
    f = np.asarray(margin, float)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    ids, keep = np.asarray(pair_customers).astype(str), np.asarray(mask, bool)
    if ids.shape != (len(e),) or keep.shape != ids.shape:
        raise ValueError('Pair metadata must align with edges.')
    if not keep.any():
        return float('nan')
    delta = np.abs(f[e[:, 1]] - f[e[:, 0]])
    values = [delta[keep & (ids == c)].mean() for c in np.unique(ids[keep])]
    return float(np.mean(values))
