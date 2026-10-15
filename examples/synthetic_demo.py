"""Synthetic-only objective walkthrough; no file or network inputs."""
import json
import numpy as np
from sgtr_demo import adjacency, derivatives, loss, stable_gate


def main():
    rng = np.random.default_rng(42)
    customers = np.repeat(['synthetic_A', 'synthetic_B', 'synthetic_C'], 4)
    months = np.tile(np.array(['2020-01', '2020-02', '2020-03', '2020-04']), 3)
    # Final customer outcome repeated on monthly training rows.
    labels = np.repeat([0, 1, 0], 4)
    margins = rng.normal(size=len(labels))
    edges = adjacency(customers, months)
    # Supplied synthetic G, not fitted from AMEX or from predictions/labels.
    g = np.array([.10, .20, .25, .40, .60, .80, .90, .95, np.nan])
    weights = stable_gate(g)
    grad, diag = derivatives(margins, labels, edges, weights)
    bce_grad, _ = derivatives(margins, labels, edges, weights, lam=0)
    result = {
        'status': 'SYNTHETIC DEMONSTRATION; NOT RESEARCH RESULTS',
        'seed': 42, 'rows': len(labels), 'natural_month_edges': len(edges),
        'stable_edges_with_penalty': int(weights.sum()), 'lambda': .03,
        'bce': loss(margins, labels, edges, weights, lam=0),
        'sgtr_objective': loss(margins, labels, edges, weights),
        'nonzero_temporal_gradient_rows': int(np.count_nonzero(grad - bce_grad)),
        'minimum_hessian_diagonal': float(diag.min()),
        'trained_model': False, 'formal_state_mapper_reproduced': False,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
