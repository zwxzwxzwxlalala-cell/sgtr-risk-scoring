# Frozen research results

These are empirical research results, **not outputs from the synthetic demo**. The exporter verified all eight included aggregate files against the final result manifest. Relative source paths, file hashes and that manifest's hash are in [`results/provenance.json`](../results/provenance.json).

## Sample and evaluation

- Development: 15,000 customers. Same-source locked holdout: 30,000 customers, zero overlap.
- Holdout monthly rows: 361,427; consecutive-natural-month edges: 330,366.
- Feature-label match: 100%; customers with nonconstant final labels: 0 in the final audit.
- Customer-level final-period evaluation: one true final observation per customer.
- Temporal statistics: within-customer means followed by equal customer means within each eligible layer.
- Uncertainty: 2,000 customer-level paired bootstrap resamples; 95% percentile intervals.

Layer membership is edge-level. One customer can appear in more than one state across time, so layer customer counts must not be summed as unique people.

## Final-period prediction

| Model | Probability scale | AUC | KS | Log Loss | Brier | Calibration intercept | Calibration slope |
|---|---|---:|---:|---:|---:|---:|---:|
| M0_ref | Raw | .952608 | .777464 | .244978 | .076114 | −.120457 | 1.151365 |
| M0_ref | Platt | .952608 | .777464 | .242471 | .075675 | .049233 | 1.032534 |
| SG-TR | Raw | .952378 | .774999 | .248028 | .076376 | −.145707 | 1.194746 |
| SG-TR | Platt | .952378 | .774999 | .243518 | .075861 | .054857 | 1.024967 |

Source: [`final_period_metrics.csv`](../results/final_period_metrics.csv). Full precision is retained there. Platt improves each model's probability scale; SG-TR still has a small probability-loss cost relative to calibrated M0_ref.

## Temporal behavior

| Quantity | M0_ref | SG-TR |
|---|---:|---:|
| Stable mean absolute raw-score change | .191844 | .135869 |
| Medium mean absolute raw-score change | .313724 | .245285 |
| High mean absolute raw-score change | .522404 | .456748 |
| SRR: High / Stable | 2.723066 | 3.361675 |

Stable reduction is 29.1773%; High response retention is 87.4320%; default-trajectory retention is 100.6231%. High edges receive no direct temporal penalty, but the shared model changes during training; High retention is therefore not expected to equal 100% mechanically.

## Paired uncertainty and costs

| Quantity | Point | 95% percentile bootstrap interval |
|---|---:|---:|
| Stable reduction | 29.1773% | [28.5735%, 29.8161%] |
| High response retained | 87.4320% | [86.8387%, 88.0056%] |
| AUC SG-TR − M0 | −.000230 | [−.000606, .000129] |
| Raw Log Loss relative worsening | 1.2448% | [.9414%, 1.5663%] |
| Platt Log Loss relative worsening | .4320% | [.0934%, .7902%] |
| Platt Brier relative worsening | .2451% | [−.1585%, .6496%] |
| Default trajectory retained | 100.6231% | [99.3865%, 101.8177%] |

An interval spanning zero does not establish exact equivalence. No post-hoc significance threshold was used to retune the model. All published intervals, including unfavorable changes, remain in [`paired_bootstrap_summary.csv`](../results/paired_bootstrap_summary.csv).

## Frozen acceptance and transport

The five conditions were: Stable reduction >20%; High response ≥80%; AUC difference ≥−.003; calibrated Log Loss relative worsening ≤2%; calibrated Brier relative worsening ≤2%. All passed, without manual override, giving the frozen classification **Core replicated**. See [`frozen_acceptance_result.json`](../results/frozen_acceptance_result.json).

Holdout Stable/Medium/High edge shares are 24.2788% / 64.5309% / 9.9777%; 1.2126% have insufficient state support. These use development-fitted mappings, not holdout quantiles. See [`state_layer_counts.csv`](../results/state_layer_counts.csv) and [`state_transportability.csv`](../results/state_transportability.csv).

Development to holdout: Stable reduction 29.5479% →29.1773%; High retention 87.4525% →87.4320%. This is same-source customer validation. Cross-institution, cross-time and deployment effects remain outside the evidence.
