# Reproduction levels

## 1. Runnable, synthetic objective demonstration

`examples/synthetic_demo.py` constructs three fictional customers and supplies fictional G values. Tests exercise the objective's derivative mathematics, month adjacency, Stable gate and customer-level averaging. It has no network inputs, no AMEX data and no dependency on LightGBM.

The demo's robust sigmoid implementation and edge-list API are presentation-layer implementations. They do not replace the frozen research training code or its saved models. Synthetic results must not be cited as research performance.

## 2. Runnable aggregate figure regeneration

`scripts/render_results.py` redraws the development/holdout comparison using `results/development_to_holdout_comparison.csv`. The tables are copied byte-for-byte from the locked result directory after checking their SHA-256 against the final research result manifest. `results/provenance.json` records relative source paths and hashes. It contains no machine-specific absolute paths.

These hashes establish the identity of exported files, not independent verification of the experiment. The public package deliberately omits customer-level scores and identifiers.

## 3. Research implementation available for inspection

`research_reference/` contains byte-identical source snapshots. Read `adaptive_tr_core.py` for training-only state fitting and the custom objective, `adaptive_tr_s6c.py` for the Hard Gate training path, `sgtr_s6d.py` for statistical confirmation, `sgtr_s7.py` for robustness/calibration, and `dynamic_shap_s8.py` for OOF explanations.

These scripts retain their original relative imports and numbered project paths. **They are not runnable as a standalone pipeline in this public candidate.** Historical stage names identify provenance; the final main model is SG-TR with Q25 and lambda .03, not every candidate in these files.

## 4. Full empirical reproduction: not shipped in this release

A complete rerun needs permitted access to the original feature and label tables, the verified sampling/feature-selection workflow, frozen customer splits, all training-fold state mappings and the full dependency/configuration bundle. Those prerequisites are not replaced by synthetic data or generated lookalike artifacts. Original research results and the manuscript remain unchanged by this packaging work.

Do not rerun the project's original locked-holdout protocol to search for better results. New evaluation work must be named as a new experiment and retain the published evidence snapshot.

## Implementation notes

- Original final models used LightGBM 4.6.0; this demo does not load or train them.
- Main BCE is a sum over monthly rows, with each customer's final label repeated on its rows.
- Prediction evaluation uses the true final observation per customer.
- Temporal averages are computed within customer, then across eligible customers. Development OOF additionally uses equal fold weighting.
- Customer-paired bootstrap resamples all observations of a customer together; the published holdout intervals use 2,000 draws.
- The learned state mapping is training-only; evaluation distributions never refit its percentiles.
- In the final analysis, all 149 continuous units were singletons. No effective correlation reduction is claimed.
- Unsupervised feature selection was not nested inside each fold. State mappings were fold specific; these are separate preprocessing steps.
- The final model serialization underwent a reversible CRLF-to-LF recovery and development-only audits before scoring resumed. Original artifacts were preserved. Historical final-model training predictions were not saved, so no claim of comparison to those missing predictions is made.
