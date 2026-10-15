# Research code snapshots

These eight files are byte-identical snapshots of the actual research code. Their SHA-256 values and original project-relative paths are recorded in `../results/provenance.json`.

| File | Read for |
|---|---|
| `adaptive_tr_core.py` | Training-fold state maps, custom objective and base training |
| `adaptive_tr_s6c.py` | Piecewise/Hard Gate comparison and final Hard Gate training path |
| `adaptive_tr_experiment.py` | Customer-level nested selection and metric aggregation |
| `state_index_s2c.py` | Final structural three-channel state-definition stage |
| `behavior_index.py` | Correlation-unit utilities imported by later state code |
| `sgtr_s6d.py` | Paired statistical confirmation and Q20/Q25/Q30 sensitivity |
| `sgtr_s7.py` | Customer-balanced robustness and fold-safe calibration |
| `dynamic_shap_s8.py` | Formal OOF dynamic-attribution sampling and analysis |

The original relative imports, numbered directories and legacy stage names are preserved for traceability. This directory is a **reading snapshot**, not a standalone full reproduction package. It deliberately omits data, fitted mapping objects, customer splits, models, detailed predictions and private paper files. Do not invoke historical training scripts expecting the public demo's synthetic data to replace those inputs.

The synthetic tests extract only the inspected `make_objective` and `gate_weight` function definitions to compare their mathematics. They do not import or execute research entry points.
