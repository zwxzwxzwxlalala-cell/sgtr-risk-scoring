# Method and research positioning

## Problem and data units

A monthly model produces both a cross-sectional ranking and a trajectory for each customer. The project asks how to reduce variation on input-stable adjacent months while retaining response on changing inputs. The target is one final default label per customer, joined to all of that customer's monthly rows. The main BCE loss is row weighted; final-period evaluation counts each customer once.

The original pipeline uses customer-disjoint outer five-fold and inner three-fold splits. Each inner-training split refits the state statistics. Parameter selection uses an acceptable domain followed by Stable-volatility reduction, with simpler candidates preferred in near ties. The final SG-TR definition is `G <= .25`, lambda `.03`. The main model was frozen before the formal holdout model results were viewed.

## Input-only state construction

The research implementation is in `research_reference/adaptive_tr_core.py::fit_apply_state`. All scales and reference distributions below are estimated on the current training customers.

1. **Continuous changes.** Scale absolute adjacent-month changes by the feature's training IQR (with a numerical floor), cap the standardized changes, and map them through a right-continuous training ECDF.
2. **Broad.** Aggregate continuous units within each of the five D/P/B/S/R prefixes; apply a second training ECDF to each group score. Broad is the maximum of the five group percentiles. Complete group support is required.
3. **Sparse Shock.** The final training implementation forms a unit-level tail statistic from the variable percentile scores; a unit is extreme when this statistic strictly exceeds its own training P99. Extreme continuous units divided by valid continuous units gives the burden. A zero-aware ECDF maps zero to zero and ranks positive burdens only among positive training burdens. This final gate rule is distinct from the raw-change P99 diagnostic used in earlier structural stress tests.
4. **Transition.** Count units with missing-to-observed, observed-to-missing or observed categorical transitions. The final training implementation divides by the total defined transition units (continuous units plus categorical singletons), then applies the zero-aware positive-only ECDF.
5. **Final G.** Take the maximum of Broad, calibrated Shock and calibrated Transition, then use a further training ECDF. Stable is `G <= .25`; High is `G >= .90`; the remainder is Medium. Missing G is marked insufficient and receives no Stable penalty.

All 149 continuous units in the final main analysis are singletons, so the Shock units are effectively feature-level units. Prefix labels group anonymous features without asserting detailed business meanings. The state construction never takes labels, scores or SHAP as inputs.

## Objective and derivatives

Let a temporal edge connect rows a and b of the same customer in consecutive natural months. With raw margin f and customer final label y:

$$
L=\sum_r\{\log(1+e^{f_r})-y_rf_r\}
+\lambda\sum_{(a,b)\in E}w_{ab}(f_b-f_a)^2,
\qquad w_{ab}=\mathbf1(G_{ab}\le .25),\quad\lambda=.03.
$$

An edge contributes `2λw(f_b−f_a)` to the gradient at b and its negative at a. Its Hessian has `2λw` on the two diagonal positions and `−2λw` on the cross positions. The LightGBM custom objective supplies the analytic gradient and **Hessian diagonal**, omitting cross-row Hessian entries. Every monthly row retains its BCE gradient, including non-Stable rows.

The public demo implements these calculations with an edge-list API and a numerically stable sigmoid. The research snapshots preserve the actual original adjacency-array implementation. The demo has no model-selection or full-state-mapping path.

## Measurements and calibration

Temporal volatility is averaged over a customer's eligible edges in a state, then equally over customers; development estimates additionally average folds equally. Stable reduction is `1 − V_SGTR,Stable / V_M0,Stable`; High retention is `V_SGTR,High / V_M0,High`. SRR is High volatility divided by Stable volatility, reported with both components.

Default trajectory uses customers with final default label 1 whose last six observations span six consecutive natural months. It measures signed final-minus-first raw margin over that window; retention is the ratio to M0_ref. This is a model-trajectory diagnostic, not a directly observed monthly default-risk process.

Platt calibration uses customer-level final-period cross-fitted development predictions. It acts on the terminal probability scale. All temporal comparisons use uncalibrated margins. SHAP analyses use OOF raw-margin contributions and their adjacent-month differences; they are descriptive model attribution.

## Direct neighbors and contribution

The project's latest literature audit identifies the following direct neighbors. This is a summary of that audit, not a claim that these techniques are absent from prior work.

| Neighbor | Shared methodological concern | Scope of this project's increment |
|---|---|---|
| Ouyang et al.: longitudinal classification consistency | Consistency across repeated observations | Input-state-selective raw-margin penalties and state-stratified credit-scoring evaluation |
| Maystre & Russo: TCSR | Temporally consistent survival predictions | A final-outcome classification problem with an input-only Stable gate |
| DeepTCSR | Learned temporally consistent survival regression | A tree-based customer-month formulation and separate stability/response diagnostics |
| Noroozizadeh et al.: Temporal-SCL | Temporal smoothness alongside feature heterogeneity | Explicit Broad/Shock/Transition state gates at adjacent prediction edges |
| Maystre et al.: TC-λ | Same final class predicted across sequence prefixes | Gates derived from input changes rather than future soft predictions |

Manifold regularization, edge-preserving smoothing and fused penalties supply conceptual antecedents. Hard gates, temporal consistency and dynamic explanations are not claimed as first inventions. The project contributes a narrow, applied combination: independently defined multi-channel input state, Stable-only temporal penalties and an evidence chain from customer-level OOF through robustness and dynamic attribution to a frozen same-source customer holdout.

The public release does not contain the retrieved third-party full-text corpus. The manuscript's verified bibliography and full closest-method matrix remain separate research materials pending their own publication review.
