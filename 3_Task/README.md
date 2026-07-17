# Entanglement Recovery for the divide-and-conquer VQE (Task 3)

## The brief

Task 2 ended with a measured interface-expressivity floor: with exact local states
and 24 SU(2)-preserving junction parameters, the divide-and-conquer VQE reaches
1.64% and stops. The advisor's reply set Task 3 (verbatim):

> (i) "Use HVA to convert dimers to the spin-liquid like state for the lower
> fragment (fragment-2) and then recombine. I am sure the combination will be
> seamless i.e. Heisenberg gate."
> (ii) "Entanglement Recovery: If necessary, modify Hamiltonian (not J→J') but by
> instead adding terms of next nearest neighbours (NNN) terms and you will see the
> non-local correlations will bring us closer to exact gs_energy."

The working hypothesis — *matching the fragments' correlation profiles before
joining makes the recombination seamless* — was registered as a three-prediction
scorecard (P1 stationarity broken, P2 floor pierced, P3 profile-RMS predicts the
final error) before any optimization ran. **All three came back refuted, each with
its mechanism located** — see §5 of the notebook.

## What was measured (19 sites)

| config | fragment-2 preparation | best error | verdict |
|---|---|---|---|
| A | bare dimers (exact local ground; Task-2 §9.1, reused) | **1.638%** | the bar |
| B | HVA layers optimized **jointly** with the junctions (34/44p) | 1.624% | = A within seed scatter (mean 2.04 ± 0.42%) |
| C | pre-trained vs H_NN + λ·H_NNN, frozen (λ sweep) | 1.788% (identity) / 27–34% (λ ≥ 0.75) | the tool self-disables |
| D | dressed by direct correlation fit to the ED target (rms 0.015–0.028) | 6.74% | profile ≠ entanglement |

Key findings, in the order the notebook derives them:

- **The target profile is NN resonance, not NNN structure** (§1): the 19-site ED
  restricted to fragment 2 weakens every dimer to −0.20…−0.47 and strengthens every
  non-dimer NN bond, while its NNN correlations stay ≈ 0 (the bare dimers already
  match them to rms 0.028). The ED pays +2.44 of local energy for this and earns
  −5.53 at the interface.
- **A stationarity theorem** (§2): the zero gradient at the identity junction is
  protected by SU(2) symmetry (Wigner–Eckart), not by local optimality — *any*
  product of S²-pure fragments is a stationary point of SU(2)-preserving junction
  gates, so no Heisenberg-gate dressing can break it (measured ≤ 2×10⁻¹⁵ on every
  dressed start; the entangled 26-site CG start, not a product, is the one
  exception at 5.4×10⁻³).
- **The NNN training Hamiltonian is self-defeating on this fragment** (§3): the
  dimer cover is the exact lowest S=0 state of H_NN + λ·H_NNN across the whole
  window λ ∈ [−0.5, +0.7] (Majumdar–Ghosh stability), the ferromagnetic branch
  leaves the singlet sector, and the frustrated branch (λ ≥ +0.75) pays +8…+10
  locally while moving *away* from the target profile. The data-driven λ* (fit the
  NNN correlations to the ED) lands inside the window: calibrating the tool
  disables the tool.
- **The advisor's "spreading" mechanism is real but below break-even** (§3.4):
  dressed starts do let the junctions harvest more interface energy (−0.67 → −1.95),
  but at 0.29–0.46 recovered per 1.0 of local energy paid (the ED's rate: 1.26).
  Across all nine configurations the final error tracks the energy paid with
  r = +0.994, and the profile RMS only with r = +0.593 (P3 reversed).
- **26 sites** (§4): with two odd fragments the lattice admits a *global dimer
  cover with one singlet on a cut bond* — the "other way to preserve SU(2)": exact
  S=0 as a depth-1 product, no Clebsch–Gordan two-register preparation, at the
  price of a higher (−39.0 vs −39.74) and symmetry-stationary start. The heavy
  cell compares junction VQEs from both sectorizations plus a joint dressing.

## Repository layout

| Path | Contents |
|---|---|
| `ER_VQE_Kagome.ipynb` | The full study: target profile, stationarity theorem, configs A–D, λ sweep, traces, the trade, P3, 26 sites, scorecard |
| `kagome_er.py` | Programmatic NNN geometry, weighted training sims, fragment dressing/fit trainers, mixed-schedule multi-seed optimizer, config-C/D pipelines + persistence, interface dimer cover, drawings, smoke test |
| `results/*.npz` | Persisted targets, sweeps and optima (auto-loaded; heavy cells resume) |
| `figures/` | Lattice/NNN map, target profile, pre-analysis, traces, trade, predictor, endgame ladder, dressed circuit |

Task-1/2 engines are imported from `../1_Task/` and `../2_Task/` (never copied);
`kagome_er.py` puts them on `sys.path` itself. `python kagome_er.py` runs the
7-assert smoke test (gradients vs finite differences, weighted-H identity, S²
preservation, cover-state prep, persistence round trips).

## Requirements & running

Same environment as Tasks 1–2 (`qiskit>=2.0`, `scipy`, `numpy`, `networkx`,
`matplotlib` + `pylatexenc`). Deterministic (fixed seeds). With `results/`
populated (this repo state) a **Run All takes minutes** except the §4 26-qubit
cell; recomputing §1–3 from scratch takes ≈ 3.5–4.5 h (the guarded cells save
incrementally and resume). The §4 run cell is gated: automated executions set
`ER_SKIP_26Q`, so only running it interactively computes (~1.5–2.5 h, peak < 4.5 GB
with the Task-2 blocked kernels; it asserts ≥ 4.5 GB free before starting).
