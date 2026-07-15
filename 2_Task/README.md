# Divide-and-Conquer VQE with SU(2)-preserving junction gates

## The idea

Sec. 4 of the paper scales VQE by splitting the lattice into subregions
("local VQE"), freezing the optimized sub-circuits, and re-joining them with a handful
of junction $R_y(\theta)$ rotations ("global VQE", Fig. 4). This task repeats that
strategy on the 19-site KAFH lattice of Task 1 without breaking SU(2): every gate —
dimer preparations, local HVA layers, *and the junction gates* — is the Heisenberg /
eSWAP gate $U_H(\theta)=e^{-i\theta(XX+YY+ZZ)/4}$, so $\langle S^2\rangle$ stays in the
correct spin sector at every stage by construction ($J=1$ uniform, no Hamiltonian
calibration anywhere).

## The partition (validated against the Task-1 `edges_19` itself)

* **Subregion 1** — sites 0–10 (11 sites, odd → doublet), 14 bonds, 5 dimers + spinon
  on site 10 (at the interface).
* **Subregion 2** — sites 11–18 (8 sites, even → singlet), 10 bonds, perfect 4-dimer
  cover — which turns out to be its exact ground state (Majumdar–Ghosh-like), so
  Step 1 for this region costs 0 gates and 0 parameters.
* **Interface** — 6 bonds; the cut splits three corner-sharing triangles, and two of
  them share site 11 (several *coupled* junctions, not one).

## Method highlights

- **Step 1 reuses Task-1 machinery unchanged** (`K.HVASimulator` on each subregion,
  L-BFGS-B + adjoint gradient); everything new lives in `kagome_dc.py`.
- **Matrix-free global VQE** (`RecombinationVQE`): the frozen concatenated state is
  absorbed into $|\psi_0\rangle$ and $H|\psi\rangle$ is applied bond-by-bond — no
  $2^{19}$ sparse operator is ever built, and
  the adjoint gradient over the junction angles costs ~3 energy evaluations.
- **The stationarity no-go, measured**: exact local ground states make the naive
  concatenation an exact critical point of the junction optimization
  ($\langle[A,H]\rangle=0$), so junction VQE *requires* random multi-starts — the
  notebook measures $\|\nabla E\|_{\theta=0}\sim10^{-16}$ and keeps the frozen state
  as a fallback candidate in every run.
- **H_SEL**: A reduced junction Hamiltonian was constructed for this cut by retaining the bonds within {junction site} ∪ {immediate neighbours}, corresponding to 21 of 30 bonds (63 of 90 Pauli terms). Its optimization performance was then compared directly with that of the full Hamiltonian.
- **Metric discipline from the start** (Task-1 Sec. 4.6c lesson): fidelities are
  projections onto the full *degenerate ground manifold* (doublets everywhere), and
  every optimization is repeated with ≥5 seeds and reported mean ± std.
- **Error attribution**: an exact-frozen control (junction VQE on top of exact local
  states) separates interface-expressivity error from local-VQE suboptimality, plus a
  spinon-placement control for the dimer-cover choice.

## Follow-ups (§9 of the notebook)

- **(a) Exact-local pipeline**: §3.1's spinon@0 cover makes subregion 1 locally exact
  (14 params, reps=1); freezing it makes the deployable circuit start exactly at the
  E₁+E₂ bound, and the junction VQE then works from perfect locals.
- **(b) Prolongation**: `RecombinationVQE.refine` restarts L-BFGS from the saved optima
  of every iteration-capped run (+400 its).
- **(c) 26 sites**: `chain(3)` nests `chain(2)` as a prefix, so subregion 1 *is* the
  19-site system — Step 1 is recycled from the Task-1 hybrid optimum via a graph
  isomorphism + statevector qubit permutation (energy preserved to 1e-8). The junction
  VQE (4 cut bonds, H_SEL objective) runs matrix-free at 2²⁶. Key physics: two odd
  fragments are both doublets and SU(2) junctions cannot change total-spin sector
  weights, so recombination must go through the Clebsch–Gordan singlet combination
  of the local multiplets (`apply_s_minus` + `singlet_combination`) — a bare Sz=0
  product start would carry 50% S=1 weight forever.

## Repository layout

| Path | Contents |
|---|---|
| `DnC_VQE_Kagome.ipynb` | The full study — partition, references, both VQE steps, interface microscopy, cost ledger, discussion |
| `kagome_dc.py` | Partition + checker, degenerate-subspace references, frozen-state embedding, matrix-free `RecombinationVQE`, H_SEL, multi-seed wrappers, resource counting, mpl circuit drawings |
| `results/*.npz` | Persisted references and optima (auto-loaded; heavy cells resume mid-grid) |
| `figures/` | Partition map, error curves, interface bond/entropy maps, colored circuits |

Task-1 modules are imported from `../1_Task/` (never copied); `kagome_dc.py` adds that
directory to `sys.path` itself.

## Requirements & running

Same environment as Task 1 (`qiskit>=2.0`, `scipy`, `numpy`, `networkx`, `matplotlib`
+ `pylatexenc` for `draw(output='mpl')`). Deterministic (fixed seeds). First full
**Run All ≈ 1.5–2.5 h** for §1–8 (the Step-2 grids dominate; each saves after every
depth, so an interrupted run resumes where it stopped); the §9 follow-ups add up to
~3 h more if recomputed from scratch, dominated by the 26-qubit junction VQE. With `results/` populated, a re-run takes
minutes (the 26-qubit cells rebuild their statevectors even when loading cached
optima). RAM: the 19-site ED needs ~2–3 GB transient (cached in `results/ed19.npz`)
and the 26-qubit junction VQE peaks at ~4 GB (blocked kernels; the cell checks for
≥4.5 GB free and refuses otherwise).
