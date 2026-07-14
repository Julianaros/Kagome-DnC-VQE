"""
kagome_dc.py
============
Utilities for the Task 2 Proof-of-Concept:

    "Apply the divide-and-conquer VQE strategy of Sec. 4 of the paper
     (local VQE + global VQE, Fig. 4) WITHOUT breaking SU(2): replace the
     paper's junction R_y(θ) gates by Heisenberg/eSWAP gates."

This module COMPLEMENTS `1_Task/kagome_hva.py` (imported below as `K`) and does
NOT duplicate it: the subregion VQEs reuse `K.HVASimulator` verbatim (it already
accepts arbitrary edge/dimer subsets), and the new recombination engine reuses
K's gate matrix (`heis_matrix`), tensor-contraction kernel (`_apply2_sv`) and
adjoint-gradient scheme. What is genuinely new here:

  - the validated 19-site partition (2 subregions + 6 interface bonds) and a
    checker that verifies it against the REAL `edges_19` of the Task-1 notebook;
  - degenerate-subspace exact references (`exact_ground_subspace`) and the
    subspace fidelity metric (lesson from Task-1 Sec. 4.6c: the 19-site ground
    state is a doublet, single-eigenvector fidelities are run-dependent);
  - embedding of the two frozen sub-circuits into one 19-qubit statevector;
  - `RecombinationVQE`: MATRIX-FREE global VQE over the interface gates only
    (H is applied bond by bond, never materialized as a sparse matrix — at 19
    qubits that saves ~1.7 GB of RAM vs the Task-1 approach and costs nothing
    at these depths);
  - `build_h_sel`: the analogue of the advisor's H_SEL (notebook cells 22-23,
    `junct_meas_lst`/`meas_lst_sel`): per junction site, keep the bonds of the
    subgraph induced on {junction} ∪ its immediate neighbours;
  - multi-seed wrappers (≥5 seeds, mean ± std) for both local and global VQEs;
  - resource accounting (params / U_H gates / CNOTs) and mpl circuit drawing
    with the frozen-vs-interface visual distinction of the paper's Fig. 4.

Energy convention identical to Task 1 (the paper's ×4):
H = Σ_(i,j) (X_iX_j + Y_iY_j + Z_iZ_j); a singlet bond scores -3, J=1 uniform
(no Hamiltonian calibration anywhere).

Requires: qiskit>=2.0, scipy, numpy, networkx, matplotlib(+pylatexenc for mpl
circuit drawings).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

# ---- import Task-1 machinery (no duplication) ----------------------------
_TASK1_DIR = Path(__file__).resolve().parent.parent / "1_Task"
if str(_TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK1_DIR))

import kagome_hva as K
from kagome_hva import (_BOND, _apply2_sv, heis_matrix,
                        heisenberg_hamiltonian, pair_correlations)


# ==========================================================================
# 1. The partition (validated against the Task-1 notebook, cell 6)
# ==========================================================================
# Subregion 1: 11 sites, ODD -> doublet target, <S^2> = 0.75.
SUB1_SITES = list(range(0, 11))
SUB1_EDGES = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6),
              (6, 7), (6, 8), (7, 8), (8, 9), (8, 10), (9, 10)]
# 5 singlets; the unpaired spinon sits on site 10, i.e. AT the interface
# (site 10 touches the cut bonds (10,11) and (10,12)).
SUB1_DIMERS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

# Subregion 2: 8 sites, EVEN -> singlet target, <S^2> = 0.
SUB2_SITES = list(range(11, 19))
SUB2_EDGES = [(11, 12), (12, 13), (12, 14), (13, 14), (14, 15), (14, 16),
              (15, 16), (16, 17), (17, 18), (16, 18)]
# Full even cover. It does NOT reuse (12,13),(14,15),(16,17) from the Task-1
# dimers_19 because that would leave sites 11 and 18 orphaned (they have no
# bond between them inside subregion 2).
SUB2_DIMERS = [(11, 12), (13, 14), (15, 16), (17, 18)]

# Interface bonds: cut in Step 1, reconnected by the junction gates in Step 2.
# 3 of the 10 triangles of edges_19 are split by this cut — (0,1,17), (1,2,11)
# and (10,11,12) — hence 6 interface bonds, not 1: several simultaneous
# "junction qubits" in the language of the paper's Fig. 4. Note (1,2,11) and
# (10,11,12) share site 11, so they are not independent junctions.
INTERFACE_EDGES = [(10, 11), (10, 12), (1, 11), (2, 11), (0, 17), (1, 17)]

# Junction sites = every site touched by an interface bond.
JUNCTION_SITES = sorted({s for e in INTERFACE_EDGES for s in e})   # 0,1,2,10,11,12,17

_TASK1_NOTEBOOK = _TASK1_DIR / "PoC_Heisenberg_layers_QSL.ipynb"


def load_task1_lattice(notebook_path=_TASK1_NOTEBOOK):
    """
    Extracts the REAL `edges_19` / `dimers_19` from the Task-1 notebook source
    (the single source of truth for the lattice — same list as the advisor's
    notebook cell 31). Used by `check_partition` so the validation can never
    silently drift from a hand-copied edge list.
    """
    import ast
    nb = json.loads(Path(notebook_path).read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "edges_19 =" in src and "dimers_19 =" in src:
            tree = ast.parse(src)
            keep = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) in ("edges_19", "dimers_19")
                            for t in node.targets)]
            ns: dict = {}
            exec(compile(ast.Module(body=keep, type_ignores=[]), "<task1>", "exec"), ns)
            return ([tuple(e) for e in ns["edges_19"]],
                    [tuple(d) for d in ns["dimers_19"]])
    raise RuntimeError(f"lattice cell not found in {notebook_path}")


def check_partition(edges_full=None, verbose=True):
    """
    Verifies that SUB1_EDGES ∪ SUB2_EDGES ∪ INTERFACE_EDGES == edges_19 as SETS
    (no duplicates, no overlaps, nothing missing). By default `edges_full` is
    loaded from the Task-1 notebook itself. Raises ValueError with the exact
    difference on any mismatch; returns a summary dict on success.
    """
    if edges_full is None:
        edges_full, _ = load_task1_lattice()
    norm = lambda es: {tuple(sorted(e)) for e in es}
    E, S1, S2, IF = (norm(edges_full), norm(SUB1_EDGES),
                     norm(SUB2_EDGES), norm(INTERFACE_EDGES))
    problems = []
    for name, a, b in [("sub1∩sub2", S1, S2), ("sub1∩interface", S1, IF),
                       ("sub2∩interface", S2, IF)]:
        if a & b:
            problems.append(f"overlap {name}: {sorted(a & b)}")
    union = S1 | S2 | IF
    if union != E:
        problems.append(f"missing from union: {sorted(E - union)}; "
                        f"extra in union: {sorted(union - E)}")
    if problems:
        raise ValueError("partition does NOT match edges_19 -> " + " | ".join(problems))
    out = dict(n_sub1=len(S1), n_sub2=len(S2), n_interface=len(IF),
               n_total=len(E), match=True)
    if verbose:
        print(f"partition check: {out['n_sub1']} + {out['n_sub2']} + "
              f"{out['n_interface']} = {out['n_total']} bonds == edges_19  ✓ "
              f"(disjoint, exact set equality)")
    return out


# ==========================================================================
# 2. Local (subregion) simulators — pure reuse of K.HVASimulator
# ==========================================================================
def relabel(pairs, sites):
    """Maps global-site pairs to local indices 0..len(sites)-1 (order of `sites`)."""
    loc = {s: k for k, s in enumerate(sites)}
    return [(loc[a], loc[b]) for a, b in pairs]


def make_subregion_sims():
    """
    Returns (sim1, sim2): K.HVASimulator instances for the two subregions in
    LOCAL indices (sub1: 11 qubits, identity relabel; sub2: 8 qubits, global
    site s -> s-11). All Task-1 machinery (adjoint gradient, sweeps, bond maps)
    works on them unchanged.
    """
    sim1 = K.HVASimulator(len(SUB1_SITES), relabel(SUB1_EDGES, SUB1_SITES),
                          relabel(SUB1_DIMERS, SUB1_SITES))
    sim2 = K.HVASimulator(len(SUB2_SITES), relabel(SUB2_EDGES, SUB2_SITES),
                          relabel(SUB2_DIMERS, SUB2_SITES))
    return sim1, sim2


# ==========================================================================
# 3. Exact references with DEGENERATE subspaces (Task-1 4.6c lesson, applied
#    from the start): fidelity = projection onto the full degenerate manifold.
# ==========================================================================
def exact_ground_subspace(n, edges, k_probe=4, deg_tol=1e-8):
    """
    Sparse Lanczos with k_probe eigenpairs; returns (E0, V, evals) where V's
    columns span the DEGENERATE ground manifold (|E_i - E0| < deg_tol·max(1,|E0|)).
    E.g. subregion 1 (odd, doublet) -> V has 2 columns; subregion 2 (even,
    singlet) -> 1 column. Using the whole manifold makes the fidelity metric
    run-independent (Task-1 4.6c: single-vector fidelities varied 0.08-0.24).
    """
    H = heisenberg_hamiltonian(n, edges).to_matrix(sparse=True).tocsr()
    w, v = eigsh(H, k=k_probe, which="SA")
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    deg = int(np.sum(np.abs(w - w[0]) < deg_tol * max(1.0, abs(w[0]))))
    if deg == k_probe:
        raise RuntimeError(f"ground degeneracy >= k_probe={k_probe}; raise k_probe")
    return float(w[0]), v[:, :deg], w


def subspace_fidelity(sv, V):
    """F = ||V† ψ||² = Σ_i |⟨v_i|ψ⟩|², the weight of ψ inside the manifold span(V)."""
    return float(np.sum(np.abs(V.conj().T @ np.asarray(sv).reshape(-1)) ** 2))


def s2_matrix_free(n, sv):
    """
    ⟨S²⟩ = 3n/4 + 2 Σ_{i<j} ⟨S_i·S_j⟩ = 3n/4 + ½ Σ_{i<j} ⟨(XX+YY+ZZ)_ij⟩, term by
    term on the statevector — never builds the S² operator (whose sparse matrix
    costs ~1.4 GB at n=19 in Task 1). Unlike K.pair_correlations it makes no
    per-pair copy of ψ (K._apply2_sv does not mutate its input), which matters at
    n=26 where each copy is ~1 GB.
    """
    psiT = np.asarray(sv).reshape((2,) * n)
    tot = 3 * n / 4.0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 0.5 * float(np.real(bond_vdot(n, psiT, psiT, _BOND, i, j)))
    return float(tot)


def sz_diagonal(n):
    """Diagonal of S^z_total in the computational basis: |0⟩ = spin up (+1/2)."""
    idx = np.arange(2 ** n, dtype=np.int64)
    pop = np.zeros(2 ** n, dtype=np.int64)
    for b in range(n):
        pop += (idx >> b) & 1
    return (n - 2 * pop) / 2.0


def project_sz(V, n, sz):
    """
    Orthonormal basis of the S^z = sz sector of the manifold span(V) (e.g. the
    S^z=+1/2 component of a doublet, to compare against HVA states, which have
    definite S^z=+1/2 by construction: dimers have S^z=0, the spinon is |0⟩).
    """
    mask = np.isclose(sz_diagonal(n), sz)
    W = V * mask[:, None]
    q, r = np.linalg.qr(W)
    keep = np.abs(np.diag(r)) > 1e-10
    return q[:, keep]


# ==========================================================================
# 4. Embedding the frozen sub-circuits into the 19-qubit register
# ==========================================================================
def embed_product(psi_low, psi_high):
    """
    |ψ_full⟩ = |ψ_high⟩ ⊗ |ψ_low⟩ (i.e. np.kron(high, low)) for Qiskit's
    little-endian convention: subregion 1 lives on qubits 0..10 (low indices),
    subregion 2 on qubits 11..18 (local qubit q -> global q+11). Validated
    against qiskit.quantum_info.Statevector in the module smoke test.
    """
    return np.kron(np.asarray(psi_high).reshape(-1),
                   np.asarray(psi_low).reshape(-1))


def frozen_state(sim1, x1, reps1, sim2, x2, reps2):
    """
    19-qubit statevector of the two FROZEN optimized sub-circuits concatenated
    (interface = identity). reps=0 means the bare dimer state of that region.
    This is the paper's 'local VQE output' that the global VQE starts from.
    """
    sv1 = (sim1.psi0.reshape(-1) if reps1 == 0
           else sim1.statevector(np.asarray(x1), reps1))
    sv2 = (sim2.psi0.reshape(-1) if reps2 == 0
           else sim2.statevector(np.asarray(x2), reps2))
    return embed_product(sv1, sv2)


# ==========================================================================
# 5. H_SEL — the advisor's reduced junction Hamiltonian, adapted
# ==========================================================================
def build_h_sel(edges_full=None, interface_edges=INTERFACE_EDGES):
    """
    Analogue of the advisor's `meas_lst_sel` (notebook cells 22-23): for EACH
    junction site j, keep the bonds of the subgraph induced on {j} ∪ N(j)
    (immediate neighbours), then take the union over junctions. In the
    advisor's 103-qubit circuit each junction is a single qubit and this rule
    produces its local triangle (e.g. junction 18 -> [17,18],[18,19],[17,19]);
    here the junctions are the 7 sites touched by the 6 interface bonds — and
    (1,2,11)/(10,11,12) sharing site 11 is handled automatically by the union.
    Returns the H_SEL bond list (a subset of edges_full, global indices).
    """
    if edges_full is None:
        edges_full, _ = load_task1_lattice()
    edges_full = [tuple(sorted(e)) for e in edges_full]
    junctions = sorted({s for e in interface_edges for s in e})
    nbrs = {j: {b if a == j else a for a, b in edges_full if j in (a, b)}
            for j in junctions}
    sel = set()
    for j in junctions:
        ball = nbrs[j] | {j}
        sel |= {e for e in edges_full if e[0] in ball and e[1] in ball}
    return sorted(sel)


# ==========================================================================
# 6. RecombinationVQE — the 'global VQE' (Step 2), matrix-free
# ==========================================================================
class RecombinationVQE:
    """
    Global VQE of the paper's Sec. 4.2 with SU(2)-preserving junction gates:
    starts from the frozen concatenated state |ψ_frozen⟩ (the Step-1 output,
    absorbed into psi0 — the frozen parameters are never re-optimized) and
    applies layers of Heisenberg gates ONLY on the interface bonds.

    MATRIX-FREE: H|ψ⟩ is evaluated bond by bond with the same 2-qubit kernel
    used for the gates (K._apply2_sv), so no 2^19-dimensional sparse matrix is
    ever built (Task-1's sim19 held ~1.7 GB of sparse operators; here the
    working set stays ≈ a few statevectors ≈ tens of MB). ⟨S²⟩ is likewise
    computed from pair correlations: S² = 3n/4 + 2 Σ_{i<j} ⟨S_i·S_j⟩.

    The adjoint gradient is the same scheme as K.energy_and_grad_schedule
    (shared-parameter accumulation included), with H applied bond-wise.
    """

    def __init__(self, psi_frozen, n=19, edges_full=None,
                 interface_edges=INTERFACE_EDGES, h_sel_edges=None):
        if edges_full is None:
            edges_full, _ = load_task1_lattice()
        self.n = n
        self.bonds_full = [tuple(e) for e in edges_full]
        self.bonds_sel = (build_h_sel(edges_full, interface_edges)
                          if h_sel_edges is None else [tuple(e) for e in h_sel_edges])
        self.interface_edges = [tuple(e) for e in interface_edges]
        self.psi0T = np.asarray(psi_frozen, dtype=complex).reshape((2,) * n)
        # blocked low-memory kernels above ~22 qubits (see Sec. 12); at small n
        # levels=0 makes them exactly equivalent to the standard path
        self.lowmem = n >= 22

    # ---- schedules -------------------------------------------------------
    def schedule(self, reps, per_bond=True):
        """
        Interface gate list [(i, j, pidx)] for `reps` layers over the 6 cut
        bonds. per_bond=True: one θ per bond per layer (n_params = 6·reps),
        the direct analogue of the paper's independent junction R_y angles.
        per_bond=False: ONE shared θ per layer (n_params = reps).
        """
        m = len(self.interface_edges)
        sched = []
        for r in range(reps):
            for k, (i, j) in enumerate(self.interface_edges):
                sched.append((i, j, r * m + k if per_bond else r))
        return sched, (reps * m if per_bond else reps)

    # ---- state and observables --------------------------------------------
    def statevector(self, theta, sched):
        psi = self.psi0T.copy()
        if self.lowmem:
            for (i, j, p) in sched:
                apply2_inplace(self.n, psi, heis_matrix(theta[p]), i, j)
        else:
            for (i, j, p) in sched:
                psi = _apply2_sv(self.n, psi, heis_matrix(theta[p]), i, j)
        return psi.reshape(-1)

    def _apply_H(self, psiT, bonds):
        """Σ_(i,j)∈bonds (XX+YY+ZZ)_ij |ψ⟩, bond by bond (never materializes H)."""
        acc = np.zeros_like(psiT)
        if self.lowmem:
            for (i, j) in bonds:
                apply2_accumulate(self.n, acc, psiT, _BOND, i, j)
        else:
            for (i, j) in bonds:
                acc += _apply2_sv(self.n, psiT, _BOND, i, j)
        return acc

    def energy(self, sv, bonds=None):
        bonds = self.bonds_full if bonds is None else bonds
        psiT = np.asarray(sv).reshape((2,) * self.n)
        if self.lowmem:      # no accumulator vector: blockwise ⟨ψ|B_ij|ψ⟩ sums
            return float(sum(np.real(bond_vdot(self.n, psiT, psiT, _BOND, i, j))
                             for (i, j) in bonds))
        return float(np.real(np.vdot(psiT.reshape(-1),
                                     self._apply_H(psiT, bonds).reshape(-1))))

    def s2(self, sv):
        """⟨S²⟩, matrix- and copy-free (delegates to s2_matrix_free)."""
        return s2_matrix_free(self.n, sv)

    def bond_energies(self, sv, bonds=None):
        """{(i,j): ⟨XX+YY+ZZ⟩_ij} — same convention as K.HVASimulator.bond_energies."""
        bonds = self.bonds_full if bonds is None else bonds
        psiT = np.asarray(sv).reshape((2,) * self.n)
        if self.lowmem:
            return {(i, j): float(np.real(bond_vdot(self.n, psiT, psiT, _BOND, i, j)))
                    for (i, j) in bonds}
        return {(i, j): float(np.real(np.vdot(sv,
                _apply2_sv(self.n, psiT, _BOND, i, j).reshape(-1))))
                for (i, j) in bonds}

    # ---- energy + adjoint gradient (H applied bond-wise) -------------------
    def energy_and_grad(self, theta, sched, n_params, bonds=None):
        """
        (E, ∂E/∂θ) over the interface angles, same adjoint scheme as Task-1
        (K.energy_and_grad_schedule): one forward + one backward pass, gradient
        accumulated at shared parameter indices; cost ≈ 3 energy evaluations
        independent of n_params.
        """
        bonds = self.bonds_full if bonds is None else bonds
        M = _BOND / 4.0
        if self.lowmem:
            # peak footprint: psi0T + psiT + bT + one block (vs ~6 full vectors
            # in the standard path — the difference between OOM and fitting at 2^26)
            psiT = self.psi0T.copy()
            for (i, j, p) in sched:
                apply2_inplace(self.n, psiT, heis_matrix(theta[p]), i, j)
            bT = self._apply_H(psiT, bonds)
            E = float(np.real(np.vdot(psiT.reshape(-1), bT.reshape(-1))))
            grad = np.zeros(n_params)
            for (i, j, p) in reversed(sched):
                grad[p] += 2.0 * np.real(-1j * bond_vdot(self.n, bT, psiT, M, i, j))
                apply2_inplace(self.n, psiT, heis_matrix(-theta[p]), i, j)
                apply2_inplace(self.n, bT, heis_matrix(-theta[p]), i, j)
            return E, grad
        psiT = self.psi0T.copy()
        for (i, j, p) in sched:
            psiT = _apply2_sv(self.n, psiT, heis_matrix(theta[p]), i, j)
        bT = self._apply_H(psiT, bonds)
        E = float(np.real(np.vdot(psiT.reshape(-1), bT.reshape(-1))))
        grad = np.zeros(n_params)
        for (i, j, p) in reversed(sched):
            Mp = _apply2_sv(self.n, psiT, M, i, j).reshape(-1)
            grad[p] += 2.0 * np.real(-1j * np.vdot(bT.reshape(-1), Mp))
            psiT = _apply2_sv(self.n, psiT, heis_matrix(-theta[p]), i, j)
            bT = _apply2_sv(self.n, bT, heis_matrix(-theta[p]), i, j)
        return E, grad

    def gradient_at_zero(self, reps=1, per_bond=True, bonds=None):
        """
        ‖∇E‖ at θ=0 (interface = identity). If both frozen sub-states are exact
        local eigenstates, the concatenation is a STATIONARY point of the
        interface optimization (⟨[A, H]⟩ = 0 for local eigenstates), so this
        should be ≈ 0 — the measured justification for random (not warm) starts.
        """
        sched, npar = self.schedule(reps, per_bond)
        E, g = self.energy_and_grad(np.zeros(npar), sched, npar, bonds=bonds)
        return E, float(np.linalg.norm(g))

    # ---- multi-seed optimization -------------------------------------------
    def optimize(self, reps, e_ref, per_bond=True, use_sel=False,
                 seeds=(0, 1, 2, 3, 4), n_starts=1, zero_start=True,
                 init_scale=0.5, maxiter=200, V_ref=None, compute_s2=True,
                 verbose=True):
        """
        Step-2 optimization: L-BFGS-B + adjoint gradient over the interface
        angles only, one independent run per seed. Each seed takes the best of
        `n_starts` random inits ~ U(-init_scale, init_scale) — random because
        θ=0 is (near-)stationary, see gradient_at_zero — plus, if `zero_start`,
        one L-BFGS run from θ=0 (cheap: it terminates immediately when the
        start is stationary) and the raw θ=0 fallback, so the reported result
        can never end ABOVE the frozen state. use_sel=True optimizes ⟨H_SEL⟩
        (the advisor's reduced junction Hamiltonian) — the reported
        energy/err_pct are ALWAYS ⟨H_full⟩, so both variants are comparable.
        e_ref: exact full-lattice ground energy. V_ref: optional degenerate
        ground manifold for subspace fidelity. Returns a list of per-seed dicts
        (energy, err_pct, s2, fidelity, x, nit, nfev, time_s, ...).
        """
        sched, npar = self.schedule(reps, per_bond)
        bonds_opt = self.bonds_sel if use_sel else None
        obj = lambda z: self.energy_and_grad(z, sched, npar, bonds=bonds_opt)
        runs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            t0 = time.perf_counter()
            inits = [rng.uniform(-init_scale, init_scale, npar)
                     for _ in range(n_starts)]
            if zero_start:
                inits.append(np.zeros(npar))
            fbest, xbest = obj(np.zeros(npar))[0], np.zeros(npar)  # frozen fallback
            nit = nfev = 0
            for x0 in inits:
                res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiter})
                nit += int(res.nit); nfev += int(res.nfev)
                if res.fun < fbest:
                    fbest, xbest = float(res.fun), res.x
            dt = time.perf_counter() - t0
            sv = self.statevector(xbest, sched)
            E = self.energy(sv)                       # always the FULL H
            rec = dict(seed=int(s), reps=int(reps), n_params=int(npar),
                       per_bond=bool(per_bond), use_sel=bool(use_sel),
                       obj=float(fbest), energy=E,
                       err_pct=abs(E - e_ref) / abs(e_ref) * 100.0,
                       s2=(self.s2(sv) if compute_s2 else np.nan),
                       fidelity=(subspace_fidelity(sv, V_ref)
                                 if V_ref is not None else np.nan),
                       nit=nit, nfev=nfev, time_s=dt, x=np.asarray(xbest))
            runs.append(rec)
            if verbose:
                tag = "H_SEL" if use_sel else "H_full"
                print(f"  seed {s}: E={E:9.4f}  err={rec['err_pct']:6.3f}%  "
                      f"<S^2>={rec['s2']:.4f}  fid={rec['fidelity']:.4f}  "
                      f"[{tag}, {npar}p, {nit} its, {dt:5.1f}s]")
        return runs

    def trace_run(self, reps, e_ref, seed=0, per_bond=True, use_sel=True,
                  maxiter=300, init_scale=0.5, verbose=True):
        """
        ONE junction-VQE run that RECORDS its trajectory (the analogue of the
        advisor's Fig. 6(a)): at every L-BFGS iteration it stores the optimized
        objective ⟨H_SEL⟩ and the FULL energy ⟨H_full⟩, so one can show that
        minimizing the reduced junction Hamiltonian drives the full energy to
        the benchmark. Costs ~2 extra energy evaluations per iteration.
        Returns dict(obj, full, x, energy, err_pct, nit, ...); obj[0]/full[0]
        are the (random) starting point.
        """
        sched, npar = self.schedule(reps, per_bond)
        bonds_opt = self.bonds_sel if use_sel else self.bonds_full
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-init_scale, init_scale, npar)
        obj_tr, full_tr = [], []

        def record(xk):
            sv = self.statevector(xk, sched)
            full = self.energy(sv)
            obj_tr.append(self.energy(sv, bonds_opt) if use_sel else full)
            full_tr.append(full)

        record(x0)
        res = minimize(lambda z: self.energy_and_grad(z, sched, npar,
                                                      bonds=bonds_opt if use_sel else None),
                       x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter}, callback=record)
        sv = self.statevector(res.x, sched)
        E = self.energy(sv)
        out = dict(obj=np.asarray(obj_tr), full=np.asarray(full_tr),
                   x=np.asarray(res.x), energy=E,
                   err_pct=abs(E - e_ref) / abs(e_ref) * 100.0,
                   nit=int(res.nit), seed=int(seed), reps=int(reps),
                   use_sel=bool(use_sel), n_params=int(npar))
        if verbose:
            print(f"traced run (seed {seed}, {npar}p, "
                  f"{'H_SEL' if use_sel else 'H_full'}): {out['nit']} its, "
                  f"E={E:.6f}  err={out['err_pct']:.3f}%")
        return out

    def refine(self, runs, e_ref, maxiter=400, V_ref=None, compute_s2=True,
               verbose=True):
        """
        PROLONGS saved runs whose optimizer hit its iteration cap: restarts
        L-BFGS-B from each run's stored optimum x (same schedule, same objective
        — H_SEL if the run used it) with a fresh `maxiter` budget. Returns new
        records in the same format (nit/nfev/time_s count only the extension).
        Must be called on the same RecombinationVQE instance (same frozen state)
        that produced the runs.
        """
        out = []
        for r in runs:
            sched, npar = self.schedule(r["reps"], r["per_bond"])
            bonds_opt = self.bonds_sel if r["use_sel"] else None
            obj = lambda z: self.energy_and_grad(z, sched, npar, bonds=bonds_opt)
            t0 = time.perf_counter()
            res = minimize(obj, np.asarray(r["x"]), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            dt = time.perf_counter() - t0
            sv = self.statevector(res.x, sched)
            E = self.energy(sv)
            rec = dict(seed=r["seed"], reps=r["reps"], n_params=npar,
                       per_bond=r["per_bond"], use_sel=r["use_sel"],
                       obj=float(res.fun), energy=E,
                       err_pct=abs(E - e_ref) / abs(e_ref) * 100.0,
                       s2=(self.s2(sv) if compute_s2 else np.nan),
                       fidelity=(subspace_fidelity(sv, V_ref)
                                 if V_ref is not None else np.nan),
                       nit=int(res.nit), nfev=int(res.nfev), time_s=dt,
                       x=np.asarray(res.x))
            out.append(rec)
            if verbose:
                print(f"  seed {r['seed']}: err {r['err_pct']:6.3f}% -> "
                      f"{rec['err_pct']:6.3f}%  (+{rec['nit']} its, {dt:5.1f}s)  "
                      f"<S^2>={rec['s2']:.4f}  fid={rec['fidelity']:.4f}")
        return out


# ==========================================================================
# 7. Multi-seed local sweeps (Step 1) and aggregation helpers
# ==========================================================================
def run_local_multiseed(sim, E_exact, V_exact=None, seeds=(0, 1, 2, 3, 4),
                        max_reps=4, n_random=2, maxiter=300, verbose=True):
    """
    Repeats K.run_hva_sweep_grad once per seed (independent restarts) and
    replaces the single-vector fidelity by the SUBSPACE fidelity onto V_exact
    (degenerate manifold) — the Task-1 4.6c metric, applied from the start.
    Returns a list of sweeps (one per seed, Task-1 result format + time_s).
    """
    sweeps = []
    for s in seeds:
        t0 = time.perf_counter()
        res = K.run_hva_sweep_grad(sim, E_exact, psi_exact=None,
                                   max_reps=max_reps, n_random=n_random,
                                   maxiter=maxiter, seed=s, verbose=False)
        dt = time.perf_counter() - t0
        for r in res:
            sv = (sim.psi0.reshape(-1) if r["reps"] == 0
                  else sim.statevector(r["x"], r["reps"]))
            r["fidelity"] = (subspace_fidelity(sv, V_exact)
                             if V_exact is not None else np.nan)
            r["time_s"] = dt
        sweeps.append(res)
        if verbose:
            last = res[-1]
            print(f"  seed {s}: reps={last['reps']}  err={last['err_pct']:8.4f}%  "
                  f"<S^2>={last['s2']:.4f}  fid={last['fidelity']:.4f}  ({dt:5.1f}s)")
    return sweeps


def aggregate_sweeps(sweeps, key="err_pct"):
    """
    Per-depth statistics across seeds: {reps: (mean, std, best_value, best_seed)}.
    Works on the output of run_local_multiseed (key: err_pct / s2 / fidelity /
    energy).
    """
    out = {}
    for reps in [r["reps"] for r in sweeps[0]]:
        vals = np.array([next(r[key] for r in sw if r["reps"] == reps)
                         for sw in sweeps], dtype=float)
        b = int(np.argmin(vals)) if key in ("err_pct", "energy") else int(np.argmax(vals))
        out[reps] = (float(vals.mean()), float(vals.std()), float(vals[b]), b)
    return out


def best_of_sweeps(sweeps, reps):
    """(x, seed_index) of the lowest-energy run at depth `reps` across seeds."""
    recs = [next(r for r in sw if r["reps"] == reps) for sw in sweeps]
    b = int(np.argmin([r["energy"] for r in recs]))
    return recs[b]["x"], b


def summarize_runs(runs, keys=("err_pct", "s2", "fidelity", "time_s", "nit")):
    """mean ± std over a list of per-seed dicts (RecombinationVQE.optimize output)."""
    return {k: (float(np.mean([r[k] for r in runs])),
                float(np.std([r[k] for r in runs]))) for k in keys}


# ==========================================================================
# 8. Resource accounting (params / U_H / CNOTs)
# ==========================================================================
def resources(n_uh, n_dimers, n_free_params, label=""):
    """
    Gate budget of a stage. Each U_H(θ) compiles to 3 CNOTs (standard eSWAP
    decomposition, as in Task 1); each singlet preparation uses 1 CNOT
    (+ H, X, Z singles). n_free_params counts only the parameters OPTIMIZED
    SIMULTANEOUSLY at that stage — the divide-and-conquer headline number.
    """
    return dict(label=label, params=int(n_free_params), u_h=int(n_uh),
                cnot=int(3 * n_uh + n_dimers), dimers=int(n_dimers))


# ==========================================================================
# 9. Persistence for recombination runs (Task-1 npz pattern)
# ==========================================================================
_RUN_META = ["seed", "reps", "n_params", "per_bond", "use_sel", "obj", "energy",
             "err_pct", "s2", "fidelity", "nit", "nfev", "time_s"]


def save_runs(runs, path):
    """Saves a list of per-seed run dicts (+their x vectors) to one .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": np.array([[float(r[k]) for k in _RUN_META] for r in runs])}
    for i, r in enumerate(runs):
        payload[f"x_{i}"] = np.asarray(r["x"], dtype=float)
    np.savez(path, **payload)


def load_runs(path):
    """Inverse of save_runs."""
    d = np.load(path)
    runs = []
    for i, row in enumerate(d["meta"]):
        r = {k: (bool(v) if k in ("per_bond", "use_sel") else
                 (int(v) if k in ("seed", "reps", "n_params", "nit", "nfev") else float(v)))
             for k, v in zip(_RUN_META, row)}
        r["x"] = d[f"x_{i}"]
        runs.append(r)
    return runs


_SWEEP_META = ["seed_idx", "reps", "n_params", "energy", "err_pct", "s2",
               "fidelity", "time_s"]


def save_sweeps(sweeps, path):
    """
    Saves the output of run_local_multiseed (list of Task-1-format sweeps, one
    per seed) to a single .npz: meta rows + one x vector per (seed, reps).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta, payload = [], {}
    for si, sw in enumerate(sweeps):
        for r in sw:
            meta.append([si, r["reps"], r["n_params"], r["energy"],
                         r["err_pct"], r["s2"], r["fidelity"],
                         r.get("time_s", np.nan)])
            payload[f"x_{si}_{r['reps']}"] = np.asarray(r["x"], dtype=float)
    payload["meta"] = np.asarray(meta, dtype=float)
    np.savez(path, **payload)


def load_sweeps(path):
    """Inverse of save_sweeps: returns the list-of-sweeps (one per seed)."""
    d = np.load(path)
    sweeps: dict = {}
    for row in d["meta"]:
        si, reps = int(row[0]), int(row[1])
        rec = dict(zip(_SWEEP_META, row))
        rec.update(seed_idx=si, reps=reps, n_params=int(row[2]),
                   x=d[f"x_{si}_{reps}"])
        sweeps.setdefault(si, []).append(rec)
    return [sorted(sw, key=lambda r: r["reps"]) for _, sw in sorted(sweeps.items())]


# ==========================================================================
# 10. Circuit drawings (mpl, Fig.-4-style frozen vs interface distinction)
# ==========================================================================
# Palette shared with the notebook figures.
COLOR_R1, COLOR_R2 = "#2a78d6", "#1baf7a"     # subregion 1 / subregion 2
COLOR_IFC, COLOR_T1 = "#e34948", "#8a8a86"    # interface gates / Task-1 grey

MPL_STYLE = {"displaycolor": {
    "UH-R1": (COLOR_R1, "#ffffff"),
    "UH-R2": (COLOR_R2, "#ffffff"),
    "UH-J":  (COLOR_IFC, "#ffffff"),
    "sub1-frozen": (COLOR_R1, "#ffffff"),
    "sub2-frozen": (COLOR_R2, "#ffffff"),
}}


def _uh_box(name, theta):
    """Opaque 2-qubit gate named for the mpl drawer (visual only, not simulated)."""
    from qiskit.circuit import Gate
    return Gate(name, 2, [float(theta)])


def _prep_and_layers(qc, dimers, edges, x, reps, gate_name, offset=0):
    for a, b in dimers:
        qc.h(a + offset); qc.x(b + offset); qc.cx(a + offset, b + offset)
        qc.z(a + offset)
    k = 0
    for _ in range(reps):
        for (i, j) in edges:
            qc.append(_uh_box(gate_name, x[k]), [i + offset, j + offset])
            k += 1


def build_region_drawing(region, x, reps):
    """
    Drawing circuit (local indices) of an optimized subregion: region=1 -> blue
    UH-R1 boxes on 11 qubits; region=2 -> green UH-R2 on 8 qubits. Draw with
    qc.draw('mpl', style=MPL_STYLE, fold=...).
    """
    from qiskit import QuantumCircuit
    if region == 1:
        n, edges, dimers, name = 11, relabel(SUB1_EDGES, SUB1_SITES), \
            relabel(SUB1_DIMERS, SUB1_SITES), "UH-R1"
    else:
        n, edges, dimers, name = 8, relabel(SUB2_EDGES, SUB2_SITES), \
            relabel(SUB2_DIMERS, SUB2_SITES), "UH-R2"
    qc = QuantumCircuit(n)
    _prep_and_layers(qc, dimers, edges, np.atleast_1d(np.asarray(x, dtype=float)),
                     reps, name)
    return qc


def build_recombined_compact(theta_ifc, reps_ifc, per_bond=True,
                             label1="Sub-1 frozen (Step 1)",
                             label2="Sub-2 dimers (exact)"):
    """
    Fig.-4-style ABBREVIATED recombined circuit: each frozen sub-circuit is
    collapsed into ONE labeled block (the paper's 'shaded' part — Fig. 4 does
    not draw its 549 terms either) and only the Step-2 junction gates appear
    in detail (red). This is the presentation figure; the fully detailed
    drawing (build_recombined_drawing) is ~6000 px tall at default folding and
    belongs in supplementary material only. Draw with fold=-1 so it fits one
    page/screen.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    qc = QuantumCircuit(19)
    qc.append(Gate("sub1-frozen", 11, [], label=label1), list(range(11)))
    qc.append(Gate("sub2-frozen", 8, [], label=label2), list(range(11, 19)))
    qc.barrier(label="freeze")
    th = np.atleast_1d(np.asarray(theta_ifc, dtype=float))
    m = len(INTERFACE_EDGES)
    for r in range(reps_ifc):
        for k, (i, j) in enumerate(INTERFACE_EDGES):
            qc.append(_uh_box("UH-J", th[r * m + k if per_bond else r]), [i, j])
    return qc


def build_recombined_drawing(x1, reps1, x2, reps2, theta_ifc, reps_ifc,
                             per_bond=True):
    """
    Full 19-qubit drawing circuit: frozen sub-circuits (blue/grey-blue UH-R1 on
    qubits 0-10, green UH-R2 on 11-18 — the paper Fig. 4 'shaded' gates) then a
    barrier and the interface layers in red (UH-J, the 'unshaded' newly
    optimized junction gates). Colors come from MPL_STYLE.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(19)
    _prep_and_layers(qc, relabel(SUB1_DIMERS, SUB1_SITES),
                     relabel(SUB1_EDGES, SUB1_SITES),
                     np.atleast_1d(np.asarray(x1, dtype=float)), reps1, "UH-R1")
    _prep_and_layers(qc, relabel(SUB2_DIMERS, SUB2_SITES),
                     relabel(SUB2_EDGES, SUB2_SITES),
                     np.atleast_1d(np.asarray(x2, dtype=float)), reps2, "UH-R2",
                     offset=11)
    qc.barrier(label="freeze")
    th = np.atleast_1d(np.asarray(theta_ifc, dtype=float))
    m = len(INTERFACE_EDGES)
    for r in range(reps_ifc):
        for k, (i, j) in enumerate(INTERFACE_EDGES):
            qc.append(_uh_box("UH-J", th[r * m + k if per_bond else r]), [i, j])
    return qc


# ==========================================================================
# 11. Scaling follow-up: the 26-site star chain (divide once more)
# ==========================================================================
# kagome_mps.kagome_star_chain(k) builds k hexagram cells in creation order, so
# chain(k-1) is a strict PREFIX of chain(k): its sites are 0..n_sub-1 and its
# edges a subset. That makes the DnC partition of the 26-site chain (k=3)
# canonical: subregion 1 = the whole 19-site chain(2) — whose optimum we already
# own from Task 1 — subregion 2 = the 7 new sites, and the interface is just the
# handful of new-cell bonds that touch the two old tip sites shared by the cells.

def star_chain_partition(n_cells):
    """
    DnC partition of the n_cells star chain: sub1 = chain(n_cells-1) (prefix
    nesting VERIFIED, raises if the generator ever changes), sub2 = the new
    sites of the last cell. Returns a dict with n, edges (full), n_sub1,
    sub1_edges, sub2_sites, sub2_edges, interface_edges (all sorted tuples).
    """
    import kagome_mps as M
    n_full, e_full, _ = M.kagome_star_chain(n_cells)
    n_sub, e_sub, _ = M.kagome_star_chain(n_cells - 1)
    E_full = {tuple(e) for e in e_full}
    E_sub = {tuple(e) for e in e_sub}
    if not E_sub <= E_full:
        raise RuntimeError("star-chain prefix nesting broken: chain(k-1) ⊄ chain(k)")
    rest = E_full - E_sub
    old = lambda s: s < n_sub
    bad = [e for e in rest if old(e[0]) and old(e[1])]
    if bad:
        raise RuntimeError(f"new-cell bonds between old sites: {bad}")
    return dict(n=n_full, edges=sorted(E_full), n_sub1=n_sub,
                sub1_edges=sorted(E_sub),
                sub2_sites=list(range(n_sub, n_full)),
                sub2_edges=sorted(e for e in rest if not old(e[0]) and not old(e[1])),
                interface_edges=sorted(e for e in rest if old(e[0]) != old(e[1])))


def graph_isomorphism(edges_a, edges_b):
    """
    One graph isomorphism {node_a -> node_b} (VF2). Any isomorphism is physically
    valid here: the Heisenberg H depends only on the edge set, so relabeling an
    eigenstate of H(edges_a) by the mapping yields an eigenstate of H(edges_b).
    Raises ValueError if the graphs are not isomorphic.
    """
    import networkx as nx
    GA = nx.Graph([tuple(e) for e in edges_a])
    GB = nx.Graph([tuple(e) for e in edges_b])
    gm = nx.algorithms.isomorphism.GraphMatcher(GA, GB)
    if not gm.is_isomorphic():
        raise ValueError("graphs are not isomorphic")
    return dict(gm.mapping)


def permute_qubits(sv, n, perm):
    """
    Statevector relabeling: returns ψ' where NEW qubit q carries the state of
    OLD qubit perm[q]. Implemented as one np.transpose on the (2,)*n tensor
    (axis a holds qubit n-1-a in the Qiskit convention). Validated against an
    explicit basis-state loop in the module smoke test.
    """
    psiT = np.asarray(sv).reshape((2,) * n)
    axes = [0] * n
    for q_new in range(n):
        axes[n - 1 - q_new] = n - 1 - perm[q_new]
    return np.transpose(psiT, axes).reshape(-1)


def task1_hybrid_state():
    """
    Rebuilds the best Task-1 19-site state (hybrid sym->full, 0.52% error, 180
    params, reps=6) from 1_Task/results/res_hyb.npz — in the ADVISOR labeling
    (edges_19/dimers_19). Returns (statevector, stored_energy). This is the
    'already-paid-for' Step-1 output of the 26-site pipeline: to use it there,
    relabel it with permute_qubits + graph_isomorphism(chain(2) -> edges_19).
    """
    edges_19, dimers_19 = load_task1_lattice()
    d = np.load(_TASK1_DIR / "results" / "res_hyb.npz")
    x = d["x_6"]
    # bypass HVASimulator.__init__: it builds the sparse H and S² operators
    # (~1.7 GB at 19 qubits) that the state reconstruction never touches
    sim = K.HVASimulator.__new__(K.HVASimulator)
    sim.n, sim.edges = 19, [tuple(e) for e in edges_19]
    sim.dimers = [tuple(dm) for dm in dimers_19]
    sim.psi0 = sim._dimer_state().reshape((2,) * 19)
    return sim.statevector(x, 6), float(d["meta"][0][2])


def available_ram_gb():
    """MemAvailable from /proc/meminfo, in GB (the 26-qubit cells check this)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def apply_s_minus(sv, n):
    """
    (S⁻_total)|ψ⟩ = (Σ_i σ⁻_i)|ψ⟩ with the |0⟩ = spin-up convention (σ⁻=|1⟩⟨0|),
    matrix-free. For an exact S=1/2, S_z=+1/2 eigenstate this returns its
    normalized S_z=−1/2 partner (‖S⁻|1/2,+1/2⟩‖ = 1); callers should still
    re-normalize defensively.
    """
    psiT = np.asarray(sv).reshape((2,) * n)
    out = np.zeros_like(psiT)
    for q in range(n):
        ax = n - 1 - q
        sl0 = [slice(None)] * n; sl0[ax] = 0
        sl1 = [slice(None)] * n; sl1[ax] = 1
        out[tuple(sl1)] += psiT[tuple(sl0)]
    return out.reshape(-1)


def singlet_combination(psi_a_up, psi_a_dn, psi_b_up, psi_b_dn):
    """
    Clebsch-Gordan S=0 combination of two spin-1/2 multiplets,
    (|a↑⟩|b↓⟩ − |a↓⟩|b↑⟩)/√2, with region a on the LOW qubits (embed_product
    convention). This is the sector-aware recombination that SU(2)-preserving
    junctions REQUIRE when both fragments are odd (doublets): junction gates
    commute with S², so the total-spin sector weights of the start are frozen
    forever — a bare S_z=0 product start carries 50% S=1 weight it can never
    shed, while this combination is exact S=0 (the sector of the even-N ground
    state). The paper's R_y junctions never face this bookkeeping — they simply
    break SU(2).
    """
    out = embed_product(psi_a_up, psi_b_dn)
    tmp = embed_product(psi_a_dn, psi_b_up)
    out -= tmp                       # in place: peak = 2 full vectors, not 4
    del tmp
    out /= np.linalg.norm(out)
    return out


# ==========================================================================
# 12. Low-memory blocked kernels (what makes n >= 22 fit in 8 GB of RAM)
# ==========================================================================
# The standard kernel (K._apply2_sv) materializes two full-size temporaries per
# gate application; inside the adjoint gradient that stacks up to ~6 vectors
# alive at once — at 2^26 (1.07 GB each) that OOMs an 8 GB machine (measured).
# These kernels process the (2,)*n tensor in 2^levels disjoint blocks over
# axes not touched by the gate (a 2-qubit gate cannot mix such blocks), so the
# transient footprint per operation drops to O(2 blocks). With levels=0 they
# reduce EXACTLY to the standard kernel (verified to 1e-12 at 19 qubits), so
# small systems keep byte-identical behavior.

import itertools as _it


def _lowmem_levels(n):
    """0 below 22 qubits (standard path); 2..6 blocks-exponent above."""
    return 0 if n < 22 else min(6, n - 20)


def _apply2_axes(T, G, ax0, ax1):
    """Out-of-place 4x4 gate on tensor AXES (ax0, ax1) of T (any block size)."""
    T2 = np.moveaxis(T, [ax0, ax1], [0, 1])
    sh = T2.shape
    out = (G @ T2.reshape(4, -1)).reshape(sh)
    return np.moveaxis(out, [0, 1], [ax0, ax1])


def _block_slices(n, axa, axb, levels):
    """Index tuples that split the first `levels` axes ∉ {axa, axb} into
    size-1 slices (slices, not ints, so axis numbering stays stable)."""
    split = [ax for ax in range(n) if ax not in (axa, axb)][:levels]
    for bits in _it.product((0, 1), repeat=len(split)):
        idx = [slice(None)] * n
        for ax, b in zip(split, bits):
            idx[ax] = slice(b, b + 1)
        yield tuple(idx)


def apply2_inplace(n, T, G, a, b, levels=None):
    """T <- (G on QUBITS a,b)·T, blockwise IN PLACE (T is a (2,)*n array)."""
    axa, axb = n - 1 - a, n - 1 - b
    levels = _lowmem_levels(n) if levels is None else levels
    for idx in _block_slices(n, axa, axb, levels):
        T[idx] = _apply2_axes(T[idx], G, axa, axb)
    return T


def apply2_accumulate(n, accT, ketT, G, a, b, levels=None):
    """accT += (G on qubits a,b)·ketT, blockwise (no full-size temporaries)."""
    axa, axb = n - 1 - a, n - 1 - b
    levels = _lowmem_levels(n) if levels is None else levels
    for idx in _block_slices(n, axa, axb, levels):
        accT[idx] += _apply2_axes(ketT[idx], G, axa, axb)
    return accT


def bond_vdot(n, braT, ketT, G, a, b, levels=None):
    """⟨bra|(G on qubits a,b)|ket⟩ accumulated block by block."""
    axa, axb = n - 1 - a, n - 1 - b
    levels = _lowmem_levels(n) if levels is None else levels
    tot = 0j
    for idx in _block_slices(n, axa, axb, levels):
        tot += np.vdot(braT[idx], _apply2_axes(ketT[idx], G, axa, axb))
    return complex(tot)
