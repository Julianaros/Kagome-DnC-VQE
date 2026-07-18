"""
kagome_er.py
============
Utilities for Task 3: Entanglement Recovery (ER).

    Advisor's brief (verbatim, reply to Task 2):
    (i)  "Use HVA to convert dimers to the spin-liquid like state for the lower
         fragment (fragment-2) and then recombine. I am sure the combination
         will be seamless i.e. Heisenberg gate."
    (ii) "Entanglement Recovery: If necessary, modify Hamiltonian (not J->J')
         but by instead adding terms of next nearest neighbours (NNN) terms and
         you will see the non-local correlations will bring us closer to exact
         gs_energy."

This module COMPLEMENTS `1_Task/kagome_hva.py` (imported as `K`) and
`2_Task/kagome_dc.py` (imported as `dc`) and does NOT duplicate them. The Task-2
partition, the matrix-free `RecombinationVQE`, the persistence formats and the
degenerate-subspace metrics are reused verbatim. What is genuinely new here:

  - programmatic NNN geometry (`nnn_pairs`): next-nearest-neighbour pairs are
    derived as graph-distance-2 pairs of the induced subgraph — never a
    hand-written list;
  - the TRAINING Hamiltonian  H_train = H_NN + lambda * H_NNN  (both terms in
    the project's x4 convention, so lambda = J2/J1 directly), built by passing
    `weights` to `K.HVASimulator` — the metric Hamiltonian is ALWAYS the
    original uncalibrated H (J=1 on the real bonds; this is the advisor's
    "not J->J'" discipline);
  - fragment-2 dressing: HVA layers on the fragment's NN bonds only, trained
    against H_train with the Task-1 adjoint gradient (`train_fragment`), and
    the frozen dressed fragment state (`dressed_fragment_state`);
  - correlation profiling against the 19-site ED ground state restricted to
    the fragment (`ed_fragment_targets`, `fragment_profile`, `profile_rms`):
    the numbers that turn "spin-liquid-like" into a measurable target;
  - a multi-seed optimizer for ARBITRARY gate schedules on RecombinationVQE
    (`optimize_schedule`) — the engine's `energy_and_grad` already accepts
    mixed fragment+junction schedules, only the multi-seed wrapper was
    interface-only; and the mixed schedule builder (`dressed_schedule`);
  - the config-C pipeline (`config_c`): pre-train fragment 2 at a given
    lambda, freeze it, junction-VQE only — the clean test of the advisor's
    "seamless" prediction — plus persistence for the lambda sweep;
  - the config-D pipeline (`fit_fragment_profile`, `config_d`): dress the
    fragment by DIRECT correlation fit to the ED target profile (motivated by
    the Sec.-3 pre-analysis: no fragment-local lambda has the target profile
    as its ground state, yet the fragment HVA can represent it);
  - the alternative SU(2) sectorization for the 26-site chain: a GLOBAL dimer
    cover with one singlet seated on an interface bond (`interface_dimer_cover`,
    `dimer_cover_state`) — reaching the S=0 sector without the Clebsch-Gordan
    combination (only possible when both fragments are odd: parity);
  - the Fig.-4-style drawing of the dressed recombined circuit
    (`build_dressed_drawing`).

Energy convention identical to Tasks 1-2 (the paper's x4):
H = sum_(i,j) (X_iX_j + Y_iY_j + Z_iZ_j); a singlet bond scores -3.
Correlation profiles use <S_i.S_j> = <XX+YY+ZZ>/4 (pure singlet: -0.75),
the `K.pair_correlations` convention.

Requires: qiskit>=2.0, scipy, numpy, networkx, matplotlib.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

# ---- import Task-1 + Task-2 machinery (no duplication) --------------------
_TASK2_DIR = Path(__file__).resolve().parent.parent / "2_Task"
if str(_TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK2_DIR))

import kagome_dc as dc          # adds ../1_Task to sys.path itself
import kagome_hva as K
from kagome_hva import _BOND, heis_matrix

_TASK2_RESULTS = _TASK2_DIR / "results"

# ==========================================================================
# 1. NNN geometry — derived, never hand-written
# ==========================================================================
def nnn_pairs(edges):
    """
    Next-nearest-neighbour pairs of the graph spanned by `edges`: all pairs at
    graph distance EXACTLY 2 (shortest path through one intermediate site,
    no direct bond). Returns sorted tuples, sorted. This is the programmatic
    definition of the advisor's "NNN terms" — on the fragment it uses the real
    fragment bonds, so it can never drift from the lattice.
    """
    import networkx as nx
    G = nx.Graph([tuple(e) for e in edges])
    d = dict(nx.all_pairs_shortest_path_length(G, cutoff=2))
    return sorted({tuple(sorted((u, v))) for u in d for v, l in d[u].items()
                   if l == 2})


def fragment2_nnn():
    """
    (global_pairs, local_pairs) of fragment 2 (Task-2 subregion 2, sites
    11..18): NNN pairs of the subgraph induced by SUB2_EDGES. Local labels are
    the fragment simulator's (global site s -> s-11). 10 pairs.
    """
    g = nnn_pairs(dc.SUB2_EDGES)
    return g, dc.relabel(g, dc.SUB2_SITES)


# ==========================================================================
# 2. Exact references for weighted Hamiltonians
# ==========================================================================
def exact_ground_subspace_weighted(n, edges, weights, k_probe=4, deg_tol=1e-8):
    """
    dc.exact_ground_subspace generalized to weighted bonds (it hard-codes
    uniform J). Same return contract: (E0, V, evals) with V spanning the
    degenerate ground manifold. Used for the H_train references of the
    fragment pre-training (2^8: instant).
    """
    H = K.heisenberg_hamiltonian(n, edges, weights).to_matrix(sparse=True).tocsr()
    w, v = eigsh(H, k=k_probe, which="SA")
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    deg = int(np.sum(np.abs(w - w[0]) < deg_tol * max(1.0, abs(w[0]))))
    if deg == k_probe:
        raise RuntimeError(f"ground degeneracy >= k_probe={k_probe}; raise k_probe")
    return float(w[0]), v[:, :deg], w


# ==========================================================================
# 3. Fragment-2 dressing: train against H_NN + lambda * H_NNN
# ==========================================================================
# The decoupling that makes this free (verified in the Task-3 pre-flight):
# `K.HVASimulator(n, edges, dimers, weights)` builds its Hamiltonian from
# (edges, weights), while `K.statevector_schedule` / `K.energy_and_grad_schedule`
# take an EXPLICIT gate list — so the ansatz gates can live on the NN bonds
# only while the objective includes the weighted NNN terms. The ansatz is never
# modified (the advisor modifies the Hamiltonian, not the circuit).

def _frag_geometry():
    """(nn_local, nnn_local, dimers_local) of fragment 2, all in local labels."""
    nn = dc.relabel(dc.SUB2_EDGES, dc.SUB2_SITES)
    dm = dc.relabel(dc.SUB2_DIMERS, dc.SUB2_SITES)
    return nn, fragment2_nnn()[1], dm


def training_sim(lam):
    """
    K.HVASimulator whose H_sparse is  H_train = H_NN + lam * H_NNN  (weights
    trick, no module changes). CAUTION: its `.edges` contains the NNN pairs —
    gates must always come from `fragment_schedule`, never from the sim's own
    per-edge layering.
    """
    nn, nnn, dm = _frag_geometry()
    return K.HVASimulator(len(dc.SUB2_SITES), nn + nnn, dm,
                          weights=[1.0] * len(nn) + [float(lam)] * len(nnn))


def fragment_schedule(reps):
    """
    Gate list [(i, j, pidx)] for `reps` HVA layers on fragment 2's NN bonds
    ONLY (local labels), one theta per bond per layer — the Task-1 full-HVA
    layout restricted to the fragment. n_params = 10 * reps.
    """
    nn, _, _ = _frag_geometry()
    sched = [(i, j, r * len(nn) + k)
             for r in range(reps) for k, (i, j) in enumerate(nn)]
    return sched, reps * len(nn)


_SIM2_PLAIN = None          # lazy: plain (J=1, NN-only) fragment simulator


def _sim2():
    global _SIM2_PLAIN
    if _SIM2_PLAIN is None:
        _, sim2 = dc.make_subregion_sims()
        _SIM2_PLAIN = sim2
    return _SIM2_PLAIN


def dressed_fragment_state(x, reps):
    """8-qubit statevector of the dressed fragment: dimer cover + `reps` HVA
    layers on the NN bonds with angles x. reps=0 returns the bare cover."""
    if reps == 0 or np.size(x) == 0:
        return _sim2().psi0.reshape(-1).copy()
    sched, _ = fragment_schedule(reps)
    return K.statevector_schedule(_sim2(), np.asarray(x, dtype=float), sched)


def train_fragment(lam, reps, seeds=(0, 1, 2, 3, 4), maxiter=300, n_random=2,
                   init_scale=0.5, targets=None, verbose=True):
    """
    Multi-seed pre-training of fragment 2 against H_train = H_NN + lam*H_NNN
    (L-BFGS-B + the Task-1 adjoint gradient, gates on NN bonds only). Each
    seed takes the best of one theta=0 start (at lam=0 the dimer cover is the
    exact H_train ground state, so the start is stationary and free) plus
    `n_random` random restarts.

    Reported per seed:
      e_train        <H_train> (the optimized objective)
      err_train_pct  vs the exact H_train ground energy (ED of 2^8)
      e_nn           <H_NN> on the ORIGINAL uncalibrated fragment H — the
                     final-metric energy; de_local = e_nn - E2_exact is the
                     local energy PAID for the dressing
      s2             <S^2> (must stay 0: U_H preserves the singlet sector)
      fid_train      subspace fidelity to the exact H_train ground manifold
      rms_nn/rms_nnn RMS of the state's <S_i.S_j> vs `targets` (the ED-19
                     profile from ed_fragment_targets), NaN if targets=None
    """
    nn, nnn, _ = _frag_geometry()
    n = len(dc.SUB2_SITES)
    sim_tr = training_sim(lam)
    sched, npar = fragment_schedule(reps)
    h_nn = K.heisenberg_hamiltonian(n, nn).to_matrix(sparse=True).tocsr()
    e2_exact, _, _ = dc.exact_ground_subspace(n, nn)
    # ferromagnetic-side grounds can be high-spin multiplets (S=1 -> deg 3,
    # S=2 -> deg 5 was measured on the lambda scan), so probe deep enough
    for k_probe in (6, 10, 16):
        try:
            e_tr_exact, V_tr, _ = exact_ground_subspace_weighted(
                n, nn + nnn, [1.0] * len(nn) + [float(lam)] * len(nnn),
                k_probe=k_probe)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"H_train ground degeneracy > 15 at lam={lam}")
    obj = lambda z: K.energy_and_grad_schedule(sim_tr, z, sched, npar)
    runs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        t0 = time.perf_counter()
        inits = [np.zeros(npar)] + [rng.uniform(-init_scale, init_scale, npar)
                                    for _ in range(n_random)]
        fbest, xbest = np.inf, None
        nit = 0
        for x0 in inits:
            res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            nit += int(res.nit)
            if res.fun < fbest:
                fbest, xbest = float(res.fun), res.x
        sv = dressed_fragment_state(xbest, reps)
        e_nn = float(np.real(np.vdot(sv, h_nn @ sv)))
        rms_nn, rms_nnn, rms_all = _rms_triple(fragment_profile(sv), targets)
        rec = dict(seed=int(s), lam=float(lam), reps=int(reps), n_params=int(npar),
                   e_train=fbest,
                   err_train_pct=abs(fbest - e_tr_exact) / abs(e_tr_exact) * 100.0,
                   e_nn=e_nn, de_local=e_nn - e2_exact,
                   s2=dc.s2_matrix_free(n, sv),
                   fid_train=dc.subspace_fidelity(sv, V_tr),
                   rms_nn=rms_nn, rms_nnn=rms_nnn, rms_all=rms_all,
                   nit=nit, time_s=time.perf_counter() - t0,
                   x=np.asarray(xbest))
        runs.append(rec)
        if verbose:
            print(f"  seed {s}: E_train={fbest:9.4f} "
                  f"(err {rec['err_train_pct']:6.3f}%)  <H_NN>={e_nn:8.4f} "
                  f"(paid {rec['de_local']:+.4f})  fid={rec['fid_train']:.4f}  "
                  f"rms_nnn={rec['rms_nnn']:.4f}  [{nit} its, "
                  f"{rec['time_s']:4.1f}s]")
    return runs


# ==========================================================================
# 4. Correlation profiling — the measurable meaning of "spin-liquid-like"
# ==========================================================================
def ed_fragment_targets(psi19):
    """
    The TARGET profile: <S_i.S_j> of the 19-site ED ground state restricted to
    fragment 2 — NN bonds (SUB2_EDGES), the derived NNN pairs, and the 6
    interface bonds for context. `psi19` should be one S_z component of the
    doublet (dc.project_sz); S_i.S_j is an SU(2) scalar, so the profile is the
    same for either component. Returns dict(nn=…, nnn=…, interface=…), each a
    {global_pair: value} dict.
    """
    nn = [tuple(e) for e in dc.SUB2_EDGES]
    nnn = fragment2_nnn()[0]
    ifc = [tuple(e) for e in dc.INTERFACE_EDGES]
    corr = K.pair_correlations(19, psi19, nn + nnn + ifc)
    return dict(nn={p: corr[p] for p in nn},
                nnn={p: corr[p] for p in nnn},
                interface={p: corr[p] for p in ifc})


def fragment_profile(sv8):
    """
    <S_i.S_j> of an 8-qubit fragment state on its NN bonds and NNN pairs,
    keyed by GLOBAL pair labels (local +11) so it is directly comparable with
    ed_fragment_targets output.
    """
    nn_g = [tuple(e) for e in dc.SUB2_EDGES]
    nnn_g, nnn_l = fragment2_nnn()
    nn_l = dc.relabel(dc.SUB2_EDGES, dc.SUB2_SITES)
    corr = K.pair_correlations(len(dc.SUB2_SITES), sv8, nn_l + nnn_l)
    out = {g: corr[l] for g, l in zip(nn_g, nn_l)}
    out.update({g: corr[l] for g, l in zip(nnn_g, nnn_l)})
    return out


def profile_rms(profile, target):
    """RMS difference over the pairs present in BOTH dicts."""
    keys = [k for k in target if k in profile]
    d = np.array([profile[k] - target[k] for k in keys])
    return float(np.sqrt(np.mean(d ** 2)))


def _rms_triple(prof, targets):
    """(rms_nn, rms_nnn, rms_all) of a fragment profile vs ed_fragment_targets
    output; (nan, nan, nan) if targets is None. rms_all (NN+NNN combined) is
    the P3 predictor — the single number for 'how spin-liquid-like'."""
    if targets is None:
        return np.nan, np.nan, np.nan
    return (profile_rms(prof, targets["nn"]), profile_rms(prof, targets["nnn"]),
            profile_rms(prof, {**targets["nn"], **targets["nnn"]}))


# ==========================================================================
# 5. The locally exact subregion 1 (Task-2 Sec. 9.1 pipeline, reused)
# ==========================================================================
# Spinon on site 0 (the free end): reps=1 reaches the exact local doublet
# (Task-2 Sec. 3.1). Kept as a helper so every Task-3 configuration shares the
# same subregion-1 state and the fragment-2 treatment is the ONLY variable.
ALT_DIMERS_SUB1 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


def exact_local_sub1(results_dir=_TASK2_RESULTS):
    """
    (sv1, x1, sim1_alt): the frozen subregion-1 state of the exact-local
    pipeline, rebuilt from 2_Task/results/spinon_control.npz (best seed,
    reps=1, 14 params). Asserts its energy is the exact E1 = -16 (this state
    IS the local ground doublet, so any Task-3 gain must come from the
    fragment-2 dressing and the junctions — controlled experiment).
    """
    sw = dc.load_sweeps(Path(results_dir) / "spinon_control.npz")
    x1, _ = dc.best_of_sweeps(sw, 1)
    sim1_alt = K.HVASimulator(len(dc.SUB1_SITES),
                              dc.relabel(dc.SUB1_EDGES, dc.SUB1_SITES),
                              ALT_DIMERS_SUB1)
    sv1 = sim1_alt.statevector(np.asarray(x1), 1)
    e1 = float(np.real(np.vdot(sv1, sim1_alt.H_sparse @ sv1)))
    assert abs(e1 - (-16.0)) < 1e-6, f"sub1 rebuild broke: E1 = {e1}"
    return sv1, np.asarray(x1), sim1_alt


# ==========================================================================
# 6. Dressed recombination — mixed schedules on RecombinationVQE
# ==========================================================================
# RecombinationVQE.energy_and_grad / .statevector already take arbitrary
# [(i, j, pidx)] schedules (pre-flight verified: adjoint == finite differences
# on a mixed fragment+junction schedule). Only the multi-seed driver was
# hard-wired to interface-only schedules — that wrapper lives here.

def mixed_schedule(frag_edges, interface_edges, reps_frag, reps_ifc,
                   per_bond=True):
    """
    Generic joint schedule: `reps_frag` HVA layers on `frag_edges` (global
    labels) followed by `reps_ifc` junction layers on `interface_edges`.
    Fragment parameters come first. Returns (sched, n_params, n_frag_params).
    Works for any lattice (the 26-site chain uses it with its own edge sets).
    """
    frag = [tuple(e) for e in frag_edges]
    ifc = [tuple(e) for e in interface_edges]
    sched = [(i, j, r * len(frag) + k)
             for r in range(reps_frag) for k, (i, j) in enumerate(frag)]
    nf = reps_frag * len(frag)
    m = len(ifc)
    for r in range(reps_ifc):
        for k, (i, j) in enumerate(ifc):
            sched.append((i, j, nf + (r * m + k if per_bond else r)))
    return sched, nf + (reps_ifc * m if per_bond else reps_ifc), nf


def dressed_schedule(reps_frag, reps_ifc, per_bond=True):
    """Config-B schedule on the 19-site partition: fragment 2's NN bonds
    (qubits 11-18) + the 6 interface bonds. See mixed_schedule."""
    return mixed_schedule(dc.SUB2_EDGES, dc.INTERFACE_EDGES,
                          reps_frag, reps_ifc, per_bond)


def grad_at_zero(rec, sched, npar, bonds=None):
    """(E, ‖∇E‖) at theta=0 for an arbitrary schedule (P1 probe)."""
    E, g = rec.energy_and_grad(np.zeros(npar), sched, npar, bonds=bonds)
    return E, float(np.linalg.norm(g))


def optimize_schedule(rec, sched, npar, e_ref, seeds=(0, 1, 2, 3, 4),
                      use_sel=False, n_starts=1, zero_start=True,
                      init_scale=0.5, maxiter=200, V_ref=None,
                      compute_s2=True, verbose=True, reps_tag=0):
    """
    dc.RecombinationVQE.optimize generalized to an ARBITRARY schedule (e.g.
    the config-B fragment+junction mix). Same contract: per-seed L-BFGS-B +
    adjoint gradient, best of `n_starts` random inits (+ theta=0 run and raw
    theta=0 fallback if zero_start), objective <H_SEL> if use_sel else
    <H_full>, reported energy/err_pct ALWAYS <H_full>. Emits dicts with the
    dc._RUN_META keys (reps stores `reps_tag`), so dc.save_runs/load_runs work
    unchanged.
    """
    bonds_opt = rec.bonds_sel if use_sel else None
    obj = lambda z: rec.energy_and_grad(z, sched, npar, bonds=bonds_opt)
    runs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        t0 = time.perf_counter()
        inits = [rng.uniform(-init_scale, init_scale, npar)
                 for _ in range(n_starts)]
        if zero_start:
            inits.append(np.zeros(npar))
        fbest, xbest = obj(np.zeros(npar))[0], np.zeros(npar)
        nit = nfev = 0
        for x0 in inits:
            res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            nit += int(res.nit); nfev += int(res.nfev)
            if res.fun < fbest:
                fbest, xbest = float(res.fun), res.x
        dt = time.perf_counter() - t0
        sv = rec.statevector(xbest, sched)
        E = rec.energy(sv)
        rec_d = dict(seed=int(s), reps=int(reps_tag), n_params=int(npar),
                     per_bond=True, use_sel=bool(use_sel),
                     obj=float(fbest), energy=E,
                     err_pct=abs(E - e_ref) / abs(e_ref) * 100.0,
                     s2=(rec.s2(sv) if compute_s2 else np.nan),
                     fidelity=(dc.subspace_fidelity(sv, V_ref)
                               if V_ref is not None else np.nan),
                     nit=nit, nfev=nfev, time_s=dt, x=np.asarray(xbest))
        runs.append(rec_d)
        if verbose:
            print(f"  seed {s}: E={E:9.4f}  err={rec_d['err_pct']:6.3f}%  "
                  f"<S^2>={rec_d['s2']:.4f}  fid={rec_d['fidelity']:.4f}  "
                  f"[{npar}p, {nit} its, {dt:5.1f}s]")
    return runs


def refine_schedule(rec, sched, npar, runs, e_ref, maxiter=200, V_ref=None,
                    compute_s2=True, verbose=True):
    """
    dc.RecombinationVQE.refine for EXTERNAL schedules: prolongs saved
    optimize_schedule runs that hit their iteration cap, restarting L-BFGS-B
    from each stored optimum x with a fresh budget. Same objective the run
    used (H_SEL if use_sel). Must be called on the same frozen state.
    """
    out = []
    for r in runs:
        bonds_opt = rec.bonds_sel if r["use_sel"] else None
        obj = lambda z: rec.energy_and_grad(z, sched, npar, bonds=bonds_opt)
        t0 = time.perf_counter()
        res = minimize(obj, np.asarray(r["x"]), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        dt = time.perf_counter() - t0
        sv = rec.statevector(res.x, sched)
        E = rec.energy(sv)
        rec_d = dict(seed=r["seed"], reps=r["reps"], n_params=int(npar),
                     per_bond=True, use_sel=r["use_sel"], obj=float(res.fun),
                     energy=E, err_pct=abs(E - e_ref) / abs(e_ref) * 100.0,
                     s2=(rec.s2(sv) if compute_s2 else np.nan),
                     fidelity=(dc.subspace_fidelity(sv, V_ref)
                               if V_ref is not None else np.nan),
                     nit=int(res.nit), nfev=int(res.nfev), time_s=dt,
                     x=np.asarray(res.x))
        out.append(rec_d)
        if verbose:
            print(f"  seed {r['seed']}: err {r['err_pct']:6.3f}% -> "
                  f"{rec_d['err_pct']:6.3f}%  (+{rec_d['nit']} its, {dt:5.1f}s)")
    return out


# ==========================================================================
# 7. Dressed-frozen pipelines: config C (lambda) and config D (profile fit)
# ==========================================================================
def fit_fragment_profile(reps, targets, seeds=(0, 1, 2, 3, 4), maxiter=800,
                         n_random=3, init_scale=0.5, verbose=True):
    """
    Config-D trainer: dress fragment 2 by DIRECT least-squares fit of its
    <S_i.S_j> profile (NN + NNN pairs) to the ED-19 restricted target — the
    literal reading of "convert the dimers into the spin-liquid-like state",
    with the Sec.-1 target as the definition of that state. No Hamiltonian is
    modified at all here; the pre-analysis motivates it: no fragment-local
    H_NN + lam*H_NNN has the target profile as its ground state, but the
    fragment HVA CAN represent it (rms_all 0.23 -> ~0.03 at reps=2).
    Numerical-gradient L-BFGS (2^8 statevectors: seconds per seed).
    Returns train_fragment-shaped records with e_train := the fit cost
    (sum of squared deviations) and fid_train/err_train_pct := NaN.
    """
    n = len(dc.SUB2_SITES)
    nn_l, nnn_l, _ = _frag_geometry()
    pairs_l = nn_l + nnn_l
    nn_g = [tuple(e) for e in dc.SUB2_EDGES]
    pairs_g = nn_g + fragment2_nnn()[0]
    t_vec = np.array([{**targets["nn"], **targets["nnn"]}[p] for p in pairs_g])
    sched, npar = fragment_schedule(reps)
    sim2 = _sim2()
    e2_exact, _, _ = dc.exact_ground_subspace(n, nn_l)

    def cost(x):
        sv = K.statevector_schedule(sim2, x, sched)
        c = K.pair_correlations(n, sv, pairs_l)
        r = np.array([c[p] for p in pairs_l]) - t_vec
        return float(r @ r)

    runs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        t0 = time.perf_counter()
        inits = [np.zeros(npar)] + [rng.uniform(-init_scale, init_scale, npar)
                                    for _ in range(n_random)]
        fbest, xbest, nit = np.inf, None, 0
        for x0 in inits:
            res = minimize(cost, x0, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            nit += int(res.nit)
            if res.fun < fbest:
                fbest, xbest = float(res.fun), res.x
        sv = dressed_fragment_state(xbest, reps)
        e_nn = float(np.real(np.vdot(sv, sim2.H_sparse @ sv)))
        rms_nn, rms_nnn, rms_all = _rms_triple(fragment_profile(sv), targets)
        rec = dict(seed=int(s), lam=np.nan, reps=int(reps), n_params=int(npar),
                   e_train=fbest, err_train_pct=np.nan,
                   e_nn=e_nn, de_local=e_nn - e2_exact,
                   s2=dc.s2_matrix_free(n, sv), fid_train=np.nan,
                   rms_nn=rms_nn, rms_nnn=rms_nnn, rms_all=rms_all,
                   nit=nit, time_s=time.perf_counter() - t0,
                   x=np.asarray(xbest))
        runs.append(rec)
        if verbose:
            print(f"  seed {s}: fit cost={fbest:.5f}  rms_all={rms_all:.4f}  "
                  f"<H_NN>={e_nn:8.4f} (paid {rec['de_local']:+.4f})  "
                  f"[{nit} its, {rec['time_s']:4.1f}s]")
    return runs


def _recombine_and_junction(best_f, sv1, edges_full, e_ref, V_ref, reps_ifc,
                            seeds_j, maxiter_j, use_sel, verbose):
    """Shared tail of configs C and D: embed the dressed fragment, measure the
    frozen start (energy + P1 gradient at theta=0), run the junction VQE."""
    sv2 = dressed_fragment_state(best_f["x"], best_f["reps"])
    psiF = dc.embed_product(sv1, sv2)
    rec = dc.RecombinationVQE(psiF, edges_full=edges_full)
    e_frozen = rec.energy(psiF)
    _, g0 = rec.gradient_at_zero(reps=1)
    if verbose:
        print(f"  fragment: <H_NN> = {best_f['e_nn']:8.4f} "
              f"(paid {best_f['de_local']:+.4f})   rms_all = "
              f"{best_f['rms_all']:.4f}   frozen 19q: E = {e_frozen:.4f} "
              f"({abs(e_frozen - e_ref)/abs(e_ref)*100:.3f}%)   "
              f"|grad|@0 = {g0:.3e}")
    runs_j = rec.optimize(reps_ifc, e_ref=e_ref, seeds=seeds_j,
                          maxiter=maxiter_j, use_sel=use_sel, V_ref=V_ref,
                          verbose=verbose)
    best_j = min(runs_j, key=lambda r: r["err_pct"])
    return dict(reps_frag=int(best_f["reps"]),
                n_frag_params=int(best_f["n_params"]),
                e_train=best_f["e_train"],
                err_train_pct=best_f["err_train_pct"],
                e_nn=best_f["e_nn"], de_local=best_f["de_local"],
                s2_frag=best_f["s2"], fid_frag=best_f["fid_train"],
                rms_nn=best_f["rms_nn"], rms_nnn=best_f["rms_nnn"],
                rms_all=best_f["rms_all"],
                e_frozen=float(e_frozen),
                err_frozen=abs(e_frozen - e_ref) / abs(e_ref) * 100.0,
                grad0=float(g0),
                e_best=best_j["energy"], err_best=best_j["err_pct"],
                fid_best=best_j["fidelity"],
                gain_ifc=best_j["energy"] - float(e_frozen),
                x_frag=np.asarray(best_f["x"]), runs_j=runs_j)


def config_c(lam, sv1, edges_full, e_ref, V_ref=None, targets=None,
             reps_frag=2, seeds_frag=(0, 1, 2, 3, 4), maxiter_frag=300,
             reps_ifc=4, seeds_j=(0, 1, 2), maxiter_j=200, use_sel=False,
             verbose=True):
    """
    One point of the Task-3b lambda sweep, end to end:
      1. pre-train fragment 2 against H_train = H_NN + lam*H_NNN (multi-seed,
         8 qubits, seconds); freeze the best seed;
      2. embed |sub1_exact> (x) |fragment2_dressed> into 19 qubits;
      3. junction VQE over the interface gates ONLY — the metric is the
         ORIGINAL uncalibrated H throughout ("not J->J'").
    Returns a flat record dict (fragment metrics + frozen-start diagnostics
    incl. the P1 gradient at theta=0 + junction runs incl. x vectors).
    """
    if verbose:
        print(f"-- config C @ lambda = {lam:.3f} "
              f"(fragment reps={reps_frag}, junction reps={reps_ifc}) --")
    fr = train_fragment(lam, reps_frag, seeds=seeds_frag, maxiter=maxiter_frag,
                        targets=targets, verbose=False)
    best_f = min(fr, key=lambda r: r["e_train"])
    out = _recombine_and_junction(best_f, sv1, edges_full, e_ref, V_ref,
                                  reps_ifc, seeds_j, maxiter_j, use_sel,
                                  verbose)
    out["lam"] = float(lam)
    return out


def config_d(sv1, edges_full, e_ref, targets, V_ref=None, reps_frag=2,
             seeds_frag=(0, 1, 2, 3, 4), maxiter_frag=800, reps_ifc=4,
             seeds_j=(0, 1, 2, 3, 4), maxiter_j=400, use_sel=False,
             verbose=True):
    """
    Config D: profile-matched dressing. Same pipeline as config_c but the
    fragment is dressed by fit_fragment_profile (correlation fit to the ED
    target) instead of energy minimization of a modified Hamiltonian. In the
    saved records lam = NaN marks a D record; e_train holds the fit cost.
    """
    if verbose:
        print(f"-- config D (profile fit, fragment reps={reps_frag}, "
              f"junction reps={reps_ifc}) --")
    fr = fit_fragment_profile(reps_frag, targets, seeds=seeds_frag,
                              maxiter=maxiter_frag, verbose=False)
    best_f = min(fr, key=lambda r: r["e_train"])
    out = _recombine_and_junction(best_f, sv1, edges_full, e_ref, V_ref,
                                  reps_ifc, seeds_j, maxiter_j, use_sel,
                                  verbose)
    out["lam"] = np.nan
    return out


# ---- persistence for config-C/D records (lambda sweep + profile dressing) --
_C_META = ["lam", "reps_frag", "n_frag_params", "e_train", "err_train_pct",
           "e_nn", "de_local", "s2_frag", "fid_frag", "rms_nn", "rms_nnn",
           "rms_all", "e_frozen", "err_frozen", "grad0", "e_best", "err_best",
           "fid_best", "gain_ifc"]


def save_c_records(records, path):
    """One .npz for a list of config_c records (incl. per-seed junction runs
    and all x vectors). Incremental-save friendly: overwrite after each lambda."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_records": np.array(len(records))}
    for i, r in enumerate(records):
        payload[f"c_{i}"] = np.array([float(r[k]) for k in _C_META])
        payload[f"cx_{i}"] = np.asarray(r["x_frag"], dtype=float)
        payload[f"jm_{i}"] = np.array([[float(rr[k]) for k in dc._RUN_META]
                                       for rr in r["runs_j"]])
        for k, rr in enumerate(r["runs_j"]):
            payload[f"jx_{i}_{k}"] = np.asarray(rr["x"], dtype=float)
    np.savez(path, **payload)


def load_c_records(path):
    """Inverse of save_c_records."""
    d = np.load(path)
    out = []
    for i in range(int(d["n_records"])):
        r = dict(zip(_C_META, d[f"c_{i}"]))
        for k in ("reps_frag", "n_frag_params"):
            r[k] = int(r[k])
        r["x_frag"] = d[f"cx_{i}"]
        runs = []
        for k, row in enumerate(d[f"jm_{i}"]):
            rr = {key: (bool(v) if key in ("per_bond", "use_sel") else
                        (int(v) if key in ("seed", "reps", "n_params", "nit",
                                           "nfev") else float(v)))
                  for key, v in zip(dc._RUN_META, row)}
            rr["x"] = d[f"jx_{i}_{k}"]
            runs.append(rr)
        r["runs_j"] = runs
        out.append(r)
    return out


# ==========================================================================
# 7b. Interface-capacity schedules (Task-3 Sec. 6: the measured lever)
# ==========================================================================
# The Sec. 1-5 diagnosis locates the DnC error budget at the interface: the
# 24-parameter cut-bond junctions rebuild ~24% of the cut entanglement and run
# at a better marginal rate than the ED (1.63 vs 1.26) but move little energy.
# These schedules widen/deepen the junction stage instead of preprocessing the
# fragment — the same spreading mechanism, applied where the books say it pays.

def wide_junction_schedule(reps, per_bond=True, edges_full=None,
                           interface_edges=None):
    """
    'Plaquette' junction layers: one layer applies a Heisenberg gate on EVERY
    bond of the H_SEL support (the 6 cut bonds first, then the intra bonds
    adjacent to the junction sites — 21 bonds at the 19-site partition). This
    hands the junction stage the same intra-near-cut bonds the ED uses to pay
    its trade, while subregion cores stay frozen. n_params = 21*reps
    (per_bond) or reps. Returns (sched, n_params, bonds).
    """
    if interface_edges is None:
        interface_edges = dc.INTERFACE_EDGES
    sel = dc.build_h_sel(edges_full, interface_edges)
    ifc = [tuple(sorted(e)) for e in interface_edges]
    bonds = ifc + [e for e in sel if e not in set(ifc)]
    m = len(bonds)
    sched = [(i, j, r * m + k if per_bond else r)
             for r in range(reps) for k, (i, j) in enumerate(bonds)]
    return sched, (reps * m if per_bond else reps), bonds


def full_release_schedule(reps_frag, reps_ifc, per_bond=True):
    """
    Staged-release endgame: ONE schedule over the bare covers
    |dimers_sub1(spinon@0)> (x) |dimers_sub2> that re-applies (1) subregion 1's
    reps=1 local layer (14 gates, its Task-2 exact-local structure), (2)
    `reps_frag` fragment-2 dressing layers, (3) `reps_ifc` junction layers —
    with ALL parameters free. Warm-started from the frozen Task-3 optima it
    reproduces the config-A state exactly at x = [x1_alt, 0…, x_junctions]
    (asserted in the Sec.-6 cells), so any decrease is a true improvement of
    the same circuit family. Returns (sched, n_params, slices) with
    slices = dict(sub1=…, frag=…, ifc=…) index ranges into x.
    """
    s1 = [tuple(e) for e in dc.SUB1_EDGES]
    frag = [tuple(e) for e in dc.SUB2_EDGES]
    ifc = [tuple(e) for e in dc.INTERFACE_EDGES]
    sched = [(i, j, k) for k, (i, j) in enumerate(s1)]
    n1 = len(s1)
    for r in range(reps_frag):
        for k, (i, j) in enumerate(frag):
            sched.append((i, j, n1 + r * len(frag) + k))
    nf = n1 + reps_frag * len(frag)
    m = len(ifc)
    for r in range(reps_ifc):
        for k, (i, j) in enumerate(ifc):
            sched.append((i, j, nf + (r * m + k if per_bond else r)))
    npar = nf + (reps_ifc * m if per_bond else reps_ifc)
    slices = dict(sub1=slice(0, n1), frag=slice(n1, nf), ifc=slice(nf, npar))
    return sched, npar, slices


# ==========================================================================
# 8. The alternative SU(2) sectorization (26 sites): a dimer ON the cut
# ==========================================================================
# When BOTH fragments are odd (doublet x doublet, e.g. the 26-site chain =
# 19 + 7), Task 2 reached the S=0 sector with the Clebsch-Gordan combination.
# The "other way to preserve SU(2)" the advisor hints at: a GLOBAL dimer cover
# in which one singlet sits ON an interface bond. The even total lattice then
# admits a perfect matching that uses exactly one cut bond, so the product of
# singlets is exact S=0 — no CG, no entangled two-register preparation. Parity
# note: this does NOT exist for odd+even splits like the 19-site partition
# (removing one endpoint from the even fragment leaves it odd), which is why
# Task 2 could not use it at 19 sites.

def interface_dimer_cover(part):
    """
    A perfect matching of the full lattice that uses EXACTLY ONE interface
    bond: for each cut bond (u, v), seat a dimer there and look for perfect
    matchings of the two fragments with u and v removed (networkx blossom
    algorithm, intra-fragment bonds only). `part` is a dc.star_chain_partition
    dict. Returns the sorted cover [(u,v)] + matching1 + matching2; raises if
    no cut bond admits one.
    """
    import networkx as nx
    sub1 = set(range(part["n_sub1"]))
    G1 = nx.Graph([tuple(e) for e in part["sub1_edges"]])
    G2 = nx.Graph([tuple(e) for e in part["sub2_edges"]])
    G1.add_nodes_from(sub1)
    G2.add_nodes_from(part["sub2_sites"])
    for (u, v) in part["interface_edges"]:
        a, b = (u, v) if u in sub1 else (v, u)
        H1, H2 = G1.copy(), G2.copy()
        H1.remove_node(a); H2.remove_node(b)
        m1 = nx.max_weight_matching(H1, maxcardinality=True)
        m2 = nx.max_weight_matching(H2, maxcardinality=True)
        if 2 * len(m1) == H1.number_of_nodes() and 2 * len(m2) == H2.number_of_nodes():
            cover = ([tuple(sorted((a, b)))]
                     + [tuple(sorted(e)) for e in m1]
                     + [tuple(sorted(e)) for e in m2])
            return sorted(cover)
    raise ValueError("no interface bond admits a global perfect matching")


# Singlet preparation as ONE 4x4 unitary in the blocked-kernel convention
# (index = 2*bit_a + bit_b): column |00> -> (|01> - |10>)/sqrt(2), i.e. the
# same state K.HVASimulator._dimer_state prepares with h/x/cx/z (verified in
# the module smoke test). Other columns: any orthonormal completion.
_SINGLET_PREP = np.array([[0, 1, 0, 0],
                          [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                          [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                          [0, 0, 0, 1]], dtype=complex)


def dimer_cover_state(n, cover):
    """
    Product of singlets over `cover` (global labels) as an n-qubit
    statevector, built IN PLACE with the Task-2 blocked kernels — safe at
    n=26 (one 1 GB vector, no full-size temporaries). Unpaired sites stay |0>.
    """
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    for (a, b) in cover:
        dc.apply2_inplace(n, psi, _SINGLET_PREP, a, b)
    return psi.reshape(-1)


# ==========================================================================
# 9. Drawing: the dressed recombined circuit (Fig.-4 style, compact)
# ==========================================================================
def build_dressed_drawing(x_frag, reps_frag, theta_ifc, reps_ifc,
                          per_bond=True, label1="Sub-1 frozen (exact, Step 1)"):
    """
    Compact 19-qubit drawing of the Task-3 pipeline: subregion 1 collapsed
    into one labeled block (paper's 'shaded' part), fragment 2's dimer cover +
    dressing layers in GREEN detail (UH-R2 — the Task-3 novelty), then the
    junction gates in RED (UH-J). Draw with
    qc.draw('mpl', style=dc.MPL_STYLE, fold=-1).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    qc = QuantumCircuit(19)
    qc.append(Gate("sub1-frozen", 11, [], label=label1), list(range(11)))
    for a, b in dc.SUB2_DIMERS:
        qc.h(a); qc.x(b); qc.cx(a, b); qc.z(a)
    xf = np.atleast_1d(np.asarray(x_frag, dtype=float))
    k = 0
    for _ in range(reps_frag):
        for (i, j) in dc.SUB2_EDGES:
            qc.append(dc._uh_box("UH-R2", xf[k]), [i, j]); k += 1
    qc.barrier(label="freeze")
    th = np.atleast_1d(np.asarray(theta_ifc, dtype=float))
    m = len(dc.INTERFACE_EDGES)
    for r in range(reps_ifc):
        for k, (i, j) in enumerate(dc.INTERFACE_EDGES):
            qc.append(dc._uh_box("UH-J", th[r * m + k if per_bond else r]), [i, j])
    return qc


# ==========================================================================
# 10. Smoke test (fast, < 1 min): every new code path against ground truth
# ==========================================================================
def smoke(verbose=True):
    """Asserts: NNN geometry, weighted-H identity, NN-only-schedule adjoint
    gradient vs finite differences, S^2 preservation of the dressing, mixed-
    schedule gradient on RecombinationVQE, dimer_cover_state vs K's singlet
    prep, and save/load round trips. Raises on any failure."""
    rng = np.random.default_rng(7)
    say = print if verbose else (lambda *a, **k: None)

    g, l = fragment2_nnn()
    assert len(g) == 10 and all(b - 11 >= 0 for p in g for b in p)
    assert l == dc.relabel(g, dc.SUB2_SITES)
    say(f"[1/7] NNN geometry: {len(g)} pairs, labels consistent")

    lam = 0.37
    nn, nnn, _ = _frag_geometry()
    sim_tr = training_sim(lam)
    h = (K.heisenberg_hamiltonian(8, nn).to_matrix(sparse=True)
         + lam * K.heisenberg_hamiltonian(8, nnn).to_matrix(sparse=True))
    v = rng.normal(size=256) + 1j * rng.normal(size=256)
    assert np.abs(sim_tr.H_sparse @ v - h @ v).max() < 1e-12
    say("[2/7] weights trick: H_sparse == H_NN + lam*H_NNN")

    sched, npar = fragment_schedule(2)
    th = rng.uniform(-0.4, 0.4, npar)
    _, gr = K.energy_and_grad_schedule(sim_tr, th, sched, npar)
    eps, k = 1e-6, 7
    e_p = K.energy_and_grad_schedule(sim_tr, th + eps * np.eye(npar)[k], sched, npar)[0]
    e_m = K.energy_and_grad_schedule(sim_tr, th - eps * np.eye(npar)[k], sched, npar)[0]
    assert abs(gr[k] - (e_p - e_m) / (2 * eps)) < 1e-7
    sv = dressed_fragment_state(th, 2)
    assert abs(dc.s2_matrix_free(8, sv)) < 1e-12
    say("[3/7] NN-only schedule vs weighted H: adjoint == FD; <S^2> == 0")

    edges_19, _ = dc.load_task1_lattice()
    sim1, sim2 = dc.make_subregion_sims()
    psiF = dc.frozen_state(sim1, [], 0, sim2, [], 0)
    rec = dc.RecombinationVQE(psiF, edges_full=edges_19)
    ms, npm, nf = dressed_schedule(1, 1)
    assert nf == 10 and npm == 16
    thm = rng.uniform(-0.3, 0.3, npm)
    _, gm = rec.energy_and_grad(thm, ms, npm)
    ep = np.zeros(npm); ep[nf + 2] = eps        # a junction parameter
    fd = (rec.energy_and_grad(thm + ep, ms, npm)[0]
          - rec.energy_and_grad(thm - ep, ms, npm)[0]) / (2 * eps)
    assert abs(gm[nf + 2] - fd) < 1e-6
    say("[4/7] mixed fragment+junction schedule: adjoint == FD at 19q")

    ref = K.HVASimulator(2, [(0, 1)], [(0, 1)]).psi0.reshape(-1)
    assert np.abs(dimer_cover_state(2, [(0, 1)]) - ref).max() < 1e-12
    cov4 = dimer_cover_state(4, [(0, 1), (2, 3)])
    ref4 = K.HVASimulator(4, [(0, 1), (2, 3)], [(0, 1), (2, 3)]).psi0.reshape(-1)
    assert np.abs(cov4 - ref4).max() < 1e-12
    say("[5/7] dimer_cover_state == K's h/x/cx/z singlet prep")

    e_tr, V_tr, _ = exact_ground_subspace_weighted(
        8, nn + nnn, [1.0] * len(nn) + [lam] * len(nnn))
    fr = train_fragment(lam, 1, seeds=(0,), maxiter=60, verbose=False)
    assert fr[0]["e_train"] >= e_tr - 1e-9 and fr[0]["s2"] < 1e-9
    say(f"[6/7] train_fragment: variational bound holds "
        f"(E_train {fr[0]['e_train']:.4f} >= exact {e_tr:.4f}), S^2 == 0")

    import tempfile
    recc = dict(lam=lam, reps_frag=1, n_frag_params=10, e_train=fr[0]["e_train"],
                err_train_pct=fr[0]["err_train_pct"], e_nn=fr[0]["e_nn"],
                de_local=fr[0]["de_local"], s2_frag=fr[0]["s2"],
                fid_frag=fr[0]["fid_train"], rms_nn=0.1, rms_nnn=0.2,
                rms_all=0.15,
                e_frozen=-27.5, err_frozen=5.0, grad0=1e-3, e_best=-28.5,
                err_best=2.2, fid_best=0.2, gain_ifc=-1.0,
                x_frag=fr[0]["x"],
                runs_j=[dict(seed=0, reps=4, n_params=24, per_bond=True,
                             use_sel=False, obj=-28.5, energy=-28.5,
                             err_pct=2.2, s2=0.75, fidelity=0.2, nit=10,
                             nfev=12, time_s=1.0, x=np.zeros(24))])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.npz"
        save_c_records([recc], p)
        back = load_c_records(p)[0]
        assert abs(back["e_train"] - recc["e_train"]) < 1e-12
        assert np.allclose(back["x_frag"], recc["x_frag"])
        assert back["runs_j"][0]["n_params"] == 24
    say("[7/7] save/load round trip for config-C records")
    say("smoke: ALL PASS")


if __name__ == "__main__":
    smoke()
