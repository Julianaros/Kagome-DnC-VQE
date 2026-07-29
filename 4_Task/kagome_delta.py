"""
kagome_delta.py — Task-4 engine: dummy-triangle insertion in corner-sharing
(Delta-chain) KAFH clusters.

The Task-4 clusters are open chains of corner-sharing triangles — the *Delta
(sawtooth) chain* of the literature (Nakamura & Kubo, PRB 53, 6393 (1996);
Sen, Shastry, Walstedt & Cava, PRB 53, 6401 (1996)) — cut out of the kagome
lattice of Tasks 1-3:

  ORIGINAL  (5 sites):  two triangles sharing corner 0:      (0,1,2)-(0,22,24)
  MODIFIED  (7 sites):  a *dummy* triangle inserted at 0:    (0,1,2)-(0,13,0')-(0',22,24)
                        site 13 is new; 0' duplicates 0.

Everything here works in the project's uncalibrated convention
H = Σ_(i,j) (XX+YY+ZZ)  (singlet bond = -3, J=1), and in LOCAL labels

  original:  0,1,2,22,24        -> 0,1,2,3,4
  modified:  0,1,2,13,0',22,24  -> 0,1,2,3,4,5,6   (A=(0,1,2), B=(0,3,4), C=(4,5,6))

Contents
  1. Lattice builders + Task-4 constants (Ahsan's clusters, exchange symmetry σ)
  2. Exact ground manifolds (dense below `dense_max`, adaptive Lanczos above)
     and matrix-free per-triangle observables (reuses the Task-2 blocked kernels)
  3. Dimer-cover combinatorics: enumeration, mutual exclusion, cover states
     (via er.dimer_cover_state), spinon bookkeeping
  4. Symmetry operators: qubit permutations, parity splitting of a manifold
  5. The singlet-absorption operator identity (Lemma 1 of the write-up)
  6. The 6<->8 map machinery: dummy contraction R (singlet teleportation),
     its manifold restriction r = V_orig† R V_mod, and the local insertion
     nullspace test (does any J: C^2 -> C^8 on site 0 alone lift GS -> GS?)
  7. Delta-chain scan (open chains and rings) for the E0 = -3·n_tri law
  8. Persistence helpers + smoke test

Imports the Task-1/2/3 engines from their folders (never copied):
K = kagome_hva, dc = kagome_dc, er = kagome_er.
`python kagome_delta.py` runs the smoke test.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _t in ("1_Task", "2_Task", "3_Task"):
    _p = str(_HERE.parent / _t)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kagome_hva as K      # noqa: E402  Task-1 engine
import kagome_dc as dc      # noqa: E402  Task-2 engine (blocked kernels)
import kagome_er as er      # noqa: E402  Task-3 engine (dimer_cover_state)

# 2-qubit bond generator XX+YY+ZZ (same convention as K/dc)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.array([[1, 0], [0, -1]], complex)
_BOND = np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z)

# ==========================================================================
# 1. Lattices and Task-4 constants
# ==========================================================================
# Ahsan's clusters, LOCAL labels (see module docstring for the global map).
ORIG_N = 5
ORIG_TRIANGLES = [(0, 1, 2), (0, 3, 4)]
ORIG_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)]
ORIG_GLOBAL = {0: "0", 1: "1", 2: "2", 3: "22", 4: "24"}

MOD_N = 7
MOD_TRIANGLES = [(0, 1, 2), (0, 3, 4), (4, 5, 6)]      # A, B(dummy), C
MOD_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4),
             (4, 5), (5, 6), (4, 6)]
MOD_GLOBAL = {0: "0", 1: "1", 2: "2", 3: "13", 4: "0'", 5: "22", 6: "24"}
DUMMY_PAIR = (3, 4)                                    # (13, 0') — the two
# dummy-triangle sites that exist only in the modified cluster; contracting
# them against a singlet (Sec. 6) is the 8->6 direction of the map.

# Exchange (mirror) automorphisms:  σ_mod = (0<->0', 1<->22, 2<->24, 13 fixed)
SIGMA_MOD = {0: 4, 4: 0, 1: 5, 5: 1, 2: 6, 6: 2, 3: 3}
# and its original-cluster analogue (swap the two triangles through the hub)
SIGMA_ORIG = {0: 0, 1: 3, 3: 1, 2: 4, 4: 2}


def delta_chain(n_tri):
    """
    Open Delta chain of `n_tri` corner-sharing triangles, uniform labels:
    triangle k = (2k, 2k+1, 2k+2), sites 0..2·n_tri.  Returns (n, edges,
    triangles).  delta_chain(2)/delta_chain(3) are the original/modified
    Task-4 clusters up to relabeling (asserted in smoke()).
    """
    triangles = [(2 * k, 2 * k + 1, 2 * k + 2) for k in range(n_tri)]
    edges = [e for (a, b, c) in triangles for e in ((a, b), (b, c), (a, c))]
    return 2 * n_tri + 1, edges, triangles


def delta_ring(n_tri):
    """
    PERIODIC Delta chain (the last triangle closes onto site 0): triangle
    k = (2k, 2k+1, (2k+2) mod 2·n_tri), sites 0..2·n_tri−1.  Used in Sec. 9
    to show that saturation survives one loop as long as a valid dimer cover
    exists (it is coverability, not acyclicity, that decides).
    """
    m = 2 * n_tri
    triangles = [(2 * k, 2 * k + 1, (2 * k + 2) % m) for k in range(n_tri)]
    edges = [e for (a, b, c) in triangles for e in ((a, b), (b, c), (a, c))]
    return m, edges, triangles


def reflection_perm(n_tri):
    """Mirror automorphism i -> 2·n_tri − i of the open chain (triangle k
    -> triangle n_tri−1−k); the n_tri=3 instance is σ_mod up to relabeling."""
    m = 2 * n_tri
    return {i: m - i for i in range(m + 1)}


# ==========================================================================
# 2. Ground manifolds and per-triangle observables
# ==========================================================================
def ground_manifold(n, edges, deg_tol=1e-8):
    """
    (E0, V, w) with V = orthonormal basis of the FULL degenerate ground
    manifold and w = the whole spectrum, by dense eigh (exact degeneracies,
    no k_probe guesswork — this is what caught dc.project_sz's garbage-column
    gotcha in the prelims).  Intended for the Task-4 clusters (n <= ~13);
    for chain scans use ground_manifold_sz, which is dense-exact INSIDE one
    S^z sector and immune to the Lanczos degeneracy failure (eigsh at
    n_tri=7 returned deg 13 instead of 16 with cover weights 0.73 — a
    silently incomplete manifold).
    """
    H = K.heisenberg_hamiltonian(n, edges).to_matrix()
    w, v = np.linalg.eigh(H)
    deg = int(np.sum(np.abs(w - w[0]) < deg_tol * max(1.0, abs(w[0]))))
    return float(w[0]), v[:, :deg], w


def sz_sector_basis(n, sz):
    """Basis bitmasks (|1> = spin down) of the S^z = sz sector, ascending."""
    n_down = round((n - 2 * sz) / 2)
    if not 0 <= n_down <= n:
        raise ValueError(f"empty sector sz={sz} for n={n}")
    from itertools import combinations
    return sorted(sum(1 << q for q in pos)
                  for pos in combinations(range(n), n_down))


def sector_hamiltonian(n, edges, sz):
    """
    H restricted to the S^z = sz sector as a scipy CSR matrix.  Per bond
    (i,j): parallel bits -> +1 on the diagonal; antiparallel -> -1 on the
    diagonal and 2 to the spin-flipped state ((XX+YY)|01> = 2|10>).
    """
    from scipy.sparse import coo_matrix
    basis = sz_sector_basis(n, sz)
    index = {b: k for k, b in enumerate(basis)}
    rows, cols, vals = [], [], []
    for k, b in enumerate(basis):
        diag = 0.0
        for (i, j) in edges:
            bi, bj = (b >> i) & 1, (b >> j) & 1
            if bi == bj:
                diag += 1.0
            else:
                diag -= 1.0
                rows.append(index[b ^ (1 << i) ^ (1 << j)])
                cols.append(k)
                vals.append(2.0)
        rows.append(k); cols.append(k); vals.append(diag)
    dim = len(basis)
    return coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr(), basis


def ground_manifold_sz(n, edges, deg_tol=1e-8):
    """
    Exact ground data via ONE dense S^z-sector diagonalization (sector
    S^z = +1/2 for odd n, 0 for even n — the sector every ground multiplet
    must visit).  Returns (E0, Vfull, w_sector, deg_full):
      Vfull    — sector ground states embedded back into the 2^n register,
      deg_full — total degeneracy, inferred rigorously: each column is
                 checked to sit at the MINIMUM of S² (3/4 for odd n, 0 for
                 even), which by zero-variance-at-the-floor makes it an exact
                 S eigenstate; a doublet contributes exactly one state to
                 S^z=+1/2 (deg = 2m), a singlet one state to S^z=0 (deg = m).
    """
    sz = 0.5 * (n % 2)
    Hs, basis = sector_hamiltonian(n, edges, sz)
    w, v = np.linalg.eigh(Hs.toarray())
    m = int(np.sum(np.abs(w - w[0]) < deg_tol * max(1.0, abs(w[0]))))
    Vfull = np.zeros((2 ** n, m), dtype=complex)
    Vfull[np.array(basis), :] = v[:, :m]
    s2_min = 0.75 if n % 2 else 0.0
    for k in range(m):
        s2 = s2_matrix_free_col(n, Vfull[:, k])
        if abs(s2 - s2_min) > 1e-7:
            raise RuntimeError(f"sector column {k} has <S²>={s2}, not the "
                               f"minimum {s2_min}: deg inference invalid")
    deg_full = (2 * m) if n % 2 else m
    return float(w[0]), Vfull, w, deg_full


def s2_matrix_free_col(n, sv):
    """<S²> of one statevector, matrix-free (thin wrapper on dc kernels)."""
    return dc.s2_matrix_free(n, sv)


def bond_energy(n, sv, bond):
    """<XX+YY+ZZ> on one bond, matrix-free (Task-2 blocked kernels)."""
    psiT = np.asarray(sv).reshape((2,) * n)
    return float(np.real(dc.bond_vdot(n, psiT, psiT, _BOND, *bond)))


def triangle_energy(n, sv, tri):
    """E_△ = Σ_{bonds of △} <XX+YY+ZZ>  (min −3: the triangle inequality)."""
    a, b, c = tri
    return sum(bond_energy(n, sv, e) for e in ((a, b), (b, c), (a, c)))


def s2_triangle(n, sv, tri):
    """<S²_△> = 9/4 + E_△/2  (operator identity S²_△ = 9/4·I + H_△/2)."""
    return 2.25 + 0.5 * triangle_energy(n, sv, tri)


def triangle_hams(n, triangles):
    """Dense H_△ per triangle (small n only; used for manifold restrictions)."""
    out = {}
    for t in triangles:
        a, b, c = t
        out[t] = K.heisenberg_hamiltonian(n, [(a, b), (b, c), (a, c)]).to_matrix()
    return out


def restrict(V, M):
    """V† M V — an operator restricted to the manifold spanned by V's columns."""
    return V.conj().T @ M @ V


# ==========================================================================
# 3. Dimer-cover combinatorics
# ==========================================================================
def enumerate_covers(triangles):
    """
    All ways to place ONE dimer per triangle with all dimers pairwise
    disjoint ("valid covers").  Returns (valid, total) with total = 3^n_tri.
    """
    per_tri = [[(a, b), (b, c), (a, c)] for (a, b, c) in triangles]
    valid = []
    for combo in product(*per_tri):
        sites = [s for e in combo for s in e]
        if len(set(sites)) == 2 * len(combo):
            valid.append(tuple(combo))
    return valid, 3 ** len(triangles)


def mutually_exclusive(cover, hub_a=0, hub_c=4, tri_a=0, tri_c=2):
    """
    Ahsan's claim (1) predicate on one cover: NOT (hub 0 dimerized inside
    triangle A  AND  hub 0' dimerized inside triangle C).
    """
    return not (hub_a in cover[tri_a] and hub_c in cover[tri_c])


def cover_state(n, cover):
    """Product of singlets over the cover; unpaired sites |0> (spin up).
    Thin wrapper over er.dimer_cover_state (Task-3 blocked-kernel prep)."""
    return er.dimer_cover_state(n, list(cover))


def spinon_sites(n, cover):
    """Sites left unpaired by the cover (they carry the free S=1/2)."""
    used = {s for e in cover for s in e}
    return [s for s in range(n) if s not in used]


# ==========================================================================
# 4. Permutation symmetries
# ==========================================================================
def permutation_matrix(n, perm):
    """
    Dense operator P_σ with (P_σ ψ)(s_{σ(1)}...) = ψ(s...): qubit q of the
    output takes qubit perm[q] of the input.  perm is a dict site->site
    (an involution for all Task-4 symmetries; small n only).
    """
    P = np.zeros((2 ** n, 2 ** n))
    for b in range(2 ** n):
        bits = [(b >> q) & 1 for q in range(n)]
        nb = sum(bits[perm[q]] << q for q in range(n))
        P[nb, b] = 1.0
    return P


def parity_split(V, P, tol=1e-8):
    """
    Split a manifold V into σ-even/odd orthonormal bases (V_even, V_odd).
    Requires [P, H] = 0 so that P restricted to the manifold is unitary with
    eigenvalues ±1 (asserted).
    """
    M = restrict(V, P)
    assert np.allclose(M @ M.conj().T, np.eye(V.shape[1]), atol=1e-8), \
        "P does not preserve the manifold"
    w, u = np.linalg.eigh((M + M.conj().T) / 2)
    assert np.all(np.abs(np.abs(w) - 1.0) < tol), f"non-±1 parity: {w}"
    return V @ u[:, w > 0], V @ u[:, w < 0]


# ==========================================================================
# 5. Lemma 1 — singlet absorption:  S²_△ (Π_singlet ⊗ I) = 3/4 (Π_singlet ⊗ I)
# ==========================================================================
def singlet_absorption_residual(n_env=2):
    """
    Operator-identity check for Lemma 1 on triangle (0,1,2) ⊗ an n_env-site
    environment: if sites (0,1) hold a singlet, S_△ acts as S_2 alone, so
    S²_△ = 3/4 on that subspace NO MATTER how site 2 entangles with the
    environment.  Returns ‖(S²_△ − 3/4)(Π_s(0,1) ⊗ I)‖_max (should be 0).
    """
    n = 3 + n_env
    tri = K.heisenberg_hamiltonian(n, [(0, 1), (1, 2), (0, 2)]).to_matrix()
    S2t = 2.25 * np.eye(2 ** n) + 0.5 * tri
    s = np.zeros(4, complex)
    s[0b01], s[0b10] = 1 / np.sqrt(2), -1 / np.sqrt(2)   # bit0=site0, bit1=site1
    Pi2 = np.outer(s, s.conj())                          # singlet projector (0,1)
    Pi = np.kron(np.eye(2 ** (n - 2)), Pi2)              # sites 0,1 = low bits
    return float(np.abs((S2t - 0.75 * np.eye(2 ** n)) @ Pi).max())


def s_minus(n):
    """
    Total lowering operator S⁻ = Σ_i (X_i − iY_i)/2 as a dense matrix, in the
    project convention |0⟩ = spin up (Z|0⟩ = +|0⟩).  NOTE the sign of the Y
    term: with (X + iY)/2 one builds S⁺ instead, which ANNIHILATES the cover
    states (they are doublet tops) and silently fills a QR basis with junk —
    the same failure family as the dc.project_sz garbage-column gotcha.
    """
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    for q in range(n):
        lab = ["I"] * n
        lab[n - 1 - q] = "X"
        labx = "".join(lab)
        lab[n - 1 - q] = "Y"
        terms += [(labx, 0.5), ("".join(lab), -0.5j)]
    return SparsePauliOp.from_list(terms).to_matrix()


def triplet_kink_state(n, pair, site, singlets):
    """
    |s(a,b)⟩ ⊗ ... ⊗ |(t(pair) ⊗ σ(site))_{S=1/2, Sz=+1/2}⟩ : the "kink in a
    triplet" state — `pair` in a spin TRIPLET, Clebsch–Gordan-coupled with the
    spin at `site` to total S=1/2, times singlets on `singlets`.  Any site not
    mentioned stays |0⟩.  This is the closed form of the contraction kernel
    (Theorem 4): with pair=(13,0') the singlet bra ⟨s| annihilates it.
      |1/2,+1/2⟩ = √(2/3)|t₊₁⟩|↓⟩ − √(1/3)|t₀⟩|↑⟩,  |0⟩ = ↑.
    """
    a, b = pair
    psi = np.zeros(2 ** n, complex)
    s2 = 1 / np.sqrt(2)
    cg = [(np.sqrt(2 / 3), (0, 0), 1),          # |t+1>|down>
          (-np.sqrt(1 / 3) * s2, (0, 1), 0),    # |t0> |up>
          (-np.sqrt(1 / 3) * s2, (1, 0), 0)]
    sing = [((s2, (0, 1)), (-s2, (1, 0)))] * len(singlets)
    for coef0, (ba, bb), bs in cg:
        for choice in product(*sing):
            coef = coef0 * np.prod([c for c, _ in choice]) if choice else coef0
            idx = (ba << a) + (bb << b) + (bs << site)
            for (c_, (x, y)), (u, v) in zip(choice, singlets):
                idx += (x << u) + (y << v)
            psi[idx] += coef
    return psi / np.linalg.norm(psi)


# ==========================================================================
# 6. The 6<->8 map: contraction, manifold restriction, local-insertion test
# ==========================================================================
def contraction_map(pair=DUMMY_PAIR):
    """
    R: C^128 -> C^32 — contract the dummy pair (13,0') = local (3,4) of the
    modified cluster against the singlet bra ⟨s| = (⟨01|−⟨10|)/√2 (same sign
    convention as er._SINGLET_PREP / K's dimer prep) and relabel the survivors
    (0,1,2,5,6) -> (0,1,2,3,4).  Singlet teleportation makes R map modified
    dimer covers onto original covers up to ±1/2 factors (checked in smoke()).
    """
    a, b = pair
    keep = [s for s in range(MOD_N) if s not in pair]      # (0,1,2,5,6)
    R = np.zeros((2 ** ORIG_N, 2 ** MOD_N), complex)
    for m in range(2 ** MOD_N):
        bits = [(m >> q) & 1 for q in range(MOD_N)]
        if bits[a] == bits[b]:
            continue                                       # ⟨s| kills |00>,|11>
        amp = (1 if (bits[a], bits[b]) == (0, 1) else -1) / np.sqrt(2)
        o = sum(bits[s] << q for q, s in enumerate(keep))
        R[o, m] += amp
    return R


def manifold_map(V_orig, V_mod, R):
    """
    r = V_orig† R V_mod : the contraction in manifold coordinates (6×8).
    Returns (r, U, svals, Wh) with r = U·diag(svals)·Wh its SVD.  rank(r)
    = how many of the 8 modified dimensions survive contraction; ker(r)
    = the doublet with no original counterpart (Sec. 8 of the notebook).
    """
    r = V_orig.conj().T @ R @ V_mod
    U, svals, Wh = np.linalg.svd(r)
    return r, U, svals, Wh


def lift_operator(J):
    """
    The LOCAL insertion candidate as a 128×32 operator: site 0 of the
    original is fed to J: C² -> C⁸ (output bits ordered (t0, t13, t0'), i.e.
    j = t0 + 2·t13 + 4·t0'), sites (1,2,22,24) pass through to (1,2,5,6).
    """
    M = np.zeros((2 ** MOD_N, 2 ** ORIG_N), complex)
    pass_map = {1: 1, 2: 2, 3: 5, 4: 6}                    # orig -> mod sites
    out_sites = (0, 3, 4)                                  # J's output legs
    for b in range(2 ** ORIG_N):
        bits = [(b >> q) & 1 for q in range(ORIG_N)]
        base = sum(bits[o] << pass_map[o] for o in pass_map)
        for j in range(8):
            t = [(j >> k) & 1 for k in range(3)]
            m = base + sum(t[k] << out_sites[k] for k in range(3))
            M[m, b] += J[j, bits[0]]
    return M


def local_insertion_nullspace(V_orig, V_mod, tol=1e-10):
    """
    Does ANY local J: C² -> C⁸ acting on site 0 alone map the WHOLE original
    ground manifold into the modified one, i.e. (1−P_mod)·lift(J)·V_orig = 0?
    The constraint is linear in J's 16 entries: build the 16-column constraint
    matrix, SVD it, return (singular values, nullspace basis as 8×2 J's).
    A nonempty nullspace would make "0' is a copy of 0" a strictly local
    statement; an empty one proves the insertion is necessarily non-local.
    """
    Pperp = np.eye(2 ** MOD_N) - V_mod @ V_mod.conj().T
    cols = []
    for j in range(8):
        for s in range(2):
            E = np.zeros((8, 2)); E[j, s] = 1.0
            cols.append((Pperp @ (lift_operator(E) @ V_orig)).reshape(-1))
    C = np.stack(cols, axis=1)                             # (128·deg) × 16
    _, svals, Wh = np.linalg.svd(C, full_matrices=False)
    null = [Wh[k].conj().reshape(8, 2) for k in range(16) if svals[k] < tol]
    return svals, null


# ==========================================================================
# 7. Delta-chain scan
# ==========================================================================
def chain_scan(n_tris, ring=False, deg_tol=1e-8):
    """
    For each n_tri: exact E0, TOTAL ground degeneracy (via the S^z-sector
    solver, see ground_manifold_sz), the sector gap, number of valid covers,
    and the rank of the cover states — the data behind Theorem 5
    (E0 = −3·n_tri, deg = 2(n_tri+1), covers = 2·n_tri+1, cover-rank =
    n_tri+1 for open chains; deg = 2, covers = 2 for rings).  The cover
    weight is measured against the sector manifold, which is legitimate
    because every cover state lives in that same sector (S^z=+1/2 spinon-up
    for odd chains, S^z=0 for rings).  Returns a list of dicts.
    """
    out = []
    build = delta_ring if ring else delta_chain
    for n_tri in n_tris:
        n, edges, tris = build(n_tri)
        E0, V, w, deg = ground_manifold_sz(n, edges, deg_tol=deg_tol)
        m = V.shape[1]
        gap_sector = float(w[m] - E0) if len(w) > m else np.nan
        covers, total = enumerate_covers(tris)
        if covers:
            W = np.stack([cover_state(n, c) for c in covers], axis=1)
            sv = np.linalg.svd(W, compute_uv=False)
            rank = int(np.sum(sv > 1e-10))
            wmin = min(float(dc.subspace_fidelity(W[:, k], V))
                       for k in range(W.shape[1]))
        else:
            rank, wmin = 0, np.nan
        out.append(dict(n_tri=n_tri, n=n, E0=E0, deg=deg,
                        gap_sector=gap_sector, n_covers=len(covers),
                        total=total, cover_rank=rank, min_cover_weight=wmin))
    return out


def save_records(records, path):
    """Persist a list of flat dicts as one npz (arrays keyed by field)."""
    keys = records[0].keys()
    np.savez(path, **{k: np.array([r[k] for r in records]) for k in keys})


def load_records(path):
    """Inverse of save_records: list of dicts."""
    d = np.load(path)
    keys = list(d.keys())
    return [dict((k, d[k][i].item()) for k in keys)
            for i in range(len(d[keys[0]]))]


# ==========================================================================
# 8. Smoke test
# ==========================================================================
def smoke(verbose=True):
    """Fast end-to-end checks of every Sec. above (runs in a few seconds)."""
    say = print if verbose else (lambda *a, **k: None)

    # lattice builders reproduce Ahsan's clusters up to relabeling
    for n_tri, edges_ref in ((2, ORIG_EDGES), (3, MOD_EDGES)):
        n, edges, _ = delta_chain(n_tri)
        rel = ({2: 0, 0: 1, 1: 2, 3: 3, 4: 4} if n_tri == 2
               else {2: 0, 0: 1, 1: 2, 3: 3, 4: 4, 5: 5, 6: 6})
        got = {tuple(sorted((rel[a], rel[b]))) for (a, b) in edges}
        assert got == {tuple(sorted(e)) for e in edges_ref}, n_tri
    say("[1] delta_chain(2/3) ≅ ORIG/MOD clusters")

    # ground manifolds
    E5, V5, _ = ground_manifold(ORIG_N, ORIG_EDGES)
    E7, V7, _ = ground_manifold(MOD_N, MOD_EDGES)
    assert abs(E5 + 6) < 1e-9 and V5.shape[1] == 6
    assert abs(E7 + 9) < 1e-9 and V7.shape[1] == 8
    say(f"[2] E0(orig) = {E5:.6f} deg 6 ; E0(mod) = {E7:.6f} deg 8")

    # frustration-freeness of the modified manifold (strong form)
    for t, Ht in triangle_hams(MOD_N, MOD_TRIANGLES).items():
        assert np.allclose(restrict(V7, Ht), -3 * np.eye(8), atol=1e-8), t
    say("[3] P H_△ P = −3 P for A, B, C")

    # covers: 7/27, all exclusion-compliant, all exact ground states
    covers, total = enumerate_covers(MOD_TRIANGLES)
    assert len(covers) == 7 and total == 27
    assert all(mutually_exclusive(c) for c in covers)
    for c in covers:
        psi = cover_state(MOD_N, c)
        assert abs(triangle_energy(MOD_N, psi, MOD_TRIANGLES[0]) + 3) < 1e-9
        assert abs(dc.subspace_fidelity(psi, V7) - 1) < 1e-9
    say("[4] 7/27 valid covers, mutual exclusion holds, all in the manifold")

    # Lemma 1 operator identity
    assert singlet_absorption_residual(n_env=2) < 1e-12
    say("[5] singlet absorption: ‖(S²_△−3/4)Π_s‖ = 0")

    # σ symmetry
    P = permutation_matrix(MOD_N, SIGMA_MOD)
    H7 = K.heisenberg_hamiltonian(MOD_N, MOD_EDGES).to_matrix()
    assert np.abs(P @ H7 - H7 @ P).max() < 1e-12
    Ve, Vo = parity_split(V7, P)
    assert Ve.shape[1] == 4 and Vo.shape[1] == 4
    say("[6] [P_σ, H] = 0 ; manifold splits 4 even + 4 odd")

    # contraction teleportation: R maps mod covers into the orig manifold
    R = contraction_map()
    r, _, svals, _ = manifold_map(V5, V7, R)
    assert np.sum(svals > 1e-10) == 6
    say(f"[7] rank(V5† R V7) = 6 (svals {np.round(svals, 4)})")

    # chain law at n_tri = 4
    rec = chain_scan([4])[0]
    assert abs(rec["E0"] + 12) < 1e-9 and rec["deg"] == 10
    assert rec["n_covers"] == 9 and rec["cover_rank"] == 5
    say(f"[8] n_tri=4: E0={rec['E0']:.4f}, deg={rec['deg']}, "
        f"covers={rec['n_covers']}, rank={rec['cover_rank']}")

    say("smoke: ALL OK")
    return True


if __name__ == "__main__":
    smoke()
