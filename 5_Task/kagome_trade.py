"""
kagome_trade.py — Task-5 engine: the trace-vs-projection duality for dummy-
triangle embeddings of the 19-site kagome cluster.

Task 5 embeds three dummy triangles (six dummy qubits, 19+6 = 25) into the
19-site KAFH cluster of Tasks 1-3 and asks for a spin-liquid-like state on
the BLACK (original) part, "tracing the dummies out at the last step".  The
Task-4 theorems turn that request into a sharp dichotomy, which this module
operationalizes:

  * TRACING the dummy qubits is the literal request, but it returns a MIXED
    black state.  The 19-site ground manifold is a single doublet (dim 2),
    so Tr(rho H19) = E19 forces rank(rho_black) <= 2 — a hard budget of
    <= 1 bit of black-blue entanglement.  Any dummy entanglement beyond
    that budget pays energy and pushes <S2> off 3/4.
  * PROJECTING each dummy pair (apex, primed copy) onto the singlet is the
    EXACT local map back (Task-4 Theorem 4, singlet teleportation): the
    black state stays pure and inside the doublet — but the projection only
    succeeds with probability p < 1, and the failed (triplet) branches are
    NOT correctable: on the 7-site miniature, uniformly over the dim-8
    ground manifold,
        singlet   p = 9/16          image weight in the orig manifold = 1
        triplet0  p = 7/48          weight = 13/21
        triplet+  p = 7/48          weight = 13/21
        triplet-  p = 7/48          weight = 13/21
    (weight = basis-invariant ratio tr(Po K P K†)/tr(K P K†); the equality
    of the three triplet channels is exact.  NOTE: an earlier session
    recorded 5/9 and 4/9 for the triplet weights — those numbers came from
    a basis-DEPENDENT per-column average and are superseded by 13/21.)
  * Between the two extremes there is a 4-point TRADE CURVE: project k of
    the three dummy pairs and trace the other 3-k, k = 0..3, and score
    success probability vs purity vs <S2> vs Tr(rho H19) vs doublet weight.
    `trade_curve` measures it on any 25-qubit state.

Depth-0 baselines (verified 2026-08-07, all reproduced by `smoke()`):
  (a1) best of the 72 maximum matchings of the 25-site lattice:
       <H19> = -27.0000 (7.363%) — exactly the bare 19-site dimer ceiling;
       black-dimer distribution 9->17, 8->22, 7->25, 6->8 arrangements.
  (a2) best product of EXACT Delta-chain ground states (Theorem 5's trap):
       <H19> = -24.4619 (16.07%) at zero polarization = -15 + min over the
       12-dim inferior manifold of its 9 black bonds; virtual-bond
       polarization cross terms squeeze it slightly further, to -24.4858.
       The chains spend 6 of their 15 bonds on primed copies that are then
       discarded — being optimal on the enlarged lattice is the WORST of
       the three depth-0 strategies here.
  (a3) product optimized against the BLACK target: <H19> = -27.665016
       (5.082%) = -15 (all-black chain) + E0 of the 8 remaining black sites.

Geometry (all asserted at import/construction time):
  * Exactly 5 corner triplets split the lattice into two delta_chain(5)
    fragments: (1,10,11), (2,11,12), (4,11,14), (6,11,16), (8,11,17) —
    all contain 11 (the chord of the 10-triangle adjacency ring must be
    cut) plus two antipodal ring cuts.  Ahsan's figure draws (6,11,16);
    `insert_dummies` defaults to it but takes any of the five.
  * The scan MUST vertex-split corner sites per TRIANGLE, not per edge:
    corners 2 and 11 share triangle (1,2,11), and an edge-based split
    silently loses the triplet (2,11,12) (bug caught in the prelims).
  * The 25-site lattice has 39 bonds and 13 triangles; 2T - N = +1,
    identical to the original (+1): dummies do NOT de-frustrate, they
    relocate the frustration (0 valid covers out of 3^13).
  * 6 of the 30 H19 bonds are VIRTUAL — absent from the 25-site lattice,
    mediated only through the dummy triangles.  Canonical set:
    (1,11), (2,11), (4,6), (5,6), (16,17), (16,18).

Qubit layout convention (used by every observable below): black sites keep
their Task-1 labels 0..18, dummies are 19..24 with pair k = (19+2k, 20+2k)
= (apex, primed copy), sorted by corner.  A 25-qubit statevector is 0.5 GB;
rho_black would be 4 TB and is NEVER formed — every black observable is
evaluated through the view M = psi.reshape(2^6, 2^19) (row = dummy config,
column = black config) or bond-by-bond with the Task-2 blocked kernels.

Contents
  1. 19-site geometry (single source of truth: dc.load_task1_lattice),
     corner sites, the exhaustive splitting-triplet scan
  2. `insert_dummies` -> Insertion (the 25-site lattice + all bookkeeping)
  3. Depth-0 baselines: maximum matchings (a1), naive DnC bound (a3),
     exact-chain-manifold product optimum (a2, with/without polarization)
  4. Black observables on n>=19-qubit states: <H19>, purity, entropy,
     collective <S2_black>, doublet weight — all matrix-free
  5. Pair projections (singlet/triplet bras) and the trade curve
  6. The 7-site miniature Bell table (p = 9/16, 7/48; weights 1, 13/21)
  7. Persistence helpers + smoke test

Imports the Task-1/2/3/4 engines from their folders:
K = kagome_hva, dc = kagome_dc, er = kagome_er, kd = kagome_delta.
`python kagome_trade.py` runs the smoke test.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import networkx as nx

_HERE = Path(__file__).resolve().parent
for _t in ("1_Task", "2_Task", "3_Task", "4_Task"):
    _p = str(_HERE.parent / _t)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kagome_hva as K      # noqa: E402  Task-1 engine
import kagome_dc as dc      # noqa: E402  Task-2 engine (blocked kernels)
import kagome_er as er      # noqa: E402  Task-3 engine (cover states)
import kagome_delta as kd   # noqa: E402  Task-4 engine (manifolds, miniature)

# 2-qubit bond generator XX+YY+ZZ (same convention as K/dc/kd)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.array([[1, 0], [0, -1]], complex)
_BOND = np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z)

E19_REF = -29.146168        # ed19.npz E0 (2_Task/results), doublet (dim 2)
ED19_PATH = _HERE.parent / "2_Task" / "results" / "ed19.npz"


def load_doublet(path=ED19_PATH):
    """(E19, V19) with V19 = the two 2^19 doublet columns of the exact
    19-site ground manifold (2_Task/results/ed19.npz), RE-ORTHONORMALIZED
    by SVD: the stored columns have |<c0|c1>| = 1.5e-4 (independent Lanczos
    runs), which would bias every doublet_weight at the 1e-4 level."""
    d = np.load(path)
    U, s, _ = np.linalg.svd(d["V"], full_matrices=False)
    assert np.all(s > 0.99), s
    return float(np.atleast_1d(d["E0"])[0]), U


# ==========================================================================
# 1. 19-site geometry and the splitting-triplet scan
# ==========================================================================
def lattice19():
    """(edges19, tri19): the 30 bonds from the Task-1 notebook (single
    source of truth) and the 10 triangles, derived as the 3-cliques of the
    bond graph (asserted: the triangles' edges ARE the 30 bonds)."""
    edges, _ = dc.load_task1_lattice()
    eset = {tuple(sorted(e)) for e in edges}
    adj = {s: set() for s in range(19)}
    for a, b in eset:
        adj[a].add(b)
        adj[b].add(a)
    tris = sorted(tuple(sorted((i, j, k))) for i in range(19)
                  for j in adj[i] if j > i for k in adj[i] & adj[j] if k > j)
    assert len(tris) == 10 and {tuple(sorted(p)) for t in tris
                                for p in combinations(t, 2)} == eset
    return sorted(eset), tris


EDGES19, TRI19 = lattice19()
E19_SET = set(EDGES19)


def corner_sites(tri19=TRI19):
    """The 11 sites shared by exactly two triangles (the possible dummy
    insertion points); C(11,3) = 165 candidate triplets."""
    cnt: dict = {}
    for t in tri19:
        for s in t:
            cnt[s] = cnt.get(s, 0) + 1
    return sorted(s for s, c in cnt.items() if c == 2)


CORNERS = corner_sites()
_TRIS_OF = {c: [i for i, t in enumerate(TRI19) if c in t] for c in CORNERS}


def splitting_triplets():
    """
    Exhaustive scan of the 165 corner triplets: which insertions split the
    black+primed part into TWO delta_chain(5) fragments?  The split is done
    per TRIANGLE (corner c appears as itself in one of its two triangles
    and as a primed copy in the other) — an edge-based split is WRONG when
    two chosen corners share a triangle (2 and 11 share (1,2,11)) and
    silently loses the triplet (2,11,12).  Returns the 5 triplets; which
    triangle hosts the copy is irrelevant here (pure relabeling).
    """
    nD, eD, _ = kd.delta_chain(5)
    GD = nx.Graph(eD)
    hits = []
    for trip in combinations(CORNERS, 3):
        lab = {(c, _TRIS_OF[c][1]): 100 + k for k, c in enumerate(trip)}
        G = nx.Graph()
        for i, t in enumerate(TRI19):
            v = [lab.get((s, i), s) for s in t]
            G.add_edges_from([(v[0], v[1]), (v[1], v[2]), (v[0], v[2])])
        comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if len(comps) == 2 and all(len(c) == 11 and nx.is_isomorphic(c, GD)
                                   for c in comps):
            hits.append(trip)
    return sorted(hits)


SPLITTING_TRIPLETS = [(1, 10, 11), (2, 11, 12), (4, 11, 14), (6, 11, 16),
                      (8, 11, 17)]
CANONICAL = (6, 11, 16)     # the triplet drawn in Ahsan's Task-5 figure


# ==========================================================================
# 2. Dummy insertion -> the 25-site lattice
# ==========================================================================
@dataclass(frozen=True)
class Insertion:
    """Everything about one dummy insertion, in GLOBAL labels (black 0..18
    as in Task 1, dummies 19..24).  Built by `insert_dummies`; every field
    is asserted against the counting/isomorphism invariants there."""
    triplet: tuple                 # the three corner sites, sorted
    new: dict                      # corner -> (apex, primed copy)
    hosts: dict                    # corner -> triangle handed to the copy
    edges: list                    # 39 bonds of the 25-site lattice
    triangles: list                # 13 triangles (10 relabeled + 3 dummy)
    virtual: list                  # 6 H19 bonds absent from `edges`
    dummy_triangles: list          # (corner, apex, copy) x 3
    dummy_edges: list              # their 9 bonds
    pairs: list                    # [(apex, copy)] x 3 — the projection pairs
    apices: list                   # the 3 apex sites
    frag_black: list               # all-black delta_chain(5) fragment (11)
    frag_mixed: list               # 8 black + 3 primed fragment (11)
    n: int = 25


def insert_dummies(triplet=CANONICAL):
    """
    Build the 25-site lattice for one of the 5 splitting triplets.  For
    each corner c the primed copy joins the fragment CONTAINING SITE 0
    (deterministic host rule; for the canonical triplet it reproduces the
    hosts of Ahsan's figure: 6->(4,5,6), 11->(1,2,11), 16->(16,17,18)),
    and c keeps its other triangle.  Dummy pair k = (19+2k, 20+2k), corners
    sorted.  Asserts: 39 bonds, 13 triangles, virtual = H19 \\ E25,
    2T - N = +1, and both fragments isomorphic to delta_chain(5).
    """
    triplet = tuple(sorted(triplet))
    if triplet not in SPLITTING_TRIPLETS:
        raise ValueError(f"{triplet} does not split the lattice; valid: "
                         f"{SPLITTING_TRIPLETS}")
    # triangle-adjacency ring+chord, cut at the triplet's corners
    Gt = nx.Graph()
    Gt.add_nodes_from(range(10))
    for c in CORNERS:
        if c not in triplet:
            Gt.add_edge(*_TRIS_OF[c])
    sides = list(nx.connected_components(Gt))
    assert len(sides) == 2
    tri0 = next(i for i, t in enumerate(TRI19) if 0 in t)
    side0 = next(s for s in sides if tri0 in s)      # copies join this side

    # simultaneous per-TRIANGLE construction (sequential per-corner edits are
    # WRONG when two chosen corners share their host triangle, e.g. corners
    # 10 and 11 both hosting (10,11,12) in the (1,10,11) insertion: that
    # triangle receives TWO primed copies)
    new, hosts, host_of = {}, {}, {}
    for k, c in enumerate(triplet):
        apex, copy = 19 + 2 * k, 20 + 2 * k
        host_i = next(i for i in _TRIS_OF[c] if i in side0)
        new[c], hosts[c], host_of[c] = (apex, copy), TRI19[host_i], host_i
    triangles = [tuple(sorted(new[s][1] if host_of.get(s) == i else s
                              for s in t)) for i, t in enumerate(TRI19)]
    dummy_triangles = [(c,) + new[c] for c in triplet]
    triangles += [tuple(sorted(t)) for t in dummy_triangles]
    dummy_edges = sorted(tuple(sorted(p)) for t in dummy_triangles
                         for p in combinations(t, 2))
    edges = {tuple(sorted(p)) for t in triangles for p in combinations(t, 2)}
    virtual = E19_SET - edges                  # the missing H19 bonds

    assert len(edges) == 39 and len(triangles) == 13
    assert 2 * len(triangles) - 25 == 1        # counting invariant
    G25 = nx.Graph(sorted(edges))
    Gf = G25.copy()
    Gf.remove_edges_from(dummy_edges)
    Gf.remove_nodes_from([a for a, _ in new.values()])   # isolated apices
    frags = sorted((sorted(c) for c in nx.connected_components(Gf)), key=len)
    assert [len(f) for f in frags] == [11, 11]
    nD, eD, _ = kd.delta_chain(5)
    GD = nx.Graph(eD)
    assert all(nx.is_isomorphic(G25.subgraph(f), GD) for f in frags)
    fb, fm = ((frags[0], frags[1]) if max(frags[0]) <= 18
              else (frags[1], frags[0]))
    assert max(fb) <= 18 and any(s > 18 for s in fm)
    return Insertion(triplet=triplet, new=new, hosts=hosts,
                     edges=sorted(edges), triangles=sorted(triangles),
                     virtual=sorted(virtual),
                     dummy_triangles=dummy_triangles, dummy_edges=dummy_edges,
                     pairs=[new[c] for c in triplet],
                     apices=[new[c][0] for c in triplet],
                     frag_black=fb, frag_mixed=fm)


def fragment_data(ins: Insertion):
    """
    Local bookkeeping for the two 11-site fragments.  Returns a dict with,
    per fragment ('black', 'mixed'): sites, loc (global->local), edges
    (the 15 chain bonds, local), h19 (the H19 bonds INDUCED on the
    fragment's black sites, local — for shared-host insertions this can
    include virtual bonds that are internal to the black fragment, e.g.
    (10,11) in the (1,10,11) insertion), and globally: virt_pairs = the
    H19 bonds CROSSING the fragments (necessarily virtual) as (black-frag
    local, mixed-frag local) pairs, mixed_blk / mixed_primed site lists.
    Canonical shape: h19 = 15/9, cross = 6; the general accounting
    15-chain + induced + cross = 30 is asserted instead.
    """
    out = {}
    for name, sites in (("black", ins.frag_black), ("mixed", ins.frag_mixed)):
        loc = {s: i for i, s in enumerate(sites)}
        eloc = [(loc[a], loc[b]) for a, b in ins.edges
                if a in loc and b in loc]
        h19 = [(loc[a], loc[b]) for a, b in EDGES19
               if a in loc and b in loc and a <= 18 and b <= 18]
        assert len(eloc) == 15
        out[name] = dict(sites=sites, loc=loc, edges=eloc, h19=h19)
    lb, lm = out["black"]["loc"], out["mixed"]["loc"]
    cross = [(a, b) for a, b in EDGES19
             if (a in lb and b in lm) or (a in lm and b in lb)]
    assert all(tuple(sorted(e)) in ins.virtual for e in cross)
    out["virt_pairs"] = [(lb[a] if a in lb else lb[b],
                          lm[b] if b in lm else lm[a])
                         for a, b in cross]
    out["mixed_blk"] = [s for s in ins.frag_mixed if s <= 18]
    out["mixed_primed"] = [s for s in ins.frag_mixed if s > 18]
    assert (len(out["black"]["h19"]) + len(out["mixed"]["h19"])
            + len(cross)) == 30
    return out


# ==========================================================================
# 3. Depth-0 baselines
# ==========================================================================
def maximum_matchings(ins: Insertion):
    """
    ALL maximum matchings (size 12, one site uncovered) of the 25-site
    lattice, by exact recursion (72 for the canonical insertion).  For a
    product of singlets, Tr(rho_black h_ij) = -3 exactly when the dimer
    (i,j) is a black-black H19 bond, so <H19> = -3 * n_black(matching):
    the analytic (a1) rule, cross-checked on a statevector in smoke().
    """
    adjac = {v: [] for v in range(ins.n)}
    for a, b in ins.edges:
        adjac[a].append(b)
        adjac[b].append(a)
    out = []

    def rec(v, used, m, skipped):
        while v < ins.n and v in used:
            v += 1
        if v == ins.n:
            if len(m) == (ins.n - 1) // 2:
                out.append(tuple(m))
            return
        if skipped is None:
            rec(v + 1, used, m, v)          # v is the one uncovered site
        for w in adjac[v]:
            if w > v and w not in used:
                m.append((v, w))
                rec(v + 1, used | {v, w}, m, skipped)
                m.pop()

    rec(0, frozenset(), [], None)
    return out


def n_black_dimers(matching):
    """Number of dimers of a matching that are black-black H19 bonds."""
    return sum(1 for a, b in matching
               if a <= 18 and b <= 18 and tuple(sorted((a, b))) in E19_SET)


def best_cover_baseline(ins: Insertion):
    """(a1): dict with the matching count, the black-dimer distribution,
    the best matching (preferring one whose 3 non-black dimers are exactly
    the dummy pairs, for reuse as a smoke-test state) and E = -3*max."""
    mms = maximum_matchings(ins)
    counts = [n_black_dimers(mm) for mm in mms]
    best = max(counts)
    pairset = {tuple(sorted(p)) for p in ins.pairs}
    pick = next((mm for mm, c in zip(mms, counts) if c == best and
                 {tuple(sorted(d)) for d in mm} >= pairset),
                mms[int(np.argmax(counts))])
    from collections import Counter
    return dict(n_matchings=len(mms), distribution=dict(Counter(counts)),
                best_black=best, energy=-3.0 * best, matching=pick)


def naive_dnc_bound(ins: Insertion):
    """
    (a3): the product bound against the BLACK target = E0 of the H19 graph
    INDUCED on the black fragment + E0 of the one induced on the mixed
    fragment's black sites (cross bonds contribute 0 for unpolarized
    factors).  Canonical: the black-fragment induced graph is exactly the
    15-bond chain, so -15 - 12.665016 = -27.665016 (5.082%); shared-host
    insertions add intra-black virtual bonds to the first graph and the
    chain value -15 no longer applies verbatim.
    """
    fd = fragment_data(ins)

    def e0_induced(sites):
        loc = {s: i for i, s in enumerate(sites)}
        e = [(loc[a], loc[b]) for a, b in EDGES19 if a in loc and b in loc]
        E0, _, _, deg = kd.ground_manifold_sz(len(sites), e)
        return E0, deg, len(e)

    Eb, degb, nb = e0_induced(ins.frag_black)
    Em, degm, nm = e0_induced(fd["mixed_blk"])
    return dict(E_chain=Eb, E_black_rest=Em, deg_rest=degm,
                energy=Eb + Em, n_rest_bonds=nm, n_black_bonds=nb)


def full_ground_manifold(n, edges):
    """
    The COMPLETE ground manifold of an odd-n cluster whose ground states
    are all doublets (the Delta chains): the S^z=+1/2 sector columns of
    kd.ground_manifold_sz plus their normalized S^- partners (dc.apply_
    s_minus — the (X - iY)/2 sign; +i builds S^+ which ANNIHILATES the
    doublet tops), orthonormalized by SVD (numpy QR without pivoting is
    NOT a reliable column-space basis — Task-4 gotcha).  Returns (E0, M)
    with M of shape (2^n, deg).
    """
    E0, V, _, deg = kd.ground_manifold_sz(n, edges)
    cols = [V[:, k] for k in range(V.shape[1])]
    for k in range(V.shape[1]):
        w = dc.apply_s_minus(V[:, k].copy(), n)
        cols.append(w / np.linalg.norm(w))
    M = np.stack(cols, axis=1)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > 1e-9).sum())
    assert r == deg, (r, deg)
    return E0, U[:, :r]


def _apply_bonds(psi, n, bonds):
    """Sum of _BOND over `bonds` applied to psi (blocked accumulate)."""
    out = np.zeros_like(psi)
    for a, b in bonds:
        dc.apply2_accumulate(n, out.reshape((2,) * n), psi.reshape((2,) * n),
                             _BOND, a, b)
    return out


def _apply_pauli(psi, n, q, P):
    T = np.moveaxis(psi.reshape((2,) * n), n - 1 - q, 0)
    return np.moveaxis(np.tensordot(P, T, axes=(1, 0)), 0,
                       n - 1 - q).reshape(-1)


def _restrict(M, apply_col):
    """Operator restricted to the manifold M (columns 2^n)."""
    cols = np.stack([apply_col(M[:, k]) for k in range(M.shape[1])], axis=1)
    return M.conj().T @ cols


def chain_product_baseline(ins: Insertion, n_starts=20, iters=60, seed=0):
    """
    (a2), Theorem 5's trap: the best <H19> over PRODUCTS of exact ground
    states of the two chain fragments.  Two levels are returned:
      energy_nocross — exact: -15 + min eigenvalue of the mixed chain's
        9 black bonds restricted to its 12-dim ground manifold (canonical:
        -15 - 9.461865 = -24.461865, 16.07%).  The minimum is a DEGENERATE
        doublet, so polarization-dependent diagnostics (<S2>) are not unique
        at this level.
      energy — alternating optimization that also exploits the virtual
        bonds via <S_i><S_j> polarization cross terms between the fragments
        (canonical: -24.4858, 15.99%).  Diagnostics (mixed-fragment purity,
        black-blue entropy, collective black <S2>) are for this optimum.
    """
    fd = fragment_data(ins)
    E0b, Mb = full_ground_manifold(11, fd["black"]["edges"])
    E0m, Mm = full_ground_manifold(11, fd["mixed"]["edges"])
    assert abs(E0b + 15) < 1e-9 and abs(E0m + 15) < 1e-9

    A9 = _restrict(Mm, lambda v: _apply_bonds(v, 11, fd["mixed"]["h19"]))
    w9 = np.linalg.eigvalsh(A9)
    e_nocross = -15.0 + float(w9[0])

    paulis = [_X, _Y, _Z]
    Xb = [[_restrict(Mb, lambda v, q=qb, P=P: _apply_pauli(v, 11, q, P))
           for P in paulis] for qb, _ in fd["virt_pairs"]]
    Xm = [[_restrict(Mm, lambda v, q=qm, P=P: _apply_pauli(v, 11, q, P))
           for P in paulis] for _, qm in fd["virt_pairs"]]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        c = rng.normal(size=Mb.shape[1]) + 1j * rng.normal(size=Mb.shape[1])
        c /= np.linalg.norm(c)
        for _ in range(iters):
            Hd = A9 + sum((c.conj() @ Xb[v][al] @ c).real * Xm[v][al]
                          for v in range(6) for al in range(3))
            d = np.linalg.eigh(Hd)[1][:, 0]
            Hc = sum((d.conj() @ Xm[v][al] @ d).real * Xb[v][al]
                     for v in range(6) for al in range(3))
            c = np.linalg.eigh(Hc)[1][:, 0]
        e = (-15.0 + (d.conj() @ A9 @ d).real
             + sum((c.conj() @ Xb[v][al] @ c).real
                   * (d.conj() @ Xm[v][al] @ d).real
                   for v in range(6) for al in range(3)))
        if best is None or e < best[0]:
            best = (e, Mb @ c, Mm @ d)
    e_cross, psi_b, psi_m = best
    diag = fragment_product_diagnostics(ins, psi_b, psi_m)
    return dict(energy_nocross=e_nocross, energy=float(e_cross),
                psi_black=psi_b, psi_mixed=psi_m,
                manifold_spectrum=w9, **diag)


def fragment_product_diagnostics(ins: Insertion, psi_b, psi_m):
    """
    Black diagnostics of the product state |psi_b> (all-black chain) x
    |psi_m> (mixed chain) x |apices up>: mixed-fragment purity and
    black-blue entropy (from tracing its 3 primed sites — the apices are
    unentangled and add nothing), and collective <S2_black> including the
    polarization cross term between the fragments.
    """
    fd = fragment_data(ins)
    lm = fd["mixed"]["loc"]
    primed_loc = [lm[s] for s in fd["mixed_primed"]]
    perm = ([10 - q for q in primed_loc]
            + [10 - q for q in range(11) if q not in primed_loc])
    M = np.transpose(psi_m.reshape((2,) * 11), perm).reshape(8, 256)
    lam = np.linalg.eigvalsh(M @ M.conj().T).clip(min=0)
    lam /= lam.sum()
    purity = float((lam ** 2).sum())
    entropy = float(-(lam[lam > 1e-14] * np.log2(lam[lam > 1e-14])).sum())

    def s2_and_spin(psi, sites_loc):
        s2 = 0.75 * len(sites_loc)
        for a, b in combinations(sites_loc, 2):
            s2 += 0.5 * kd.bond_energy(11, psi, (a, b))
        spin = 0.5 * sum(np.array([np.vdot(psi, _apply_pauli(psi, 11, q, P))
                                   .real for P in (_X, _Y, _Z)])
                         for q in sites_loc)
        return s2, spin

    s2_b, spin_b = s2_and_spin(psi_b, list(range(11)))
    s2_m, spin_m = s2_and_spin(psi_m, [lm[s] for s in fd["mixed_blk"]])
    return dict(purity=purity, entropy=entropy,
                s2_black=float(s2_b + s2_m + 2.0 * spin_b @ spin_m))


# ==========================================================================
# 3b. Cut accounting: WHY one insertion beats another
# ==========================================================================
def bond_energies19(psi19):
    """{bond: <XX+YY+ZZ>} over the 30 H19 bonds of a 19-qubit state.  On the
    exact ground doublet these are multiplet invariants (the bond operator
    is an SU(2) scalar, so Wigner-Eckart makes them identical on both
    columns — asserted in smoke to 2e-13), which is what makes them a
    legitimate per-bond 'price list' of the target state."""
    T = psi19.reshape((2,) * 19)
    return {b: float(np.real(dc.bond_vdot(19, T, T, _BOND, b[0], b[1])))
            for b in EDGES19}


def cut_accounting(ins: Insertion, E_bond, E19=E19_REF):
    """
    The exact energy accounting of an insertion at product level.  The
    insertion induces a BIPARTITION of the 19 black sites (the two
    fragments); the H19 bonds joining them are the `crossing` ones, and
    they are exactly the bonds a product state cannot pay for.  Then

        a3  =  E0(H19 minus the crossing bonds)                    (identity)
            =  E19 - sum_cross + defect,   defect <= 0

    where `sum_cross` is the crossing bonds' energy IN THE TARGET STATE
    (what the cut throws away) and `defect` is what re-optimizing the two
    disconnected pieces wins back.  Returns both terms plus the per-bond
    detail, so the comparison between insertions is a decomposition and not
    a correlation: `sum_cross` = (number of crossing bonds) x (their mean
    energy), and only the FIRST factor differs across insertions here.
    """
    fd = fragment_data(ins)
    lb, lm = fd["black"]["loc"], fd["mixed"]["loc"]
    cross = [b for b in ins.virtual
             if (b[0] in lb and b[1] in lm) or (b[0] in lm and b[1] in lb)]
    internal = [b for b in ins.virtual if b not in cross]
    a3 = naive_dnc_bound(ins)["energy"]
    sum_cross = sum(E_bond[b] for b in cross)
    return dict(triplet=ins.triplet, crossing=cross, internal=internal,
                n_cross=len(cross), n_virtual=len(ins.virtual),
                sum_cross=sum_cross, mean_cross=sum_cross / len(cross),
                sum_virtual=sum(E_bond[b] for b in ins.virtual),
                a3=a3, defect=a3 - (E19 - sum_cross),
                err_pct=100.0 * abs(a3 - E19) / abs(E19))


def cut_identity_residual(ins: Insertion):
    """
    Verifies a3 = E0(H19 with the crossing bonds deleted) directly: builds
    that 19-site graph, solves each connected component exactly, and
    returns |sum of component ground energies - a3|.  (The identity is why
    `cut_accounting` is an accounting and not a fit.)
    """
    import networkx as nx
    fd = fragment_data(ins)
    lb, lm = fd["black"]["loc"], fd["mixed"]["loc"]
    cross = {b for b in ins.virtual
             if (b[0] in lb and b[1] in lm) or (b[0] in lm and b[1] in lb)}
    G = nx.Graph([e for e in EDGES19 if e not in cross])
    G.add_nodes_from(range(19))
    tot = 0.0
    for comp in nx.connected_components(G):
        sites = sorted(comp)
        loc = {s: i for i, s in enumerate(sites)}
        e = [(loc[a], loc[b]) for a, b in EDGES19
             if a in loc and b in loc and tuple(sorted((a, b))) not in cross]
        if not e:
            continue
        tot += kd.ground_manifold_sz(len(sites), e)[0]
    return abs(tot - naive_dnc_bound(ins)["energy"])


def block_entropy(psi, n, block):
    """Von Neumann entropy (bits) of `block` (a list of qubit labels) for a
    PURE n-qubit state — the internal-entanglement measure of Sec. 1c, to
    be contrasted with the black|blue entropy of Lemma B (a different cut
    of a different bipartition)."""
    ax_keep = [n - 1 - q for q in block]
    ax_rest = [a for a in range(n) if a not in ax_keep]
    M = np.transpose(psi.reshape((2,) * n),
                     ax_keep + ax_rest).reshape(2 ** len(block), -1)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-14] / s.sum()
    return float(-(s * np.log2(s)).sum())


def sz_basis(n, sz):
    """Sorted uint32 bitmasks of the S^z = sz sector (|1> = spin down).
    Vectorized-friendly replacement of kd.sz_sector_basis for large n
    (C(25,12) = 5,200,300 states)."""
    n_down = round((n - 2 * sz) / 2)
    if not 0 <= n_down <= n:
        raise ValueError(f"empty sector sz={sz} for n={n}")
    basis = np.fromiter((sum(1 << q for q in pos)
                         for pos in combinations(range(n), n_down)),
                        dtype=np.uint32)
    basis.sort()
    return basis


def sector_ground(n, edges, sz=None, k=6, maxiter=None, tol=0, ncv=None,
                  v0=None):
    """
    Low spectrum of H = sum XX+YY+ZZ inside one S^z sector, MATRIX-FREE:
    the matvec is one diagonal multiply plus, per bond, a fancy-indexed
    gather/scatter over the antiparallel states (per bond the spin-flip
    partner is unique, so `y[rows] += 2 x[cols]` is race-free).  H is real
    in this basis, so everything runs in float64 (a sector vector at n=25
    is 42 MB vs 0.5 GB for the complex full register).  Returns (w, V,
    basis): the k lowest Ritz values, sector eigencolumns, and bitmasks.

    CAVEAT (Task-4 gotcha, restated for Lanczos at 5.2M): eigsh can silently
    return an incomplete degenerate multiplet.  Callers must check the
    bottom-cluster multiplicity against per-column <S2> on the embedded
    states (see smoke / the ed25 precompute) rather than trust k alone.
    """
    sz = 0.5 * (n % 2) if sz is None else sz
    basis = sz_basis(n, sz)
    diag = np.zeros(len(basis))
    bond_idx = []
    for (i, j) in edges:
        anti = ((basis >> i) ^ (basis >> j)) & 1 == 1
        diag += np.where(anti, -1.0, 1.0)
        flipped = basis[anti] ^ np.uint32((1 << i) | (1 << j))
        cols = np.searchsorted(basis, flipped).astype(np.int32)
        rows = np.nonzero(anti)[0].astype(np.int32)
        bond_idx.append((rows, cols))

    from scipy.sparse.linalg import LinearOperator, eigsh

    def matvec(x):
        y = diag * x
        for rows, cols in bond_idx:
            y[rows] += 2.0 * x[cols]
        return y

    A = LinearOperator((len(basis),) * 2, matvec=matvec, dtype=np.float64)
    w, V = eigsh(A, k=k, which="SA", maxiter=maxiter, tol=tol, ncv=ncv, v0=v0)
    order = np.argsort(w)
    return w[order], V[:, order], basis


def embed_sector(vec, basis, n):
    """Sector column -> full 2^n complex statevector."""
    out = np.zeros(2 ** n, complex)
    out[basis.astype(np.int64)] = vec
    return out


def s2_expect(psi, n, sites=None):
    """Collective <S2> (and, second return, the residual ||(S2 - <S2>)psi||
    as an exact-eigenstate check) over `sites` (default: all n)."""
    sites = range(n) if sites is None else sites
    parts = []
    for alpha in "xyz":
        acc = np.zeros_like(psi)
        for q in sites:
            _acc_pauli(acc, psi, n, q, alpha, 0.5)
        parts.append(acc)
    s2 = float(sum(np.vdot(a, a).real for a in parts))
    s2psi = np.zeros_like(psi)
    for alpha, a in zip("xyz", parts):
        for q in sites:
            _acc_pauli(s2psi, a, n, q, alpha, 0.5)
    resid = float(np.linalg.norm(s2psi - s2 * psi))
    return s2, resid


# ==========================================================================
# 4. Black observables on n-qubit states (black = qubits 0..18, matrix-free)
# ==========================================================================
def h19_energy(psi, n):
    """Tr(rho_black H19) = <psi| H19 |psi>, bond by bond with the blocked
    kernel (local black operators need no reduced density matrix)."""
    T = psi.reshape((2,) * n)
    return float(sum(dc.bond_vdot(n, T, T, _BOND, a, b).real
                     for a, b in EDGES19))


def black_gram(psi, n):
    """G = M M† with M = psi.reshape(2^(n-19), 2^19): the (tiny) dummy-side
    Gram matrix.  Its spectrum IS the spectrum of rho_black."""
    M = psi.reshape(2 ** (n - 19), 2 ** 19)
    return M @ M.conj().T


def black_purity_entropy(psi, n):
    """(purity, entropy_bits, rank_eps) of rho_black, from the Gram
    spectrum; rank_eps counts eigenvalues > 1e-12 (the <=2 budget test).
    A (numerically) zero state returns (nan, nan, 0)."""
    lam = np.linalg.eigvalsh(black_gram(psi, n)).clip(min=0)
    if lam.sum() < 1e-28:
        return float("nan"), float("nan"), 0
    lam = lam / lam.sum()
    nz = lam[lam > 1e-14]
    return (float((lam ** 2).sum()),
            float(-(nz * np.log2(nz)).sum()),
            int((lam > 1e-12).sum()))


def doublet_weight(psi, n, V19):
    """Weight of rho_black inside the exact 19-site ground doublet:
    sum_k ||M @ conj(g_k)||^2 (columns g_k of V19)."""
    M = psi.reshape(2 ** (n - 19), 2 ** 19)
    return float(sum(np.linalg.norm(M @ np.conj(V19[:, k])) ** 2
                     for k in range(V19.shape[1])))


def _acc_pauli(acc, psi, n, q, alpha, coef):
    """acc += coef * sigma^alpha_q psi, via axis slices (no full temps)."""
    accT, psiT = acc.reshape((2,) * n), psi.reshape((2,) * n)
    ax = n - 1 - q
    s0 = tuple(slice(None) if k != ax else 0 for k in range(n))
    s1 = tuple(slice(None) if k != ax else 1 for k in range(n))
    if alpha == "x":
        accT[s0] += coef * psiT[s1]
        accT[s1] += coef * psiT[s0]
    elif alpha == "y":
        accT[s0] += -1j * coef * psiT[s1]
        accT[s1] += 1j * coef * psiT[s0]
    else:
        accT[s0] += coef * psiT[s0]
        accT[s1] -= coef * psiT[s1]


def black_s2(psi, n, sites=range(19)):
    """
    Collective <S2> of the black sites on an n-qubit PURE state (equals
    Tr(rho_black S2_black)): sum_alpha || sum_i (sigma^alpha_i/2) psi ||^2,
    one 2^n accumulator at a time (~1.5 GB transient peak at n = 25).
    """
    tot = 0.0
    for alpha in "xyz":
        acc = np.zeros_like(psi)
        for q in sites:
            _acc_pauli(acc, psi, n, q, alpha, 0.5)
        tot += float(np.vdot(acc, acc).real)
        del acc
    return tot


def black_moments(psi, n, V19=None):
    """One-stop diagnostics dict for an n-qubit state (black = 0..18)."""
    pur, ent, rank = black_purity_entropy(psi, n)
    out = dict(energy=h19_energy(psi, n), purity=pur, entropy=ent,
               rank=rank, s2_black=black_s2(psi, n))
    if V19 is not None:
        out["doublet_weight"] = doublet_weight(psi, n, V19)
    return out


# ==========================================================================
# 5. Pair projections and the trade curve
# ==========================================================================
# Bell bras B[bit_a, bit_b] on a pair (a, b), a < b; the singlet sign
# matches er._SINGLET_PREP / kd.contraction_map (+ for |01> = bit_a=0).
_s2 = 1 / np.sqrt(2)
BELL_BRAS = {
    "singlet": np.array([[0, _s2], [-_s2, 0]], complex),
    "triplet0": np.array([[0, _s2], [_s2, 0]], complex),
    "triplet+": np.array([[1, 0], [0, 0]], complex),
    "triplet-": np.array([[0, 0], [0, 1]], complex),
}


def project_pairs(psi, n, outcomes):
    """
    Contract |psi> (n qubits) against Bell bras on qubit pairs:
    `outcomes` = {(a, b): bra_name}.  Returns (phi, prob, kept) with
    phi the NORMALIZED post-selected state on the remaining qubits (kept,
    ascending, relabeled 0..len-1 in order — so black labels survive
    whenever only dummies (>18) are projected), prob = ||contraction||^2.
    Successive tensordots only ever shrink the tensor (0.5 GB -> ...).
    """
    T = psi.reshape((2,) * n)
    axis_q = list(range(n - 1, -1, -1))          # axis k holds qubit axis_q[k]
    for (a, b), name in outcomes.items():
        B = BELL_BRAS[name].conj()
        ia, ib = axis_q.index(a), axis_q.index(b)
        T = np.tensordot(B, T, axes=([0, 1], [ia, ib]))
        axis_q = [q for q in axis_q if q not in (a, b)]
    prob = float(np.vdot(T, T).real)
    kept = sorted(axis_q)
    phi = T.reshape(-1)
    if prob > 1e-15:
        phi = phi / np.sqrt(prob)
    return phi, prob, kept


def trade_point(psi, subset, pairs, V19=None):
    """
    One point of the trade curve on a 25-qubit state: project the pairs in
    `subset` (indices into `pairs`) onto singlets, TRACE the rest.  Returns
    the black_moments of the post-selected state plus (k, subset, prob).
    """
    outcomes = {pairs[i]: "singlet" for i in subset}
    phi, prob, kept = project_pairs(psi, 25, outcomes)
    if prob < 1e-14:                 # dead branch (e.g. a pair in a pure
        m = dict(energy=np.nan, purity=np.nan, entropy=np.nan,  # triplet):
                 rank=0, s2_black=np.nan)                       # NO survivor
        if V19 is not None:
            m["doublet_weight"] = np.nan
    else:
        m = black_moments(phi, len(kept), V19=V19)
    m.update(k=len(subset), subset=tuple(subset), prob=prob)
    return m


def trade_curve(psi, V19=None, pairs=None, ins: Insertion = None):
    """
    THE central Task-5 measurement: all 2^3 = 8 projection subsets of the
    three dummy pairs on one 25-qubit state, grouped by k = |subset| =
    number of pairs projected onto singlets (the other 3-k are traced).
    k = 0 is Ahsan's literal request (pure trace), k = 3 is the exact
    Theorem-4 contraction (pure black state, probabilistic).  Returns a
    list of 8 point dicts (energy, purity, entropy, s2_black, doublet
    weight, success probability), subsets in k-ascending order.
    """
    if pairs is None:
        pairs = (ins or insert_dummies()).pairs
    pairs = [tuple(p) for p in pairs]
    points = []
    for k in range(len(pairs) + 1):
        for subset in combinations(range(len(pairs)), k):
            points.append(trade_point(psi, subset, pairs, V19=V19))
    return points


def embed_fragments(ins: Insertion, psi_black, psi_mixed, apex_state=None):
    """
    The 25-qubit product state |psi_black> (on ins.frag_black) x
    |psi_mixed> (on ins.frag_mixed) x |apices> (default |000> = all up),
    in the global qubit layout.  This is how the depth-0 fragment states
    (a2/a3) enter the generic 25-qubit machinery of Secs. 4-5; smoke()
    cross-checks its <H19>/purity/<S2> against the 11-qubit diagnostics.
    One 0.5 GB outer product plus one transposed copy (~1 GB transient).
    """
    if apex_state is None:
        apex_state = np.zeros(8, complex)
        apex_state[0] = 1.0
    F = np.multiply.outer(
        np.multiply.outer(psi_black.reshape((2,) * 11),
                          psi_mixed.reshape((2,) * 11)),
        apex_state.reshape((2, 2, 2)))
    # source axis of global qubit g: fragment axis j holds local qubit 10-j
    src = {}
    for base, sites in ((0, ins.frag_black), (11, ins.frag_mixed)):
        for j in range(11):
            src[sites[10 - j]] = base + j
    for j, a in enumerate(ins.apices):        # apex axis j holds apices[2-j]
        src[ins.apices[2 - j]] = 22 + j
    perm = [src[24 - k] for k in range(25)]   # target axis k <-> qubit 24-k
    return np.ascontiguousarray(np.transpose(F, perm)).reshape(-1)


def product_embed(n, factors):
    """
    General product embedding: `factors` = [(vec, sites), ...] with disjoint
    site lists covering 0..n-1 exactly; vec_k is a statevector on len(sites)
    qubits whose LOCAL qubit i sits on GLOBAL site sites[i].  Returns the
    2^n product state (same transpose bookkeeping as embed_fragments).
    """
    tensors, src, pos = [], {}, 0
    for vec, sites in factors:
        m = len(sites)
        tensors.append(np.asarray(vec, complex).reshape((2,) * m))
        for j in range(m):
            src[sites[m - 1 - j]] = pos + j
        pos += m
    assert sorted(src) == list(range(n)), "factors must cover 0..n-1"
    F = tensors[0]
    for T in tensors[1:]:
        F = np.multiply.outer(F, T)
    perm = [src[n - 1 - k] for k in range(n)]
    return np.ascontiguousarray(np.transpose(F, perm)).reshape(-1)


# ==========================================================================
# 6. The 7-site miniature Bell table (exact rationals)
# ==========================================================================
def miniature_bell_table():
    """
    Bell projections of the dummy pair (13, 0') = local (3, 4) on the
    UNIFORM mixture over the dim-8 ground manifold of the Task-4 modified
    cluster: for each outcome b, p_b = tr(K_b P_mod K_b†)/8 and weight_b =
    tr(P_orig K_b P_mod K_b†) / tr(K_b P_mod K_b†) — both basis-invariant.
    Exact values (asserted in smoke): p = 9/16, 7/48, 7/48, 7/48 and
    weight = 1, 13/21, 13/21, 13/21.  The three triplet channels are
    exactly equivalent, and NO local unitary on the black part can restore
    them to the manifold (weight < 1): the postselection cost is real.
    """
    _, Vm, _ = kd.ground_manifold(7, kd.MOD_EDGES)
    _, Vo, _ = kd.ground_manifold(5, kd.ORIG_EDGES)
    Po = Vo @ Vo.conj().T
    table = {}
    for name in BELL_BRAS:
        num = den = 0.0
        for k in range(Vm.shape[1]):
            v, p, kept = project_pairs(Vm[:, k], 7,
                                       {tuple(kd.DUMMY_PAIR): name})
            den += p
            num += p * float(np.vdot(v, Po @ v).real)
        table[name] = dict(p=den / Vm.shape[1], weight=num / den)
    return table


# ==========================================================================
# 6b. The two theorems in general form (system-independent statements)
# ==========================================================================
def budget_state(V, n_anc=None):
    """
    LEMMA B, general form.  Given an orthonormal ground manifold V (2^n x d)
    of any cluster, returns the state

        |Psi*> = d^(-1/2) sum_k |g_k> (x) |k>_anc      on n + n_anc qubits

    which saturates the rank bound: Tr(rho H) = E0 EXACTLY while rho =
    P_manifold/d, i.e. purity 1/d, S(system|anc) = log2(d) bits, rank d.
    Together with 'Tr(rho H) = E0 <=> supp(rho) is inside the manifold'
    (H - E0 >= 0 with kernel = the manifold), this shows log2(d) is the
    EXACT entanglement budget of the trace reading — 1 bit for a doublet
    is the corollary d = 2, not a feature of this lattice.
    """
    d = V.shape[1]
    n_anc = int(np.ceil(np.log2(d))) if n_anc is None else n_anc
    assert 2 ** n_anc >= d, "not enough ancilla qubits to label the manifold"
    psi = np.zeros(V.shape[0] * 2 ** n_anc, complex)
    M = psi.reshape(2 ** n_anc, V.shape[0])      # row = ancilla config
    for k in range(d):
        M[k] = V[:, k] / np.sqrt(d)
    return psi


def lemma_b_family(specs, deg_tol=1e-8):
    """
    Executable check of the general Lemma B over a FAMILY of clusters.
    `specs` = [(label, n, edges), ...].  For each: solve the exact ground
    manifold (dimension d), build budget_state, and verify
      Tr(rho H) = E0,  purity = 1/d,  S = log2 d,  rank = d,
    and that leaving the manifold costs energy: mixing in a first-excited
    column raises Tr(rho H) strictly above E0.  Returns one record per
    cluster.  Used to show the budget is log2(dim manifold) for d = 2, 4,
    6, 8, ... — not just the doublet.
    """
    out = []
    for label, n, edges in specs:
        H = K.heisenberg_hamiltonian(n, edges).to_matrix()
        w, v = np.linalg.eigh(H)
        d = int(np.sum(np.abs(w - w[0]) < deg_tol * max(1.0, abs(w[0]))))
        V = v[:, :d]
        n_anc = int(np.ceil(np.log2(d)))
        psi = budget_state(V, n_anc)
        M = psi.reshape(2 ** n_anc, 2 ** n)
        rho = M.conj().T @ M               # small only for small n
        E = float(np.real(np.trace(rho @ H)))
        lam = np.linalg.eigvalsh(M @ M.conj().T).clip(min=0)
        lam = lam[lam > 1e-14]
        # rank d+1 comparison: swap one manifold column for the first excited
        Vx = V.copy()
        Vx[:, -1] = v[:, d]
        psix = budget_state(Vx, n_anc)
        Mx = psix.reshape(2 ** n_anc, 2 ** n)
        Ex = float(np.real(np.trace((Mx.conj().T @ Mx) @ H)))
        out.append(dict(label=label, n=n, d=d, E0=float(w[0]), E_budget=E,
                        purity=float((lam ** 2).sum()),
                        entropy=float(-(lam * np.log2(lam)).sum()),
                        log2d=float(np.log2(d)),
                        rank=int(len(lam)), E_rank_excess=Ex,
                        gap=float(w[d] - w[0])))
    return out


def bell_channel_data(V, n, pair, P_target=None):
    """
    THEOREM P, general form.  For any cluster with ground manifold V
    (2^n x D) and any dummy pair, returns per Bell outcome b:
      p      = tr(K_b P K_b^dag) / D              (uniform prior on P)
      weight = tr(P_target K_b P K_b^dag) / tr(K_b P K_b^dag)
      gram   = K_b P K_b^dag restricted to the manifold, trace-normalized
    all basis-invariant (functions of the projector P = V V^dag alone).
    The Gram matrices are the correctability data: branch b is repairable
    by a b-conditioned unitary/isometry on the survivors IFF its normalized
    Gram equals the singlet one; `gram_distance` reports the deviation.
    """
    D = V.shape[1]
    out = {}
    for name in BELL_BRAS:
        cols = []
        for k in range(D):
            v, p, _ = project_pairs(V[:, k], n, {tuple(pair): name})
            cols.append(v * np.sqrt(p))
        Kv = np.stack(cols, axis=1)               # 2^(n-2) x D
        tr = float(np.real(np.vdot(Kv, Kv)))
        G = Kv.conj().T @ Kv
        rec = dict(p=tr / D, gram=(G / np.trace(G).real if tr > 1e-14
                                   else G))
        if P_target is not None and tr > 1e-14:
            rec["weight"] = float(np.real(np.trace(
                Kv.conj().T @ (P_target @ Kv)))) / tr
        out[name] = rec
    return out


def gram_distance(data, ref="singlet"):
    """||Ghat_b - Ghat_ref||_F for each outcome of `bell_channel_data`:
    0 iff branch b is correctable to the reference branch."""
    Gr = data[ref]["gram"]
    return {k: float(np.linalg.norm(v["gram"] - Gr)) for k, v in data.items()}


def su2_equivariance_residual(n, pair, seed=0):
    """
    The singlet branch is SU(2)-EQUIVARIANT: K_s U^{(x)n} = U^{(x)(n-2)} K_s
    for every global rotation U (the singlet bra is the unique SU(2)
    invariant of two spins).  Returns the operator-norm residual for a
    random SU(2) element — the structural reason the singlet channel, and
    only it, is Task-4's contraction map R.
    """
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=3)
    nrm = np.sqrt(a * a + b * b + c * c)
    th = 0.7
    Ug = (np.cos(th / 2) * np.eye(2)
          - 1j * np.sin(th / 2) * (a * _X + b * _Y + c * _Z) / nrm)

    def glob(U, m):
        out = np.array([[1.0 + 0j]])
        for _ in range(m):
            out = np.kron(out, U)
        return out

    keep = [s for s in range(n) if s not in pair]
    Ks = np.zeros((2 ** len(keep), 2 ** n), complex)
    B = BELL_BRAS["singlet"].conj()
    for m in range(2 ** n):
        bits = [(m >> q) & 1 for q in range(n)]
        amp = B[bits[pair[0]], bits[pair[1]]]
        if amp:
            Ks[sum(bits[s] << q for q, s in enumerate(keep)), m] += amp
    return float(np.linalg.norm(Ks @ glob(Ug, n) - glob(Ug, n - 2) @ Ks))


# --- 6b'. The open path: Lemma J and the entanglement-swap state ----------
_SINGLET2 = np.zeros(4, complex)
_SINGLET2[0b01], _SINGLET2[0b10] = 1 / np.sqrt(2), -1 / np.sqrt(2)


def swap_state(ins: Insertion, mixed_col=0):
    """
    The entanglement-swap candidate: singlets on (corner, apex), the exact
    mixed-chain ground on the fragment that carries the primed copies, and
    the 8 remaining black sites of the all-black fragment in the ground
    state of their induced H19 graph.  Projecting all three pairs performs
    three entanglement swaps (amplitude -1/2 each, p = 1/64 EXACTLY) and
    teleports the mixed chain onto the black lattice with copies -> corners:
    the survivor is [chain ground, virtual bonds ACTIVE] (x) [8-site ground],
    with the identity E_proj = -15 + E0(8-site piece) = the a3 bound (the
    two partitions are mirror images).  Teleportation alone relocates where
    the energy lives; it does not create any.
    """
    fdat = fragment_data(ins)
    _, Vm, _, _ = kd.ground_manifold_sz(11, fdat["mixed"]["edges"])
    rest = [s for s in ins.frag_black if s not in ins.triplet]
    loc8 = {s: i for i, s in enumerate(rest)}
    e8 = [(loc8[a], loc8[b]) for a, b in EDGES19 if a in loc8 and b in loc8]
    _, V8, _, deg8 = kd.ground_manifold_sz(len(rest), e8)
    factors = [(_SINGLET2, [c, ins.new[c][0]]) for c in sorted(ins.triplet)]
    factors += [(Vm[:, mixed_col], ins.frag_mixed), (V8[:, 0], rest)]
    return product_embed(ins.n, factors)


def decoupled_state(ins: Insertion, psi19):
    """
    |psi19> (x) |s>_(pair1) (x) |s>_(pair2) (x) |s>_(pair3): the 25-qubit
    state whose dummy pairs are already singlets and factorized.  Its k=3
    projection returns |psi19> EXACTLY with p = 1 (the Bell bra hits its own
    singlet), so with psi19 = the 19-site ground state it reads 0.000% in
    BOTH readings — while its <H25> error is 12.48%.  This is the exact
    counterexample showing that minimizing <H25> is MISALIGNED with the
    black objective, and simultaneously the Lemma-B corollary made
    concrete (pure + exact <=> dummies inert).
    """
    return product_embed(ins.n,
                         [(psi19, list(range(19)))]
                         + [(_SINGLET2, list(p)) for p in ins.pairs])


def lemma_j_check(gates, mode="pair"):
    """
    LEMMA J (junction inertness under projection), checked on one dummy
    triangle with legs (c, a, p) = (corner, apex, primed copy).  `gates` =
    [((leg, leg), theta), ...] with legs from 'cap' — an arbitrary
    SU(2)-preserving circuit U on the triangle.  The k=3 projection
    contracts <s|_(a,p); if the START has two triangle legs in a mutual
    singlet, Schur's lemma forces the contracted circuit to act on the
    remaining spin-1/2 as a SCALAR (Hom_SU(2)(1/2,1/2) is 1-dim):

      mode='pair':  start |s>_(a,p)  ->  <s|U|s> : C^2_c -> C^2_c = lam*1
      mode='swap':  start |s>_(c,a)  ->  <s|_(a,p) U |s>_(c,a) :
                    C^2_p -> C^2_c = lam * (teleport map)

    Returns (lam, residual) with residual = ||M - lam*ref||: the survivor
    of ANY junction circuit from such starts is invariant, only the branch
    probability changes, by prod_triangles |lam|^2.  (For a single gate on
    (c,a) or (c,p) in 'pair' mode, |lam|^2 = (5+3cos theta)/8; products of
    non-commuting gates need the full contraction, hence this helper.)
    """
    q = {"c": 0, "a": 1, "p": 2}
    U = np.eye(8, dtype=complex)
    for (l1, l2), th in gates:
        g = K.heis_matrix(th)
        G3 = np.zeros((8, 8), complex)       # legs (c,a,p), c most significant
        i1, i2 = q[l1], q[l2]
        for r in range(8):
            br = [(r >> (2 - k)) & 1 for k in range(3)]
            for cix in range(8):
                bc = [(cix >> (2 - k)) & 1 for k in range(3)]
                if all(br[k] == bc[k] for k in range(3) if k not in (i1, i2)):
                    G3[r, cix] = g[br[i1] * 2 + br[i2], bc[i1] * 2 + bc[i2]]
        U = G3 @ U
    U6 = U.reshape((2,) * 6)                 # (C,A,P, c,a,p), out then in
    s2 = _SINGLET2.reshape(2, 2)
    if mode == "pair":
        # start |s>_(a,p): map c_in -> c_out;  ref = identity
        M = np.einsum("AP,CAPcap,ap->Cc", s2.conj(), U6, s2)
    else:
        # start |s>_(c,a), free leg p: teleport map p_in -> c_out;
        # at U = 1 it equals (1/2)*identity (whence p = 1/64 for 3 pairs)
        M = np.einsum("AP,CAPcap,ca->Cp", s2.conj(), U6, s2)
    lam = complex(np.trace(M) / 2)
    return lam, float(np.linalg.norm(M - lam * np.eye(2)))


# --- 6c. Exact (rational) arithmetic for the Bell table -------------------
def _rref_nullspace(rows, ncol):
    """Nullspace basis of an integer/Fraction matrix over Q (list of rows),
    by exact Gauss-Jordan.  Returns a list of Fraction basis vectors."""
    from fractions import Fraction
    M = [[Fraction(x) for x in r] for r in rows]
    piv, r = [], 0
    for c in range(ncol):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    free = [c for c in range(ncol) if c not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * ncol
        v[f] = Fraction(1)
        for i, c in enumerate(piv):
            v[c] = -M[i][f]
        basis.append(v)
    return basis


def _rational_projector(n, edges, E0, sectors):
    """
    EXACT projector (Fraction entries, 2^n x 2^n) onto the ground manifold,
    assembled from the S^z sectors listed in `sectors`.  Legitimate because
    H has INTEGER entries in the computational basis and E0 is an integer
    here, so ker(H - E0) is a RATIONAL subspace: the manifold projector is
    a rational matrix and every trace built from it is a rational number,
    with no floating point anywhere.
    """
    from fractions import Fraction
    P = [[Fraction(0)] * (2 ** n) for _ in range(2 ** n)]
    for sz in sectors:
        basis = [int(b) for b in kd.sz_sector_basis(n, sz)]
        idx = {b: i for i, b in enumerate(basis)}
        m = len(basis)
        A = [[0] * m for _ in range(m)]
        for k, bstate in enumerate(basis):
            diag = 0
            for (i, j) in edges:
                bi, bj = (bstate >> i) & 1, (bstate >> j) & 1
                if bi == bj:
                    diag += 1
                else:
                    diag -= 1
                    A[idx[bstate ^ (1 << i) ^ (1 << j)]][k] += 2
            A[k][k] += diag - E0                       # H - E0 on the sector
        ns = _rref_nullspace([list(r) for r in zip(*A)], m)   # rows of A^T
        if not ns:
            continue
        # orthogonal projector B (B^T B)^{-1} B^T over Q
        BtB = [[sum(u[k] * v[k] for k in range(m)) for v in ns] for u in ns]
        d = len(ns)
        aug = [row[:] + [Fraction(int(i == j)) for j in range(d)]
               for i, row in enumerate(BtB)]
        for c in range(d):                              # Gauss-Jordan invert
            p = next(i for i in range(c, d) if aug[i][c] != 0)
            aug[c], aug[p] = aug[p], aug[c]
            inv = Fraction(1) / aug[c][c]
            aug[c] = [x * inv for x in aug[c]]
            for i in range(d):
                if i != c and aug[i][c] != 0:
                    f = aug[i][c]
                    aug[i] = [x - f * y for x, y in zip(aug[i], aug[c])]
        Ginv = [row[d:] for row in aug]
        C = [[sum(Ginv[a][b] * ns[b][k] for b in range(d)) for k in range(m)]
             for a in range(d)]                          # (B^TB)^{-1} B^T
        for a in range(d):
            for k in range(m):
                if ns[a][k] == 0:
                    continue
                gk = basis[k]
                for l in range(m):
                    if C[a][l] != 0:
                        P[gk][basis[l]] += ns[a][k] * C[a][l]
    return P


def exact_bell_table(pair=None):
    """
    THEOREM P's numbers in EXACT RATIONAL ARITHMETIC — no eigensolver, no
    rounding.  Builds the rational ground-manifold projectors of the 7-site
    modified and 5-site original Task-4 clusters (integer H, integer E0 =>
    rational kernel), applies the integer-scaled Bell bras, and returns
    Fractions:  p_b = tr(K_b P K_b^T)/(s_b * tr P),
                w_b = tr(P_orig K_b P K_b^T)/tr(K_b P K_b^T),
    with s_b = 2 for the (anti)symmetric bras (the 1/sqrt2 squared) and 1
    for |00>,|11>.  Returns {name: (p, weight)} as Fraction pairs.
    """
    from fractions import Fraction
    pair = kd.DUMMY_PAIR if pair is None else pair
    a, b = pair
    Pm = _rational_projector(kd.MOD_N, kd.MOD_EDGES, -9, (0.5, -0.5))
    Po = _rational_projector(kd.ORIG_N, kd.ORIG_EDGES, -6, (0.5, -0.5))
    keep = [s for s in range(kd.MOD_N) if s not in pair]
    # integer-scaled bras: (bits_a, bits_b) -> coefficient
    BRAS = {"singlet": ({(0, 1): 1, (1, 0): -1}, 2),
            "triplet0": ({(0, 1): 1, (1, 0): 1}, 2),
            "triplet+": ({(0, 0): 1}, 1),
            "triplet-": ({(1, 1): 1}, 1)}
    trP = sum(Pm[i][i] for i in range(2 ** kd.MOD_N))
    out = {}
    for name, (bra, scale) in BRAS.items():
        # K rows: 5-qubit index; columns: 7-qubit index
        cols = {}                                    # out_idx -> [(m, coef)]
        for m in range(2 ** kd.MOD_N):
            bits = [(m >> q) & 1 for q in range(kd.MOD_N)]
            co = bra.get((bits[a], bits[b]))
            if co:
                o = sum(bits[s] << q for q, s in enumerate(keep))
                cols.setdefault(o, []).append((m, co))
        # M = K P K^T  (5-qubit space), exact
        M = {}
        for o1, l1 in cols.items():
            for o2, l2 in cols.items():
                v = sum(c1 * c2 * Pm[m1][m2]
                        for m1, c1 in l1 for m2, c2 in l2)
                if v:
                    M[(o1, o2)] = v
        trM = sum(v for (o1, o2), v in M.items() if o1 == o2)
        p = Fraction(trM, 1) / (scale * trP)
        w = (sum(Po[o2][o1] * v for (o1, o2), v in M.items()) / trM
             if trM else None)
        out[name] = (p, w)
    return out


# ==========================================================================
# 7. Persistence (kd.save_records / kd.load_records are reused as-is)
# ==========================================================================
def save_trade(points, path):
    """Persist a trade curve (list of flat dicts) as one npz."""
    kd.save_records([{k: (str(v) if isinstance(v, tuple) else v)
                      for k, v in p.items()} for p in points], path)


def load_trade(path):
    """Inverse of save_trade (subset comes back as its str repr)."""
    return kd.load_records(path)


# ==========================================================================
# 8. Smoke test
# ==========================================================================
def smoke(verbose=True, heavy=True):
    """
    Re-asserts every headline number of the Task-5 prelims (2026-08-07).
    `heavy=True` adds the 25-qubit statevector self-test (~1 GB transient).
    """
    say = print if verbose else (lambda *a, **k: None)
    ok = True

    trips = splitting_triplets()
    assert trips == SPLITTING_TRIPLETS, trips
    say(f"[1/7] splitting triplets = {trips}  (canonical {CANONICAL})")

    ins = insert_dummies()          # all invariants asserted inside
    assert ins.pairs == [(19, 20), (21, 22), (23, 24)]
    assert ins.hosts == {6: (4, 5, 6), 11: (1, 2, 11), 16: (16, 17, 18)}
    assert ins.frag_black == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    say(f"[2/7] canonical 25-site lattice: 39 bonds, 13 triangles, "
        f"virtual {ins.virtual}, 2T-N=+1, fragments ~ delta_chain(5)")

    a1 = best_cover_baseline(ins)
    assert (a1["n_matchings"], a1["best_black"]) == (72, 9)
    assert a1["distribution"] == {9: 17, 8: 22, 7: 25, 6: 8}
    say(f"[3/7] (a1) 72 maximum matchings, distribution "
        f"{a1['distribution']}, best <H19> = {a1['energy']:.4f}")

    a3 = naive_dnc_bound(ins)
    assert abs(a3["energy"] + 27.665016) < 5e-6 and a3["deg_rest"] == 1
    a2 = chain_product_baseline(ins, n_starts=6, iters=40, seed=0)
    assert abs(a2["energy_nocross"] + 24.461865) < 1e-6
    assert a2["energy"] < -24.47 and abs(a2["entropy"] - 2.55) < 0.05
    say(f"[4/7] (a3) = {a3['energy']:.6f}; (a2) no-cross = "
        f"{a2['energy_nocross']:.6f}, with polarization = "
        f"{a2['energy']:.4f} (purity {a2['purity']:.3f}, "
        f"S = {a2['entropy']:.4f} bits, <S2> = {a2['s2_black']:.4f})")

    tab = miniature_bell_table()
    assert abs(tab["singlet"]["p"] - 9 / 16) < 1e-12
    assert abs(tab["singlet"]["weight"] - 1.0) < 1e-10
    for t in ("triplet0", "triplet+", "triplet-"):
        assert abs(tab[t]["p"] - 7 / 48) < 1e-12
        assert abs(tab[t]["weight"] - 13 / 21) < 1e-10
    # cross-check the singlet channel against Task-4's contraction map
    _, Vm, _ = kd.ground_manifold(7, kd.MOD_EDGES)
    R = kd.contraction_map()
    v, p, _ = project_pairs(Vm[:, 0], 7, {tuple(kd.DUMMY_PAIR): "singlet"})
    assert np.allclose(v * np.sqrt(p), R @ Vm[:, 0], atol=1e-12)
    say(f"[5/7] miniature Bell table: p = 9/16 & 3 x 7/48, weights 1 & "
        f"3 x 13/21 = {13 / 21:.6f}; singlet channel == kd.contraction_map")

    # 5b. the same numbers in EXACT rational arithmetic, plus the general
    # forms of the two theorems and the cut identity
    from fractions import Fraction
    ex = exact_bell_table()
    assert ex["singlet"] == (Fraction(9, 16), Fraction(1))
    assert all(ex[t] == (Fraction(7, 48), Fraction(13, 21))
               for t in ("triplet0", "triplet+", "triplet-"))
    assert sum(p for p, _ in ex.values()) == 1
    assert su2_equivariance_residual(7, kd.DUMMY_PAIR) < 1e-12
    fam = lemma_b_family([(f"chain{m}",) + kd.delta_chain(m)[:2]
                          for m in (1, 2, 3)])
    assert all(abs(r["entropy"] - r["log2d"]) < 1e-12
               and abs(r["purity"] - 1 / r["d"]) < 1e-12
               and r["E_rank_excess"] > r["E0"] + 1e-6 for r in fam)
    E_bond = bond_energies19(load_doublet()[1][:, 0])
    assert abs(sum(E_bond.values()) - E19_REF) < 1e-5
    cuts = {t: cut_accounting(insert_dummies(t), E_bond)
            for t in SPLITTING_TRIPLETS}
    assert all(cut_identity_residual(insert_dummies(t)) < 1e-9
               for t in SPLITTING_TRIPLETS)
    mc = [abs(c["mean_cross"]) for c in cuts.values()]   # magnitudes
    assert max(mc) / min(mc) < 1.02          # 'cheap bonds' refuted
    assert {c["n_cross"] for c in cuts.values() if c["err_pct"] < 1} == {4}
    assert abs(cuts[(2, 11, 12)]["a3"] + 28.9474) < 5e-4
    G0 = nx.Graph(insert_dummies().edges)
    assert all(nx.is_isomorphic(nx.Graph(insert_dummies(t).edges), G0)
               for t in SPLITTING_TRIPLETS)
    # Lemma J: Schur scalars (one-gate closed form; any circuit -> scalar)
    l1, r1 = lemma_j_check([(("c", "a"), 0.7)], mode="pair")
    assert r1 < 1e-12 and abs(abs(l1) ** 2 - (5 + 3 * np.cos(0.7)) / 8) < 1e-12
    _, r5 = lemma_j_check([(("c", "a"), .5), (("a", "p"), -.9),
                           (("c", "p"), 1.1), (("c", "a"), -.3)], mode="pair")
    ls, rs = lemma_j_check([], mode="swap")
    assert r5 < 1e-12 and rs < 1e-12 and abs(ls + 0.5) < 1e-12
    say(f"[5b/7] exact rationals reproduced with Fractions; Lemma B general "
        f"(d = {[r['d'] for r in fam]} -> S = "
        f"{[round(r['entropy'], 3) for r in fam]} bits); cut identity exact "
        f"for all five; mean_C spread "
        f"{100 * (max(mc) / min(mc) - 1):.1f}% (count, not cost); the five "
        f"25-site lattices are isomorphic")

    if heavy:
        cover = [tuple(d) for d in a1["matching"]]
        assert {tuple(sorted(p)) for p in ins.pairs} <= \
            {tuple(sorted(d)) for d in cover}
        psi = er.dimer_cover_state(25, cover)
        m = black_moments(psi, 25)
        assert abs(m["energy"] + 27) < 1e-9 and abs(m["purity"] - 1) < 1e-9
        assert m["rank"] == 1 and abs(m["s2_black"] - 0.75) < 1e-9
        phi, prob, kept = project_pairs(psi, 25,
                                        {p: "singlet" for p in ins.pairs})
        assert abs(prob - 1) < 1e-9 and kept == list(range(19))
        assert abs(h19_energy(phi, 19) + 27) < 1e-9
        say(f"[6/7] 25q self-test: best cover <H19> = {m['energy']:.4f}, "
            f"pure (purity {m['purity']:.3f}, rank {m['rank']}), <S2> = "
            f"{m['s2_black']:.4f}; k=3 projection: prob = {prob:.4f}, "
            f"black state on qubits 0..18 with <H19> = "
            f"{h19_energy(phi, 19):.4f}")
        # embed the (a2) optimum and re-measure it with the generic
        # 25-qubit machinery: energies and diagnostics must agree
        psi25 = embed_fragments(ins, a2["psi_black"], a2["psi_mixed"])
        m2 = black_moments(psi25, 25)
        assert abs(m2["energy"] - a2["energy"]) < 1e-8
        assert abs(m2["purity"] - a2["purity"]) < 1e-8
        assert abs(m2["s2_black"] - a2["s2_black"]) < 1e-8
        pt0 = trade_point(psi25, (), ins.pairs)
        assert (abs(pt0["energy"] - a2["energy"]) < 1e-8
                and abs(pt0["prob"] - 1) < 1e-12)
        say(f"[7/7] embed_fragments((a2) optimum): 25q machinery reproduces "
            f"<H19> = {m2['energy']:.4f}, purity {m2['purity']:.3f}, "
            f"<S2> = {m2['s2_black']:.4f}; trade_point(k=0) agrees")
        # the swap identity: E_proj = a3 exactly, at p = 1/64
        psi_sw = swap_state(ins)
        pt_sw = trade_point(psi_sw, (0, 1, 2), ins.pairs)
        a3e = naive_dnc_bound(ins)["energy"]
        assert abs(pt_sw["prob"] - 1 / 64) < 1e-9
        assert abs(pt_sw["energy"] - a3e) < 1e-6
        say(f"[7b/7] swap identity: k=3 projection gives E = "
            f"{pt_sw['energy']:.6f} = a3 ({a3e:.6f}) at p = "
            f"{pt_sw['prob']:.6f} = 1/64 EXACTLY (teleportation relocates "
            f"energy, never creates it)")
        # Prop. 9: <H25> is a misaligned proxy for the black readout
        E19r, V19r = load_doublet()
        psi_perp = decoupled_state(ins, V19r[:, 0])
        pt_p = trade_point(psi_perp, (0, 1, 2), ins.pairs, V19=V19r)
        Tp = psi_perp.reshape((2,) * 25)
        h25p = float(sum(np.real(dc.bond_vdot(25, Tp, Tp, _BOND, a_, b_))
                         for a_, b_ in ins.edges))
        assert abs(pt_p["prob"] - 1.0) < 1e-9
        assert abs(pt_p["energy"] - E19r) < 1e-6
        assert abs(h25p + 33.380308) < 1e-5
        say(f"[7c/7] Prop. 9: |g19> x |s>^3 projects to E19 EXACTLY at p = 1 "
            f"({pt_p['energy']:.6f}) while <H25> = {h25p:.4f} sits 12.5% "
            f"above E25 -> minimizing <H25> is MISALIGNED with the black "
            f"objective")
    else:
        say("[6/7][7/7] skipped (heavy=False)")
    say("smoke: ALL PASS")
    return ok


if __name__ == "__main__":
    smoke(verbose=True, heavy=("--light" not in sys.argv))
