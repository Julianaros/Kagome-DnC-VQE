# Dummy-Triangle Insertion: Proof by Construction (Task 4)

## The brief

Advisor's assignment (toward the next article). Original network: two corner-sharing
triangles (0,1,2)–(0,22,24). Modified network: a **dummy triangle** inserted at the
shared corner — (0,1,2)–(0,13,0')–(0',22,24), site 13 new, 0' duplicating 0. Prove or
refute:

> (1) The optimal dimer arrangement guarantees that a dimer with spin 0 and a dimer
> with spin 0' are mutually exclusive.
> (2) The spin-liquid state and the gs_energy can be reconstructed accurately from
> the bond energies of (0,1,2) and (0',22,24) — at bottom, 0' is a *variational
> copy* of 0 and vice versa.

**Both claims are TRUE and proved by construction**, with (1) functioning as the
combinatorial lemma of (2). Everything is stated as explicit lemmas/theorems in the
notebook, each verified by executable asserts (dense/sector-exact diagonalization,
machine precision).

## The proof chain (7 sites, convention H = Σ (XX+YY+ZZ), singlet = −3)

- **Lemma 1 (singlet absorption).** S²_△ (Π_singlet ⊗ 1) = ¾ (Π_singlet ⊗ 1): a dimer
  on two sites of a triangle pins E_△ = −3 *identically*, however the third site
  entangles with the rest. This is the entire mechanism behind "three dimers, one per
  triangle".
- **Lemma 2 (= claim 1).** Three pairwise-disjoint dimers necessarily take one bond
  per triangle; exactly 7/27 selections are valid; every one satisfies the exclusion
  (pigeonhole: all three dummy bonds touch {0,0'}); conversely exclusion ⟹
  extendable (7 = 4·1 + 1·3): **exclusion ≡ solvability**.
- **Theorem 1 (= claim 2, energy).** E₀ = −9, degeneracy 8, and the manifold is
  frustration-free in the strong sense P H_△ P = −3P for A, B, C **despite
  max|[S²_A,S²_B]| = 2** (bound + cover witness + zero variance at the floor).
  Corollary: P(H_A+H_C)P = −6P — the bond energies of (0,1,2) and (0',22,24) alone
  reconstruct E₀(original) = −6 exactly on *every* modified ground state.
- **Proposition 3 (copy, symmetry half).** σ = (0 0')(1 22)(2 24) is an exact
  automorphism, [P_σ,H] = 0; manifold splits 4 even ⊕ 4 odd (original: 4 ⊕ 2);
  mirrored observables agree exactly on σ-eigenstates.
- **Theorem 4 (copy, map half).** The singlet contraction R = ⟨s|₁₃,₀′ is strictly
  local, SU(2)-equivariant, and maps the 8-dim manifold **onto** the 6-dim original
  one by *singlet teleportation on covers* (coefficient 1 when the dummy dimer is
  (13,0'), else ±½ with the spinon pulled back out of the dummy). Closed-form kernel:
  s(1,2) ⊗ s(22,24) ⊗ [t(13,0') ⊗ σ₀] — the doublet with the dummy pair in a
  *triplet*. Sector structure of rr†: 1 (×4) ⊕ ¼ (×2). So G₈ ≅ G₆ ⊕ one kink doublet:
  the extra doublet is the Δ-chain kink localized on the inserted triangle — the one
  piece with no original counterpart.
- **Theorem 4b.** No *local* insertion exists (empty nullspace of a linear constraint,
  smallest singular value 1/√2): deletion is local, insertion provably is not. Bond
  energies transport exactly sector-wise; sector-mixing states reweigh (up to ≈1.1
  per bond) while E_A + E_C = −6 never wavers.
- **Theorem 5 (chains).** The clusters are open Δ-(sawtooth) chains (Nakamura–Kubo /
  Sen–Shastry–Walstedt–Cava 1996). For n = 1..7 (asserted): E₀ = −3n, deg = 2(n+1),
  2n+1 covers spanning the full Sz = +½ sector. Each insertion: ΔE₀ = −3, one extra
  kink slot. Rings still saturate (deg 2, 2 covers): **coverability, not acyclicity,
  decides**.
- **Proposition 6 (limit).** The witness needs a disjoint one-dimer-per-triangle
  cover ⟹ 2T ≤ N. Kagome bulk: T = 2N/3 ⟹ 2T = 4N/3 > N, impossible at any size;
  19-site cluster: T = 10, 0/3¹⁰ covers, E₀ = −29.146 > −30 = −3T. The insertion
  preserves coverability; re-embedding into kagome destroys it.

## Repository layout

| Path | Contents |
|---|---|
| `Dummy_Triangle_KAFH.ipynb` | The full study: clusters, bound vs saturation, Lemmas 1–2, Theorem 1 + corollary, σ symmetry, the 6↔8 map (teleportation table, kernel, no-local-insertion), Δ-chain/ring scan, kagome obstruction, paper-ready summary (§9), referee cell |
| `kagome_delta.py` | Task-4 engine: Δ-chain/ring builders, dense + Sz-sector-exact ground manifolds, cover combinatorics, permutation symmetries, singlet-absorption check, contraction/lift maps, triplet-kink closed forms, chain scan, persistence, smoke test |
| `results/*.npz` | Manifolds, map structure, chain scan, kagome check (auto-loaded) |
| `figures/` | Clusters, the 7 covers, kernel doublet bond map, Δ-chain ledger |

Task-1/2/3 engines are imported from `../1_Task/`, `../2_Task/`, `../3_Task/` (never
copied); `kagome_delta.py` puts them on `sys.path` itself. `python kagome_delta.py`
runs the 8-assert smoke test.

Two solver gotchas are handled explicitly (see module docstrings): dense manifolds
avoid the Task-2 `dc.project_sz` garbage-column issue, and the chain scan avoids
`eigsh`'s silent failure to resolve high degeneracies (it returned deg 13 instead of
16 at n_tri = 7) by one dense diagonalization *inside the Sz = +½ sector*, with the
total degeneracy inferred rigorously from ⟨S²⟩ = ¾ per column (zero variance at the
spectrum floor).

## Requirements & running

Same environment as Tasks 1–3 (`qiskit>=2.0`, `scipy`, `numpy`, `matplotlib`).
Deterministic (fixed seeds). With `results/` populated (this repo state) the notebook
re-runs in ~1 minute; from scratch the only nontrivial cell is the chain scan (~1 min).
The paper-ready formal summary is §9 of the notebook.
