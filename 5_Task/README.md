# Task 5 — The Trace-Projection Trade: Dummy-Triangle Embeddings of the 19-Site KAFH Cluster

Task 5 embeds three dummy triangles (six dummy qubits, 19+6 = 25) into the 19-site
kagome cluster of Tasks 1–3 and asks for a spin-liquid-like state on the black
(original) part, "tracing the dummies out at the last step". The Task-4 theorems turn that request into a sharp, quantitative dichotomy — the deliverable of this task:

- *Trace* the dummies (the literal request): the black state is deterministic but
  generically mixed. Exactness Tr(ρH₁₉) = E₁₉ confines supp(ρ) to the ground manifold, so rank(ρ) ≤ d and S(black|blue) ≤ log₂d bits — 1 bit here, since the 19-site manifold is a doublet (Lemma B, proved for general d and verified at d = 2,4,6,8). Pure *and* exact forces the dummies inert.
- **Project** each dummy pair (apex, primed copy) onto the singlet (Task-4 Theorem 4): the black state is pure and exact — but probabilistic, and the failed (triplet) branches are provably uncorrectable (Gram criterion). On the 7-site miniature, in *exact rational arithmetic*: p(singlet) = 9/16 with image weight 1; each triplet channel p = 7/48 with weight 13/21.
- Projecting k = 0..3 of the three pairs and tracing the rest interpolates: the
  trade curve (success probability vs purity vs ⟨S²⟩ vs Tr(ρH₁₉) vs doublet
  weight) is the central measurement. On the exact ground state of H₂₅ the two
  readings of the *same* state differ by 16×: 16.3% traced (4.45 bits spent) versus 1.0% projected (doublet weight 0.907, p = 0.153) — the latter 3.9× better than Task-2's partition (3.93%) and just short of Task-3's best (0.938%), which needed 84 optimized parameters where this needs none.
- The obstruction in one line: the target needs **3.15 bits** of entanglement across the very cut the dummy triangles occupy, every bit of dummy mediation is a bit of black–blue correlation, and an exact traced readout may spend at most 1 bit of it. Projection escapes the bound by paying in probability instead: its survivor is pure and carries 2.59 of those 3.15 bits.
- And the embedding gives no *preparation* advantage: junction circuits act on the
  projected survivor as a scalar (Lemma J), teleporting exact fragments returns the product bound a₃ exactly at p = 1/64 (swap identity), and minimizing ⟨H₂₅⟩ is anti-ordered with the black readout — |g₁₉⟩⊗|s⟩³ reads 0.000% in both readings at p = 1 while sitting 12.48% above E₂₅ (Prop. 9). Since the projection is onto, min E_proj = E₁₉: the projected problem *is* the original one.
- By-products worth reusing: the five admissible embeddings are the *same graph*
  (Prop. 7), so choosing an insertion is choosing a *readout*, not a Hamiltonian; and the insertion scan doubles as a *partition finder* for the original 19-site problem. Two of the five induce a cut with 4 crossing bonds instead of 6, and their zero-parameter product state reaches *0.682%* (a₃ = −28.9474), five times better than the partition Task 2 engineered by hand. The cut identity
  a₃ = E₀(H₁₉ ∖ cut) = E₁₉ − Σ_C⟨h_ij⟩_g + Δ holds to machine zero for all five
  (Prop. 8), and the mechanism is the *number* of crossing bonds, not their strength: the mean crossing-bond energy is equal across insertions to 1.4%.

## Contents

- [`Trade_Duality_KAFH.ipynb`](Trade_Duality_KAFH.ipynb) — the narrative notebook
  (33 cells): the assignment, the verdict and a section roadmap; Prop. 1–2 (five
  insertion triplets, canonical (6,11,16); 2T − N = +1, 0/3¹³ covers, six virtual bonds); Prop. 3 and the depth-0 floors (product ceiling −27; Theorem-5 trap −24.4619/−24.4858; naive DnC −27.665016); Lemma B in general log₂d form with the budget-saturating state; Theorem P in
  general SU(2)-covariant form, its exact rational corollary and the Gram criterion; ed25; the trade curve (Fig. 3, central); Prop. 8, the cut identity
  a₃ = E₀(H₁₉ ∖ cut) and the design rule; Prop. 7, the five embeddings are one
  graph, so the insertion is a readout choice; the equal-parameter head-to-head;
  Lemma J, the swap identity and Prop. 9, which close all three routes to a
  *preparable* good state — junction circuits act on the projected survivor as a scalar, fragment teleportation returns the product bound exactly, and minimizing ⟨H₂₅⟩ is anti-ordered with the black readout; the 1-bit-vs-3.15-bit discussion; summary with the five-task arc, the (now much narrower) open question, and a reconciliation table; **Prop. 10** and the advisor's product objective measured (the pair bond *is* the projection probability, P_s = (1 − ⟨h_pair⟩)/4, so the triangle energy is the wrong knob and E_△ = −9 is degenerate between p = 1 and p = 1/64, while the λ-scan of the repaired linear form is monotone but costs ≈ 0.004 of probability per point of energy with zero doublet weight throughout); and a 43-item referee cell.
- [`kagome_trade.py`](kagome_trade.py) — the Task-5 engine (imports Tasks 1–4):   insertion geometry and the exhaustive triplet scan, depth-0 baselines,
  cut accounting, matrix-free black observables on 25-qubit states (⟨H₁₉⟩,
  Gram-spectrum purity and entropy, collective ⟨S²_black⟩, doublet weight), Bell-pair projections and the trade curve, the general forms of both theorems
  (`lemma_b_family`, `bell_channel_data`, `gram_distance`,
  `su2_equivariance_residual`), exact rational Bell arithmetic (`exact_bell_table`, rational kernel projectors over ℚ), an S^z-sector Lanczos solver for H₂₅ (dim 5,200,300, matrix-free matvec), product embeddings, the open-path tools (`swap_state`, `lemma_j_check` — the per-triangle Schur scalar of any SU(2)-preserving junction circuit), and a smoke test (`python kagome_trade.py`) that re-asserts every headline number.
- `results/*.npz` — persisted artifacts (auto-loaded by the notebook): `ed25.npz`,
  `ed25_2_11_12.npz`, `trade_*.npz`, `vqe25_junction_*.npz`, `vqe25_h25_*.npz`,
  `open_path_probes.npz`, `h25_readings.npz`, `product_objective.npz`,
  `lambda_scan.npz`.
- `figures/` — the seven notebook figures (project palette, greyscale-safe markers).
- [`email_draft_ahsan.md`](email_draft_ahsan.md), [`email_reply_ahsan.md`](email_reply_ahsan.md)
  — the correspondence that drove §11: the summary sent to the advisor, and the reply to
  his counter-proposal (minimise the product of the two energies) with the measured
  λ-front.

## Conventions

Same as Tasks 1–4: H = Σ (XX+YY+ZZ) per bond (singlet bond = −3, J = 1 uncalibrated); E₁₉ = −29.146168 (doublet). Qubit layout: black sites keep their Task-1 labels 0..18; dummy pair k = (apex, copy) = (19+2k, 20+2k), corners sorted — ρ_black is always the trace over the six top qubits, and is never formed explicitly (the view M = ψ.reshape(2^6, 2^19) carries its full spectrum).
