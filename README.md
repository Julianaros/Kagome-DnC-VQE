# Qintern 2026: Variational Quantum Eigensolvers for the Kagome Antiferromagnet

Research internship project (advisor: Prof. Muhammad Ahsan) on the Kagome
antiferromagnetic Heisenberg model (KAFH), building on the paper
*"Utility-scale Experimental Quantum Computation with Hardware Efficient
Ansätze and Calibrated Hamiltonian"*.

| Folder | Contents |
|---|---|
| `1_Task/` | **PoC**: Heisenberg-gate (eSWAP) HVA layers reach the benchmark energy *without* Hamiltonian calibration, preserving SU(2) by construction. 19 sites: 0.52% error, ⟨S²⟩ = 0.75 exact. |
| `2_Task/` | **Divide-and-conquer VQE** (paper Sec. 4 / Fig. 4) with SU(2)-preserving junction gates: local VQEs + junction-only global VQE, H_SEL validation (paper Fig. 6(a) analogue), error attribution with controls, and the 26-site extension via Clebsch–Gordan sector-aware recombination. Measured floor: 1.64% with 24 junction parameters. |
| `3_Task/` | **Entanglement Recovery**: the advisor's fragment-dressing brief (HVA dressing, NNN training terms) tested against a registered scorecard and refuted with mechanism, SU(2)-protected stationarity theorem, Majumdar–Ghosh stability window of the training Hamiltonian, exchange rates below break-even, then turned into the project's best divide-and-conquer result by investing the capacity at the interface instead: *plaquette* junction layers reach 0.938% @ 84 interface parameters with both locals frozen (monolithic: 0.80% @ 120), scaling to 26 sites with 28× the junction gain (0.015% vs the exact naive bound) and ⟨S²⟩ = 0 audited at 2²⁶. |
| `4_Task/` | **Dummy-triangle insertion, proof by construction**: inserting a triangle at the shared corner of (0,1,2)–(0,22,24) splits site 0 into 0 and 0'. Both claims proved mutual exclusion of the 0 / 0' dimers is *equivalent* to the existence of a valid cover (7 of 27), and that cover is the witness saturating E₀ = −9 with P H_△ P = −3P on all three mutually non-commuting triangles, so the bond energies of (0,1,2) and (0',22,24) reconstruct E₀(original) = −6 exactly on every ground state. "0' is a variational copy of 0" is made precise twice: an exact exchange symmetry, and a local SU(2)-equivariant singlet contraction mapping the 8-dim manifold onto the 6-dim one, whose kernel is a closed-form kink doublet. Generalizes to Δ-chains (E₀ = −3n) and rings; a counting obstruction (2T ≤ N) says why it cannot defrustrate the kagome bulk. |
| `5_Task/` | **The trace–projection trade**: three dummy triangles (6 dummy qubits, 25 sites) embedded in the 19-site cluster, and the advisor's "trace them out at the last step" taken literally, where it meets a principled obstruction. *Lemma B*: exactness confines ρ_black to the ground manifold, so a traced readout carries at most log₂ d bits of black–blue entanglement, one bit for a doublet, and pure *and* exact forces the dummies inert. *Theorem P*: the singlet branch is the SU(2)-equivariant Task-4 contraction, the triplet branches are uncorrectable by a Gram criterion, and p = 9/16, 3 × 7/48 with image weights 1, 3 × 13/21 come out of exact ℚ arithmetic. Read both ways, the *same* exact 25-site ground state gives 16.3% traced (4.45 bits spent, mixed) and 1.015% projected (pure, doublet weight 0.907, p = 0.153); the k = 0..3 trade curve measures the exchange rate. Preparation gets no free lunch: junction circuits act on the projected survivor as a Schur scalar (Lemma J), teleporting exact fragments returns the product bound at p = 1/64, and minimising ⟨H₂₅⟩ is anti-ordered with the black readout (Prop. 9). By-products: the five embeddings are one graph, so the insertion is a *readout* choice (Prop. 7), and the scan doubles as a partition finder, a 4-bond cut of the original lattice reaching 0.682% with zero parameters. |

The through-line: SU(2)-exact gates on the *uncalibrated* Hamiltonian are enough,
first monolithically (Task 1), then split-and-rejoined (Task 2), and with the
divide-and-conquer error budget located and paid where it lives, at the interface
(Task 3). Task 4 changes register from numerics to proof: on corner-sharing
clusters the same dimer picture that drives the ansätze is provably exact, and the
counting that makes it exact is also what fails on the frustrated bulk. Task 5 turns
those proofs back on the algorithm and finds its boundary: enlarging the problem with
dummy qubits buys an exactly solvable preparation and a readout *theory*, not a better
spin liquid. What you do at the last step is the whole question, and the answer is a
budget: tracing is deterministic and capped at log₂ d bits of black–blue entanglement,
projecting is exact and pure but probabilistic, and the three routes to preparing a good
state through the dummies (junction circuits, fragment teleportation, optimising the
enlarged Hamiltonian) are each closed by a theorem or an exact counterexample.

See each folder's README for physics, results and run instructions
(`2_Task` imports the `1_Task` modules, `3_Task` imports both, `4_Task` imports all three and `5_Task` imports all four via relative paths; keep the folder layout as is).
