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

The through-line of the three tasks: SU(2)-exact gates on the *uncalibrated*
Hamiltonian are enough, first monolithically (Task 1), then split-and-rejoined
(Task 2), and finally with the divide-and-conquer error budget located and paid
where it lives, at the interface (Task 3).

See each folder's README for physics, results and run instructions
(`2_Task` imports the `1_Task` modules and `3_Task` imports both via relative
paths; keep the folder layout as is).
