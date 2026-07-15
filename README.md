# Qintern 2026: Variational Quantum Eigensolvers for the Kagome Antiferromagnet

Research internship project (advisor: Prof. Muhammad Ahsan) on the Kagome
antiferromagnetic Heisenberg model (KAFH), building on the paper
*"Utility-scale Experimental Quantum Computation with Hardware Efficient
Ansätze and Calibrated Hamiltonian"*.

| Folder | Contents |
|---|---|
| `1_Task/` | **PoC**: Heisenberg-gate (eSWAP) HVA layers reach the benchmark energy *without* Hamiltonian calibration, preserving SU(2) by construction. 19 sites: 0.52% error, ⟨S²⟩ = 0.75 exact. |
| `2_Task/` | **Divide-and-conquer VQE** (paper Sec. 4 / Fig. 4) with SU(2)-preserving junction gates: local VQEs + junction-only global VQE, H_SEL validation (paper Fig. 6(a) analogue), error attribution with controls, and the 26-site extension via Clebsch–Gordan sector-aware recombination. |

See each folder's README for physics, results and run instructions
(`2_Task` imports the `1_Task` modules via a relative path; keep the
folder layout as is).
