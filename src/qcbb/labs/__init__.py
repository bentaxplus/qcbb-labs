"""Low autocorrelation binary sequences (LABS), QOBLIB problem 02.

Minimize E(s) = sum_{k=1}^{n-1} C_k^2 with C_k = sum_i s_i s_{i+k} over
s in {-1, +1}^n. `LabsProblem` is the QCBB problem; `AnnealingSampler` is
the SA backend and `LabsAngleSchedule` the angle schedule of the QAOA backend.
"""

from qcbb.labs.angles import LabsAngleSchedule
from qcbb.labs.annealing import AnnealingSampler
from qcbb.labs.energy import energy, lattice_step
from qcbb.labs.hamiltonian import NodeHamiltonian
from qcbb.labs.problem import LabsProblem
from qcbb.labs.skew import SkewHamiltonian, SkewSampler

__all__ = [
    "AnnealingSampler",
    "LabsAngleSchedule",
    "LabsProblem",
    "NodeHamiltonian",
    "SkewHamiltonian",
    "SkewSampler",
    "energy",
    "lattice_step",
]
