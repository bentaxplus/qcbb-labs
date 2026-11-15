"""The angle schedule of the QAOA sampler for LABS (paper, Sec. 3.1).

Shaydulin et al. (Sci. Adv. 10, eadm6761, 2024) found fixed QAOA angles for
the LABS master Hamiltonian that transfer across lengths: gamma_l = g_l / n,
beta_l = b_l, with (g, b) the same for all n at each depth p = 1..33. The
vectors (g, b) are in data/qaoa_parameters.json, taken from QOKit
(Apache-2.0).

A node Hamiltonian has 1- and 3-spin terms that the master does not have,
and no transfer rule is known for it. We rescale the published angles:

    gamma_l = lambda g_l / n,   beta_l = b_l,

with n the sequence length (not the qubit count F). lambda = 1 for the
master, the published rule. For node Hamiltonians lambda = 0.875 at p = 1
and 0.625 at larger p; for the skew-symmetric restriction lambda = 0.5. These
constants are empirical, chosen on class roots whose ground state is the
global optimum, and then frozen. The proof does not depend on them.
"""

import json
from functools import cache
from importlib import resources

import numpy as np

from qcbb.labs.hamiltonian import NodeHamiltonian
from qcbb.labs.skew import SkewHamiltonian
from qcbb.problem import Hamiltonian
from qcbb.qaoa.backend import Floats

MASTER_SCALE = 1.0
SKEW_SCALE = 0.5


def node_scale(p: int) -> float:
    """Return the angle scale lambda of node Hamiltonians at depth p."""
    return 0.875 if p == 1 else 0.625


@cache
def fixed_angles(p: int) -> tuple[Floats, Floats]:
    """Return the length-free angles (g, b) of depth p, 1 <= p <= 33."""
    path = resources.files("qcbb.labs") / "data" / "qaoa_parameters.json"
    table = json.loads(path.read_text())["p"]
    if str(p) not in table:
        raise ValueError(f"no fixed angles for p = {p}")
    row = table[str(p)]
    return np.array(row["gamma"]), np.array(row["beta"])


class LabsAngleSchedule:
    """The fixed angles rescaled, gamma_l = lambda g_l / n, beta_l = b_l, for LABS.

    Args:
        p: Number of QAOA layers.
        scale: The angle scale lambda; if None, `MASTER_SCALE` for the master,
            `SKEW_SCALE` for the skew-symmetric restriction and `node_scale(p)`
            otherwise.
    """

    def __init__(self, p: int = 12, scale: float | None = None) -> None:
        self.p = p
        self.scale = scale
        self.gamma, self.beta = fixed_angles(p)

    def __call__(self, hamiltonian: Hamiltonian) -> tuple[Floats, Floats]:
        """Return (gammas, betas) for a LABS node or skew-symmetric Hamiltonian.

        Raises:
            TypeError: If the Hamiltonian is not a LABS Hamiltonian.
        """
        if not isinstance(hamiltonian, NodeHamiltonian | SkewHamiltonian):
            raise TypeError("LabsAngleSchedule needs a LABS Hamiltonian")
        n = hamiltonian.n
        if self.scale is not None:
            scale = self.scale
        elif isinstance(hamiltonian, SkewHamiltonian):
            scale = SKEW_SCALE
        elif hamiltonian.size == n:
            scale = MASTER_SCALE
        else:
            scale = node_scale(self.p)
        return scale * self.gamma / n, self.beta
