"""The fixed-angle QAOA sampler and the backends that execute its circuits.

The QAOA sampler measures the depth-p QAOA state of a node Hamiltonian in the
Z basis (paper, Sec. 3.1). An angle schedule gives the angles of each
Hamiltonian, a `QaoaJob` describes the circuit independently of the backend
(the Ising terms and the angles), and a `Backend` executes jobs: a quantum
device or the exact noiseless emulation. Each backend compiles a job to its
own gates.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from qcbb.problem import Hamiltonian, Monomial, Spins
from qcbb.sampler import SampleSet, SizeError, collect

Floats = NDArray[np.float64]

log = logging.getLogger("qcbb.qaoa")


class AngleSchedule(Protocol):
    """QAOA angles for a node Hamiltonian."""

    def __call__(self, hamiltonian: Hamiltonian) -> tuple[Floats, Floats]:
        """Return (gammas, betas), one pair per layer."""
        ...


@dataclass(frozen=True)
class QaoaJob:
    """One QAOA circuit: Ising terms on F qubits and the angles of its p layers.

    The circuit is exp(-i beta_l sum_j X_j) exp(-i gamma_l sum_S w_S Z_S / 2)
    for l = 1..p, applied to |+>^F; a measured 1 on qubit j means z_j = -1.

    Attributes:
        size: Number of qubits F.
        terms: Monomials S with integer weights w_S (constant term dropped).
        gammas: Phase angles, one per layer.
        betas: Mixer angles, one per layer.
    """

    size: int
    terms: tuple[tuple[Monomial, int], ...]
    gammas: tuple[float, ...]
    betas: tuple[float, ...]

    @classmethod
    def of(cls, hamiltonian: Hamiltonian, gammas: Floats, betas: Floats) -> "QaoaJob":
        """Return the job of a Hamiltonian with the given angles."""
        terms = tuple(sorted(hamiltonian.monomials()[1].items()))
        return cls(
            hamiltonian.size,
            terms,
            tuple(float(g) for g in gammas),
            tuple(float(b) for b in betas),
        )


@dataclass(frozen=True)
class Estimate:
    """Resources of a batch of compiled circuits, computed before running them.

    Attributes:
        circuits: Number of circuits (after packing).
        qubits: Largest number of physical qubits in one circuit.
        two_qubit_gates: Two-qubit gates over all circuits.
        depth: Largest circuit depth.
        fidelity: Smallest estimated circuit fidelity, the product of
            (1 - error) over gates and measurements.
        shots: Shots per circuit.
        cost: Cost in the provider's own unit (Quantinuum HQC), or None.
    """

    circuits: int
    qubits: int
    two_qubit_gates: int
    depth: int
    fidelity: float
    shots: int
    cost: float | None = None


class Backend(Protocol):
    """Executes QAOA jobs and returns the measured spins."""

    max_size: int

    def run(self, jobs: Sequence[QaoaJob], shots: int, seed: int | None) -> list[Spins]:
        """Return the measured spins of each job, shape (accepted shots, F).

        Error detection may reject shots, so a job can return fewer than
        `shots` rows, or none.
        """
        ...


class QaoaSampler:
    """Sample node Hamiltonians with fixed-angle QAOA on a backend.

    Args:
        angles: The angle schedule of the Hamiltonians.
        backend: Executes the circuits; the exact noiseless emulation if None.
    """

    def __init__(self, angles: AngleSchedule, backend: Backend | None = None) -> None:
        if backend is None:
            from qcbb.qaoa.emulator import Emulator

            backend = Emulator()
        self.angles = angles
        self.backend = backend

    def sample(
        self,
        problems: Sequence[Hamiltonian],
        shots: int,
        seed: int | None = None,
    ) -> list[SampleSet]:
        """Run one batch of QAOA circuits, one per node Hamiltonian.

        Args:
            problems: Node Hamiltonians to sample.
            shots: Number of measurements per Hamiltonian.
            seed: Seed for reproducible runs.

        Returns:
            One sample set per Hamiltonian, with exact energies.

        Raises:
            SizeError: If a Hamiltonian has more than `backend.max_size` qubits.
        """
        for h in problems:
            if h.size > self.backend.max_size:
                raise SizeError(
                    f"{h.size} qubits; the backend takes {self.backend.max_size}"
                )
        jobs = [QaoaJob.of(h, *self.angles(h)) for h in problems]
        spins = self.backend.run(jobs, shots, seed) if jobs else []
        return [collect(h, s) for h, s in zip(problems, spins, strict=True)]
