"""Exact noiseless emulation of the QAOA circuit.

A node Hamiltonian on F qubits is diagonal in the computational basis. Basis
state x has spins z_i = (-1)^(bit i of x), so a monomial over the set S of
spins has the value (-1)^popcount(x & S) and

    E(x) = sum_S w_S (-1)^popcount(x & S).

E is the Hadamard transform of the coefficient vector w (indexed by the bit
mask of S), computed exactly in integers in F 2^F steps. QAOA with depth p
then prepares

    |psi> = prod_l exp(-i beta_l sum_j X_j) exp(-i gamma_l E / 2) |+>^F,

the convention of Shaydulin et al. (2024): exp(-i gamma w Z_S / 2) per term,
which is rzz(gamma w) for two spins, and rx(2 beta) per qubit. The samples
are drawn from the measurement distribution of |psi>. Below PARALLEL_SIZE
qubits the kernels run on one thread, since thread start-up would cost more
than the work.
"""

from collections.abc import Sequence

import numpy as np
from numba import get_num_threads, njit, prange
from numpy.typing import NDArray

from qcbb.problem import Hamiltonian, Ints, Monomial, Spins
from qcbb.qaoa.backend import Floats, QaoaJob

Amplitudes = NDArray[np.complex128]

PARALLEL_SIZE = 16


def diagonal(hamiltonian: Hamiltonian) -> Ints:
    """Return the exact energies of all 2^F basis states.

    Args:
        hamiltonian: Node Hamiltonian on F qubits.

    Returns:
        E(x) for x = 0, ..., 2^F - 1.
    """
    return _diagonal(*hamiltonian.monomials(), hamiltonian.size)


def qaoa_state(energies: Ints, size: int, gammas: Floats, betas: Floats) -> Amplitudes:
    """Return the QAOA state for the given diagonal and angles.

    Args:
        energies: Diagonal E(x) of the Hamiltonian, shape (2^F,).
        size: Number of qubits F.
        gammas: Phase angles, one per layer.
        betas: Mixer angles, one per layer.

    Returns:
        The state vector, shape (2^F,).
    """
    psi = np.full(energies.size, 2.0 ** (-size / 2), dtype=np.complex128)
    g = np.asarray(gammas, dtype=np.float64)
    b = np.asarray(betas, dtype=np.float64)
    if size < PARALLEL_SIZE:
        _evolve(psi, energies, size, g, b)
    else:
        _evolve_parallel(psi, energies, size, g, b, 4 * get_num_threads())
    return psi


def spins(states: Ints, size: int) -> Spins:
    """Return the spins z_i = (-1)^(bit i) of basis states, shape (m, F)."""
    b = (states[:, None] >> np.arange(size)) & 1
    return (1 - 2 * b).astype(np.int8)


class Emulator:
    """Exact noiseless emulation: compute the QAOA state and draw shots from it.

    The emulator counts what a quantum device would have to run: `circuits`
    (jobs run, one circuit each) and `shots` (measurements over all of them).

    Args:
        max_size: Largest number of qubits accepted (memory 32 * 2^F bytes).
    """

    def __init__(self, max_size: int = 26) -> None:
        self.max_size = max_size
        self.circuits = 0
        self.shots = 0

    def run(self, jobs: Sequence[QaoaJob], shots: int, seed: int | None) -> list[Spins]:
        """Return `shots` measured assignments of each QAOA state.

        Args:
            jobs: QAOA jobs.
            shots: Number of measurements per job.
            seed: Seed of the measurement draws.

        Returns:
            Spins of the measured states, shape (shots, F) per job.

        Raises:
            ValueError: If a job has more than `max_size` qubits.
        """
        rng = np.random.default_rng(seed)
        out = []
        for job in jobs:
            if job.size > self.max_size:
                raise ValueError(f"F = {job.size} exceeds max_size = {self.max_size}")
            self.circuits += 1
            self.shots += shots
            e = _diagonal(0, dict(job.terms), job.size)
            psi = qaoa_state(e, job.size, np.array(job.gammas), np.array(job.betas))
            cdf = np.cumsum(psi.real**2 + psi.imag**2)
            cdf /= cdf[-1]
            states = np.searchsorted(cdf, rng.random(shots), side="right")
            out.append(spins(np.minimum(states, cdf.size - 1), job.size))
        return out


def _diagonal(constant: int, terms: dict[Monomial, int], size: int) -> Ints:
    """Return E(x) as the Hadamard transform of the coefficient vector."""
    w = np.zeros(1 << size, dtype=np.int64)
    w[0] = constant
    for mono, coeff in terms.items():
        w[sum(1 << i for i in mono)] += coeff
    if size < PARALLEL_SIZE:
        _walsh_hadamard(w, size)
    else:
        _walsh_hadamard_parallel(w, size, 4 * get_num_threads())
    return w


@njit(cache=True)
def _pair(t: int, j: int) -> tuple[int, int]:
    """Return the t-th pair of basis states that differ only in bit j."""
    a = ((t >> j) << (j + 1)) | (t & ((1 << j) - 1))
    return a, a | (1 << j)


@njit(cache=True)
def _add_sub(w: Ints, j: int, start: int, stop: int) -> None:
    """Butterfly (u, v) -> (u + v, u - v) on bit j for pairs start..stop-1."""
    for t in range(start, stop):
        a, b = _pair(t, j)
        u, v = w[a], w[b]
        w[a] = u + v
        w[b] = u - v


@njit(cache=True)
def _phase(psi: Amplitudes, e: Ints, angle: float, start: int, stop: int) -> None:
    """Multiply amplitudes start..stop-1 by exp(-i angle E(x))."""
    for x in range(start, stop):
        t = angle * e[x]
        psi[x] *= complex(np.cos(t), -np.sin(t))


@njit(cache=True)
def _mix(psi: Amplitudes, j: int, beta: float, start: int, stop: int) -> None:
    """Apply exp(-i beta X_j) on the pairs start..stop-1 of bit j."""
    c, s = np.cos(beta), np.sin(beta)
    for t in range(start, stop):
        a, b = _pair(t, j)
        u, v = psi[a], psi[b]
        psi[a] = c * u - 1j * s * v
        psi[b] = c * v - 1j * s * u


@njit(cache=True)
def _walsh_hadamard(w: Ints, size: int) -> None:
    """Replace w by its Hadamard transform, in place."""
    for j in range(size):
        _add_sub(w, j, 0, w.size >> 1)


@njit(cache=True)
def _evolve(psi: Amplitudes, e: Ints, size: int, gammas: Floats, betas: Floats) -> None:
    """Apply the QAOA layers to psi, in place."""
    for layer in range(gammas.size):
        _phase(psi, e, 0.5 * gammas[layer], 0, psi.size)
        for j in range(size):
            _mix(psi, j, betas[layer], 0, psi.size >> 1)


@njit(parallel=True, cache=True)
def _walsh_hadamard_parallel(w: Ints, size: int, chunks: int) -> None:
    """Multithreaded `_walsh_hadamard`."""
    half = w.size >> 1
    for j in range(size):
        for c in prange(chunks):
            _add_sub(w, j, c * half // chunks, (c + 1) * half // chunks)


@njit(parallel=True, cache=True)
def _evolve_parallel(
    psi: Amplitudes, e: Ints, size: int, gammas: Floats, betas: Floats, chunks: int
) -> None:
    """Multithreaded `_evolve`."""
    m, half = psi.size, psi.size >> 1
    for layer in range(gammas.size):
        for c in prange(chunks):
            _phase(psi, e, 0.5 * gammas[layer], c * m // chunks, (c + 1) * m // chunks)
        for j in range(size):
            for c in prange(chunks):
                _mix(psi, j, betas[layer], c * half // chunks, (c + 1) * half // chunks)
