"""Shared test helpers: stub samplers and random nodes."""

import itertools
from collections.abc import Sequence

import numpy as np

from qcbb import Hamiltonian, SampleSet, collect
from qcbb.labs import NodeHamiltonian
from qcbb.labs.energy import autocorrelations


class WorstSampler:
    """Adversarial sampler: always returns the all-(+1) assignment."""

    def sample(
        self, problems: Sequence[Hamiltonian], shots: int, seed: int | None = None
    ) -> list[SampleSet]:
        return [collect(h, np.ones((1, h.size), dtype=np.int8)) for h in problems]


class EmptySampler:
    """Returns no samples at all."""

    def sample(
        self, problems: Sequence[Hamiltonian], shots: int, seed: int | None = None
    ) -> list[SampleSet]:
        return [collect(h, np.empty((0, h.size), dtype=np.int8)) for h in problems]


def random_node(
    n: int, rng: np.random.Generator, fixed: int | None = None
) -> np.ndarray:
    """Return a partial sequence with `fixed` random spins (a random count if None)."""
    spins = np.zeros(n, dtype=np.int8)
    if fixed is None:
        fixed = int(rng.integers(0, n + 1))
    pos = rng.choice(n, size=fixed, replace=False)
    spins[pos] = rng.choice(np.array([1, -1], dtype=np.int8), size=fixed)
    return spins


def node(n: int, fixed: dict[int, int]) -> NodeHamiltonian:
    """Return the node Hamiltonian of length n with the given spins fixed."""
    spins = np.zeros(n, dtype=np.int8)
    for i, v in fixed.items():
        spins[i] = v
    return NodeHamiltonian(spins)


def all_sequences(n: int) -> np.ndarray:
    """Return all 2^n sequences, shape (2^n, n)."""
    return np.array(list(itertools.product((1, -1), repeat=n)), dtype=np.int8)


def completions(node: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all completions of a node and their autocorrelations."""
    h = NodeHamiltonian(node)
    x = h.complete(all_sequences(h.size).reshape(2**h.size, h.size))
    return x, autocorrelations(x)
