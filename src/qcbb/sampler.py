"""The sampler and its samples.

A sampler maps a node Hamiltonian on F free spins to q samples: distinct
assignments with counts (paper, Sec. 3.1). The samples enter QCBB only as
incumbent candidates, with their energies recomputed exactly, and as the
conflict values that choose the branching variable. They never enter a bound
or a prune, so any sampler that returns +-1 assignments gives a correct
result; `collect` enforces that contract.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qcbb.problem import Hamiltonian, Ints, Spins


@dataclass(frozen=True, eq=False)
class SampleSet:
    """Distinct samples of one node Hamiltonian.

    Attributes:
        spins: Free-variable values +1 or -1, shape (m, F), sorted by energy.
        counts: How often each row was drawn, shape (m,).
        energies: Exact energies of the rows, shape (m,).
    """

    spins: Spins
    counts: Ints
    energies: Ints

    @property
    def best(self) -> int:
        """Lowest energy in the set."""
        return int(self.energies[0])


def collect(hamiltonian: Hamiltonian, spins: Spins) -> SampleSet:
    """Merge raw samples into a sample set with exact energies.

    Args:
        hamiltonian: The node Hamiltonian that was sampled.
        spins: Raw free-variable values, shape (shots, F).

    Returns:
        Distinct rows with counts and exact energies, lowest energy first.

    Raises:
        ValueError: If the shape is not (shots, F) or a value is not +1 or -1.
    """
    raw = np.asarray(spins)
    if raw.ndim != 2 or raw.shape[1] != hamiltonian.size:
        raise ValueError(f"samples must have shape (shots, {hamiltonian.size})")
    if not np.isin(raw, (-1, 1)).all():
        raise ValueError("samples must take the values +1 and -1")
    rows, counts = np.unique(raw.astype(np.int8), axis=0, return_counts=True)
    e = hamiltonian.energies(rows)
    order = np.argsort(e, kind="stable")
    return SampleSet(rows[order], counts[order].astype(np.int64), e[order])


class Sampler(Protocol):
    """Samples low-energy states of node Hamiltonians."""

    def sample(
        self,
        problems: Sequence[Hamiltonian],
        shots: int,
        seed: int | None = None,
    ) -> list[SampleSet]:
        """Sample each node Hamiltonian.

        Args:
            problems: Node Hamiltonians to sample.
            shots: Number of samples per Hamiltonian.
            seed: Seed for reproducible runs.

        Returns:
            One sample set per Hamiltonian, in the same order.
        """
        ...
