"""The LABS energy and its lattice.

A sequence is s in {-1, +1}^n. The autocorrelation at lag k is
C_k = sum_{i=1}^{n-k} s_i s_{i+k}, and the energy is E = sum_{k=1}^{n-1} C_k^2.
Arrays store C_k at index k - 1.
"""

import numpy as np

from qcbb.problem import Ints, Spins


def autocorrelations(s: Spins) -> Ints:
    """Return C_1, ..., C_{n-1} of one or many sequences.

    Args:
        s: Sequences of +1 and -1, shape (n,) or (m, n).

    Returns:
        C_k at index k - 1 of the last axis, shape (n - 1,) or (m, n - 1).
    """
    x = np.asarray(s, dtype=np.int64)
    n = x.shape[-1]
    c = np.empty((*x.shape[:-1], n - 1), dtype=np.int64)
    for k in range(1, n):
        c[..., k - 1] = (x[..., : n - k] * x[..., k:]).sum(axis=-1)
    return c


def energy(s: Spins) -> int:
    """Return the exact energy E(s) of one sequence, shape (n,)."""
    c = autocorrelations(s)
    return int(c @ c)


def energies(s: Spins) -> Ints:
    """Return the exact energies of many sequences, shape (m, n) -> (m,)."""
    c = autocorrelations(s)
    e: Ints = np.einsum("ij,ij->i", c, c)
    return e


def lattice_step(n: int) -> int:
    """Return the lattice step Delta of the energies of length n (paper, lattice lemma).

    All energies of length n differ by multiples of Delta: 8 if n = 2 (mod 4),
    else 4. So a node with LB > UB - Delta holds no sequence with E < UB.
    """
    return 8 if n % 4 == 2 else 4
