"""Symmetry of the LABS energy, the class roots and the reversal rule.

Negation (s -> -s), alternation (s_i -> (-1)^(i-1) s_i) and reversal
(s_i -> s_{n+1-i}) leave E unchanged and generate a group of order 8 (paper,
symmetry reduction theorem). Every orbit has exactly one element with
s_1 = s_2 = +1 in each of the two cosets {eta^a alpha^b s} and
{eta^a alpha^b rho s}; the lexicographic minimum of the two (with +1 before
-1) is the canonical representative.

The search fixes s_1 = s_2 = +1, and the reversal rule drops the class roots
that cannot hold a canonical representative.
"""

import itertools

import numpy as np

from qcbb.problem import Spins


def kappa(s: Spins) -> Spins:
    """Return the element of {eta^a alpha^b s} with s_1 = s_2 = +1.

    Args:
        s: Sequence or sequence prefix of length >= 2.

    Returns:
        kappa(s), a copy.
    """
    t: Spins = s * s[0]
    if t[1] < 0:
        t = t * np.where(np.arange(t.size) % 2 == 0, 1, -1).astype(s.dtype)
    return t


def canonical(s: Spins) -> Spins:
    """Return the canonical representative of the orbit of s.

    Args:
        s: Full sequence.

    Returns:
        lexmin(kappa(s), kappa(rho s)) with +1 ordered before -1.
    """
    a, b = kappa(s), kappa(s[::-1])
    return a if _lex_le(a, b) else b


def class_roots(n: int, m: int) -> Spins:
    """Return the class roots: all end blocks of length m with s_1 = s_2 = +1.

    Args:
        n: Sequence length.
        m: Number of fixed spins at each end, 2 <= m <= n // 2.

    Returns:
        Partial sequences (0 = free), one row per root, shape (2^(2m-2), n).
    """
    rows = []
    for spins in itertools.product((1, -1), repeat=2 * m - 2):
        s = np.zeros(n, dtype=np.int8)
        s[:2] = 1
        s[2:m] = spins[: m - 2]
        s[n - m :] = spins[m - 2 :]
        rows.append(s)
    return np.array(rows, dtype=np.int8)


def passes_reversal_rule(spins: Spins, m: int) -> bool:
    """Return False if no completion of the node can be canonical (reversal rule).

    With s_1 = s_2 = +1 and the first and last m spins fixed, the first m
    entries p' of kappa(rho s) depend only on the suffix. The node is dropped
    if p' is lexicographically smaller than the prefix p.

    Args:
        spins: Partial sequence with s_1 = s_2 = +1 and both ends of length m fixed.
        m: Length of the fixed prefix and suffix, m >= 2.

    Returns:
        True if the node may hold a canonical sequence.
    """
    p = spins[:m]
    p_rev = kappa(spins[::-1][:m])
    return _lex_le(p, p_rev)


def _lex_le(a: Spins, b: Spins) -> bool:
    """Return a <= b in lexicographic order with +1 before -1."""
    diff = np.flatnonzero(a != b)
    return diff.size == 0 or a[diff[0]] > b[diff[0]]
