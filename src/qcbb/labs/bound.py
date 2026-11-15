"""The bound, the lag budgets and propagation for LABS (paper, appendices).

Ranges. The lag-k products split into free chains, maximal runs of free
products along a chain r, r + k, r + 2k, ... Free chains share no free spin,
so the range R_k of C_k at a node is the arithmetic progression

    R_k = {lo_k, lo_k + g_k, ..., hi_k},

with step g_k = 4 if both ends of every free chain are fixed, 2 otherwise, and
g_k = 0 if lag k has no free product. mu_k = min |R_k|.

Pair congruence. For k < n/2, C_k + C_{n-k} = n (mod 4), so the pair contributes
at least m_k = min {a^2 + b^2 : a in R_k, b in R_{n-k}, a + b = n (mod 4)},
and

    LB = sum_{k < n/2} m_k + [n even] mu_{n/2}^2 <= E(s)

for every completion s.

Lag budgets and propagation. Every completion with E <= T has C_k^2 <= b_k
= T - LB_{-k}, where LB_{-k} = LB - m_k + mu_{n-k}^2 (middle lag:
LB - mu_k^2). If the only element of R_k inside the budget is lo_k (or
hi_k), every free chain takes its minimum (maximum); the free chains whose extreme
has a unique assignment are forced. Forcing is repeated until no spin changes.

All arithmetic is in integers. Arrays of lags store lag k at index k.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from qcbb.problem import Ints, Spins

Stats = NDArray[np.int64]
"""Per-lag statistics, one row (lo_k, hi_k, g_k, mu_k, m_k) per lag."""


@njit(cache=True)
def lag_range(s: Ints, k: int) -> tuple[int, int, int]:
    """Return (lo, hi, g), the range R_k of C_k.

    Args:
        s: Partial sequence, 0 for free spins, shape (n,).
        k: Lag, 1 <= k <= n - 1.

    Returns:
        Smallest element, largest element and step of R_k.
    """
    n = s.size
    c = 0
    lo = 0
    hi = 0
    step4 = True
    free = False
    for r in range(k):
        last = (n - 1 - r) // k
        start = 0
        sa = s[r]
        a = sa
        for t in range(1, last + 1):
            b = s[r + t * k]
            if a != 0 and b != 0:
                c += a * b
                start = t
                sa = b
            elif b != 0 or t == last:
                e = t - start
                free = True
                if sa != 0 and b != 0:
                    hi += e if sa * b == 1 else e - 2
                    lo += -e if 1 - 2 * (e % 2) == sa * b else 2 - e
                else:
                    step4 = False
                    lo -= e
                    hi += e
                start = t
                sa = b
            a = b
    g = 0 if not free else (4 if step4 else 2)
    return c + lo, c + hi, g


@njit(cache=True)
def min_abs(lo: int, hi: int, g: int) -> int:
    """Return mu = min |v| over the progression {lo, lo + g, ..., hi}."""
    if lo >= 0:
        return lo
    if hi <= 0:
        return -hi
    x = lo % g
    return min(x, g - x)


@njit(cache=True)
def residue_min(lo: int, hi: int, g: int, rho: int) -> int:
    """Return min |v| over the elements v = rho (mod 4) of a progression, or -1."""
    a = lo
    if (a - rho) % 4 != 0:
        if g != 2 or lo + 2 > hi or (lo + 2 - rho) % 4 != 0:
            return -1
        a = lo + 2
    b = hi if (hi - rho) % 4 == 0 else hi - 2
    return min_abs(a, b, 4)


@njit(cache=True)
def pair_min(stats: Stats, k: int, n: int) -> int:
    """Return m_k, the pair congruence minimum of the lags k and n - k."""
    j = n - k
    best = -1
    for rho in range(4):
        a = residue_min(stats[k, 0], stats[k, 1], stats[k, 2], rho)
        if a < 0:
            continue
        b = residue_min(stats[j, 0], stats[j, 1], stats[j, 2], (n - rho) % 4)
        if b < 0:
            continue
        v = a * a + b * b
        if best < 0 or v < best:
            best = v
    return best


@njit(cache=True)
def node_bound(s: Ints, stats: Stats, threshold: int) -> int:
    """Return the pair congruence bound LB and fill the per-lag statistics.

    stats[k] = (lo_k, hi_k, g_k, mu_k, m_k), with m_k stored at both lags of a
    pair and mu_k^2 at the middle lag. The scan stops early, with a partial
    sum above the threshold, once LB > T is certain.

    Args:
        s: Partial sequence, 0 for free spins, shape (n,).
        stats: Work array, shape (n, 5).
        threshold: T; pass a large value for the full bound.

    Returns:
        LB, or a partial sum > T.
    """
    n = s.size
    total = 0
    for k in range(1, (n + 1) // 2):
        for j in (k, n - k):
            lo, hi, g = lag_range(s, j)
            stats[j, 0] = lo
            stats[j, 1] = hi
            stats[j, 2] = g
            stats[j, 3] = min_abs(lo, hi, g)
        m = pair_min(stats, k, n)
        stats[k, 4] = m
        stats[n - k, 4] = m
        total += m
        if total > threshold:
            return total
    if n % 2 == 0:
        k = n // 2
        lo, hi, g = lag_range(s, k)
        mu = min_abs(lo, hi, g)
        stats[k, 0] = lo
        stats[k, 1] = hi
        stats[k, 2] = g
        stats[k, 3] = mu
        stats[k, 4] = mu * mu
        total += mu * mu
    return total


@njit(cache=True)
def budget(stats: Stats, k: int, n: int, threshold: int, lb: int) -> int:
    """Return the lag budget b_k = T - LB_{-k}."""
    if 2 * k == n:
        return int(threshold - lb + stats[k, 4])
    mu = stats[n - k, 3]
    return int(threshold - lb + stats[k, 4] - mu * mu)


@njit(cache=True)
def force(s: Ints, k: int, sign: int) -> int:
    """Fix the spins of every free chain of lag k whose products must all equal `sign`.

    A free chain with one fixed end is fixed from that end; one with two
    fixed ends only if s_first s_last = sign^e; one with no fixed end is
    left free (propagation theorem, claim 3).

    Args:
        s: Partial sequence, changed in place.
        k: Lag.
        sign: -1 (every free chain at its minimum) or +1 (at its maximum).

    Returns:
        The number of spins fixed.
    """
    n = s.size
    fixed = 0
    for r in range(k):
        last = (n - 1 - r) // k
        start = 0
        for t in range(1, last + 1):
            a = s[r + (t - 1) * k]
            b = s[r + t * k]
            if a != 0 and b != 0:
                start = t
                continue
            if b == 0 and t < last:
                continue
            e = t - start
            sa = s[r + start * k]
            if sa != 0 and (b == 0 or sa * b == (sign if e % 2 else 1)):
                v = sa
                for i in range(start + 1, t if b != 0 else t + 1):
                    v *= sign
                    s[r + i * k] = v
                    fixed += 1
            elif sa == 0 and b != 0:
                v = b
                for i in range(t - 1, start - 1, -1):
                    v *= sign
                    s[r + i * k] = v
                    fixed += 1
            start = t
    return fixed


@njit(cache=True)
def _propagate(s: Ints, threshold: int, stats: Stats) -> bool:
    """Force spins by the lag budgets until no spin changes.

    Repeats: bound; LB > T means no completion has E <= T; else force the
    first lag whose budget admits a single extreme of R_k, and start again.

    Args:
        s: Partial sequence, changed in place.
        threshold: T.
        stats: Work array, shape (n, 5); holds the final node's statistics.

    Returns:
        False if no completion has E <= T, else True.
    """
    n = s.size
    while True:
        lb = node_bound(s, stats, threshold)
        if lb > threshold:
            return False
        fixed = 0
        for k in range(1, n):
            lo, hi, g = stats[k, 0], stats[k, 1], stats[k, 2]
            if lo == hi:
                continue
            b = budget(stats, k, n, threshold, lb)
            if lo * lo <= b and (lo + g) * (lo + g) > b:
                fixed = force(s, k, -1)
            elif hi * hi <= b and (hi - g) * (hi - g) > b:
                fixed = force(s, k, 1)
            if fixed:
                break
        if not fixed:
            return True


def lag_ranges(spins: Spins) -> Stats:
    """Return the ranges R_k of all lags.

    Args:
        spins: Partial sequence, 0 for free spins, shape (n,).

    Returns:
        Rows (lo_k, hi_k, g_k) with lag k at index k - 1, shape (n - 1, 3).
    """
    s = spins.astype(np.int64)
    return np.array([lag_range(s, k) for k in range(1, s.size)], dtype=np.int64)


def statistics(spins: Spins) -> tuple[int, Stats]:
    """Return LB and the per-lag statistics (lo, hi, g, mu, m) of a node."""
    stats = np.zeros((spins.size, 5), dtype=np.int64)
    lb = node_bound(spins.astype(np.int64), stats, np.iinfo(np.int64).max)
    return int(lb), stats


def lower_bound(spins: Spins) -> int:
    """Return the pair congruence bound LB, a lower bound on E over all completions.

    Args:
        spins: Partial sequence, 0 for free spins, shape (n,).

    Returns:
        The integer lower bound.
    """
    return statistics(spins)[0]


def budgets(spins: Spins, threshold: int) -> Ints:
    """Return the lag budgets b_k = T - LB_{-k}.

    Every completion with E <= T has C_k^2 <= b_k for all k.

    Args:
        spins: Partial sequence, 0 for free spins, shape (n,).
        threshold: T = UB - Delta.

    Returns:
        b with lag k at index k - 1, shape (n - 1,).
    """
    lb, stats = statistics(spins)
    n = spins.size
    return np.array(
        [budget(stats, k, n, threshold, lb) for k in range(1, n)], dtype=np.int64
    )


def propagate(spins: Spins, threshold: int) -> Spins | None:
    """Return the node with all forced spins fixed, or None if E <= T is impossible.

    Args:
        spins: Partial sequence, 0 for free spins, shape (n,).
        threshold: T = UB - Delta.

    Returns:
        A partial sequence whose completions include every completion of the
        node with E <= T, or None if there is no such completion.
    """
    s = spins.astype(np.int64)
    stats = np.zeros((s.size, 5), dtype=np.int64)
    return s.astype(np.int8) if _propagate(s, threshold, stats) else None
