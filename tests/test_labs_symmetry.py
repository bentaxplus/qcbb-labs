"""The symmetry group, the class roots and the reversal rule."""

import itertools

import numpy as np

from qcbb.labs import LabsProblem
from qcbb.labs.energy import energy
from qcbb.labs.symmetry import canonical, class_roots, kappa, passes_reversal_rule


def orbit(s: np.ndarray) -> list[np.ndarray]:
    """Return the eight images of s under negation, alternation and reversal."""
    alt = np.where(np.arange(s.size) % 2 == 0, 1, -1).astype(s.dtype)
    out = []
    for neg, alter, rev in itertools.product((1, -1), (False, True), (False, True)):
        t = s[::-1] if rev else s
        t = t * alt if alter else t
        out.append(neg * t)
    return out


def test_orbit_keeps_the_energy_and_has_one_canonical_element() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.choice(np.array([1, -1], dtype=np.int8), size=11)
        images = orbit(s)
        assert {energy(t) for t in images} == {energy(s)}
        assert len({canonical(t).tobytes() for t in images}) == 1
        assert kappa(s)[:2].tolist() == [1, 1]


def test_reversal_rule_keeps_every_canonical_sequence() -> None:
    n, m = 10, 3
    for spins in itertools.product((1, -1), repeat=n - 2):
        s = np.array((1, 1, *spins), dtype=np.int8)
        node = s.copy()
        node[m : n - m] = 0
        if np.array_equal(canonical(s), s):
            assert passes_reversal_rule(node, m)


def test_class_roots_and_reversal_rule_keep_2112_of_4096() -> None:
    roots = class_roots(12, 4)
    assert roots.shape == (64, 12)
    assert np.all(roots[:, :2] == 1)
    assert np.all(roots[:, 4:8] == 0)
    assert len(LabsProblem(20, boundary=7).class_roots()) == 2112
