"""The sampler contract and the SA backend."""

import numpy as np
import pytest
from helpers import node

from qcbb import collect
from qcbb.labs import AnnealingSampler, NodeHamiltonian, instances
from qcbb.labs.annealing import _flip
from qcbb.labs.energy import autocorrelations, energy


def test_collect_enforces_the_sampler_contract() -> None:
    h = NodeHamiltonian.master(5)
    with pytest.raises(ValueError):
        collect(h, np.zeros((2, 5), dtype=np.int8))
    with pytest.raises(ValueError):
        collect(h, np.ones((2, 4), dtype=np.int8))
    assert collect(h, np.empty((0, 5), dtype=np.int8)).spins.shape == (0, 5)


def test_flip_delta_equals_the_energy_difference() -> None:
    rng = np.random.default_rng(4)
    for _ in range(200):
        n = int(rng.integers(2, 25))
        s = rng.choice(np.array([1, -1], dtype=np.int8), size=n)
        j = int(rng.integers(0, n))
        c = autocorrelations(s)
        before = energy(s)
        de = _flip(s, c, j, False)
        assert np.array_equal(c, autocorrelations(s)) and before == energy(s)
        assert _flip(s, c, j, True) == de
        assert energy(s) == before + de
        assert np.array_equal(c, autocorrelations(s))


def test_annealing_finds_the_optimum_of_the_master() -> None:
    n = 20
    result = AnnealingSampler(sweeps=500).sample(
        [NodeHamiltonian.master(n)], shots=32, seed=1
    )[0]
    assert result.best == instances.get(n).energy


def test_annealing_keeps_fixed_spins_and_is_reproducible() -> None:
    h = node(12, {0: -1, 11: 1})
    a = AnnealingSampler(sweeps=50).sample([h], shots=8, seed=2)[0]
    b = AnnealingSampler(sweeps=50).sample([h], shots=8, seed=2)[0]
    seqs = h.complete(a.spins)
    assert np.all(seqs[:, 0] == -1) and np.all(seqs[:, 11] == 1)
    assert np.array_equal(h.energies(a.spins), a.energies)
    assert np.array_equal(a.spins, b.spins) and np.array_equal(a.counts, b.counts)
