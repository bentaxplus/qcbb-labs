"""The energy, its lattice, the QOBLIB table and the node Hamiltonian."""

import numpy as np
import pytest
from helpers import all_sequences, random_node

from qcbb.labs import NodeHamiltonian, instances, lattice_step
from qcbb.labs.energy import autocorrelations, energies, energy


def test_autocorrelations_and_energy_by_definition() -> None:
    rng = np.random.default_rng(0)
    s = rng.choice(np.array([1, -1], dtype=np.int8), size=17)
    c = [sum(int(s[i]) * int(s[i + k]) for i in range(17 - k)) for k in range(1, 17)]
    assert autocorrelations(s).tolist() == c
    assert energy(s) == sum(x * x for x in c)
    assert autocorrelations(s[None]).tolist() == [c]


@pytest.mark.parametrize("n", range(3, 15))
def test_energies_lie_on_the_lattice_and_their_minimum_is_the_qoblib_optimum(
    n: int,
) -> None:
    e = energies(all_sequences(n))
    assert set(e % 4) == {(n // 2) % 4}  # energy residue
    assert len(set(e % lattice_step(n))) == 1  # lattice step
    assert e.min() == instances.get(n).energy


def test_qoblib_instance_table() -> None:
    table = instances.load()
    assert len(table) == 99
    assert sum(i.proven for i in table.values()) == 65
    assert instances.get("labs030").energy == 59


def evaluate(constant: int, terms: dict, z: np.ndarray) -> np.ndarray:
    total = np.full(z.shape[0], constant, dtype=np.int64)
    for mono, w in terms.items():
        total += w * np.prod(z[:, list(mono)], axis=1, dtype=np.int64)
    return total


def test_node_hamiltonian_monomials_give_the_energy() -> None:
    for seed in range(5):
        rng = np.random.default_rng(seed)
        h = NodeHamiltonian(random_node(13, rng, fixed=5))
        constant, terms = h.monomials()
        z = rng.choice(np.array([1, -1], dtype=np.int8), size=(50, h.size))
        assert np.array_equal(evaluate(constant, terms, z), h.energies(z))
        assert max(len(m) for m in terms) <= 4


def test_master_has_only_2_and_4_spin_terms() -> None:
    _, terms = NodeHamiltonian.master(10).monomials()
    assert {len(m) for m in terms} == {2, 4}


def test_complete_inserts_the_free_spins() -> None:
    h = NodeHamiltonian(np.array([1, 0, 0, 0, 0, -1], dtype=np.int8))
    assert h.size == 4
    s = h.complete(np.array([1, -1, 1, -1], dtype=np.int8))
    assert s.tolist() == [1, 1, -1, 1, -1, -1]
