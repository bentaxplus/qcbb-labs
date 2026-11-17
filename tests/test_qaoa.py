"""The QAOA sampler: emulator, angle schedule, circuits, Iceberg, packing, devices."""

from collections.abc import Sequence

import numpy as np
import pytest
from helpers import node

from qcbb import Hamiltonian, Settings, SizeError, solve
from qcbb.labs import (
    LabsAngleSchedule,
    LabsProblem,
    NodeHamiltonian,
    SkewHamiltonian,
    SkewSampler,
    instances,
)
from qcbb.labs.energy import autocorrelations
from qcbb.qaoa import (
    Emulator,
    IbmDevice,
    QaoaJob,
    QaoaSampler,
    QuantinuumDevice,
    cnots_per_layer,
    device,
    diagonal,
    program,
    qaoa_circuit,
    qaoa_state,
    term_counts,
)
from qcbb.qaoa.emulator import spins


def all_states(h: Hamiltonian) -> np.ndarray:
    return spins(np.arange(1 << h.size), h.size)


def test_diagonal_equals_the_energies_of_all_basis_states() -> None:
    for h in (node(13, {0: 1, 4: -1, 9: 1}), SkewHamiltonian(15)):
        assert np.array_equal(diagonal(h), h.energies(all_states(h)))


@pytest.mark.parametrize(
    ("n", "p", "overlap"), [(12, 12, 0.1735586813), (16, 4, 0.0205712526)]
)
def test_ground_state_probability_matches_qokit(n: int, p: int, overlap: float) -> None:
    h = NodeHamiltonian.master(n)
    e = diagonal(h)
    psi = qaoa_state(e, n, *LabsAngleSchedule(p)(h))
    assert e.min() == instances.get(n).energy
    assert np.sum(np.abs(psi[e == e.min()]) ** 2) == pytest.approx(overlap, abs=1e-9)


def test_term_counts_and_cnots_of_the_master() -> None:
    h = NodeHamiltonian.master(20)
    assert term_counts(h) == {1: 0, 2: 90, 3: 0, 4: 525}
    assert cnots_per_layer(h) == 2 * 90 + 6 * 525


class NotLabs:
    size = 2


def test_angle_schedule_scales_master_nodes_and_skew_restriction() -> None:
    rule = LabsAngleSchedule(4)
    g_master, _ = rule(NodeHamiltonian.master(20))
    g_node, _ = rule(node(20, {0: 1}))
    g_skew, _ = rule(SkewHamiltonian(21))
    assert np.allclose(g_node, 0.625 * g_master)
    assert np.allclose(g_skew * 21, 0.5 * g_master * 20)
    g_node_p1, _ = LabsAngleSchedule(1)(node(20, {0: 1}))
    g_master_p1, _ = LabsAngleSchedule(1)(NodeHamiltonian.master(20))
    assert np.allclose(g_node_p1, 0.875 * g_master_p1)
    with pytest.raises(TypeError):
        rule(NotLabs())  # type: ignore[arg-type]


def test_qaoa_sampler_gives_exact_energies_and_is_reproducible() -> None:
    h = node(16, {0: 1, 1: 1, 15: -1})
    sampler = QaoaSampler(LabsAngleSchedule(4))
    a = sampler.sample([h], shots=200, seed=5)[0]
    b = sampler.sample([h], shots=200, seed=5)[0]
    assert np.array_equal(a.spins, b.spins) and np.array_equal(a.counts, b.counts)
    assert np.array_equal(h.energies(a.spins), a.energies)
    assert a.counts.sum() == 200


def job(n: int, fixed: dict[int, int], p: int = 1) -> QaoaJob:
    h = node(n, fixed)
    return QaoaJob.of(h, *LabsAngleSchedule(p)(h))


def probabilities(qc: object) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(qc).probabilities())


def logical(prog: object, size: int) -> tuple[float, np.ndarray]:
    """Exact acceptance and logical distribution of an Iceberg program."""
    qc = prog.circuit.remove_final_measurements(inplace=False)  # type: ignore[attr-defined]
    probs = probabilities(qc)
    width = qc.num_qubits
    x = np.arange(probs.size)
    b = ((x[:, None] >> np.arange(width)) & 1).astype(np.uint8)
    m, f = b[:, :-1], b[:, -1:]
    keep = (f[:, 0] == 0) & (m.sum(axis=1) % 2 == 0)
    y = (m[:, :size] ^ m[:, -1:]) @ (1 << np.arange(size))
    out = np.bincount(y[keep], weights=probs[keep], minlength=1 << size)
    return float(probs[keep].sum()), out


def test_circuit_matches_the_emulated_state() -> None:
    pytest.importorskip("qiskit")
    j = job(10, {0: 1, 3: -1, 9: 1}, p=2)
    h = node(10, {0: 1, 3: -1, 9: 1})
    assert all(term_counts(h).values())
    psi = qaoa_state(diagonal(h), h.size, np.array(j.gammas), np.array(j.betas))
    assert np.allclose(probabilities(qaoa_circuit(j)), np.abs(psi) ** 2, atol=1e-10)


@pytest.mark.parametrize("fixed", [{0: 1, 8: -1, 3: 1}, {0: 1, 8: -1}])
def test_iceberg_is_exact_without_noise(fixed: dict[int, int]) -> None:
    pytest.importorskip("qiskit")
    j = job(9, fixed)
    accepted, dist = logical(program(j, iceberg=True), j.size)
    assert accepted == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(dist, probabilities(qaoa_circuit(j)), atol=1e-10)


def test_iceberg_detects_a_single_error() -> None:
    pytest.importorskip("qiskit")
    j = job(9, {0: 1, 8: -1, 3: 1})
    for gate in ("x", "z"):
        prog = program(j, iceberg=True)
        qc = prog.circuit.remove_final_measurements(inplace=False)
        data = list(qc.data)
        k = j.size + j.size % 2
        cut = 1 + (k + 1) + (k + 2)
        qc.data = data[:cut]
        getattr(qc, gate)(2)
        for inst in data[cut:]:
            qc.append(inst)
        qc.measure_all()
        broken = type(prog)(qc, prog.size, True)
        accepted, _ = logical(broken, j.size)
        assert accepted == pytest.approx(0.0, abs=1e-10)


def test_packing_keeps_each_distribution() -> None:
    pytest.importorskip("qiskit")
    jobs = [job(7, {0: 1}), job(7, {0: 1, 6: -1}), job(6, {0: 1, 1: 1})]
    progs = [program(j) for j in jobs]
    assert device.groups(progs, 12) == [[0, 1], [2]]
    combined = device.combine(progs)
    probs = probabilities(combined.remove_final_measurements(inplace=False))
    start = 0
    for j in jobs:
        rest = probs.reshape(-1, 1 << j.size, 1 << start).sum(axis=(0, 2))
        assert np.allclose(rest, probabilities(qaoa_circuit(j)), atol=1e-10)
        start += j.size


def check_spins(z: list[np.ndarray], jobs: list[QaoaJob], shots: int) -> None:
    for s, j in zip(z, jobs, strict=True):
        assert s.shape[1] == j.size and 0 < s.shape[0] <= shots
        assert np.isin(s, (-1, 1)).all()


@pytest.mark.parametrize("iceberg", [False, True])
def test_ibm_local_emulation(iceberg: bool) -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    jobs = [job(9, {0: 1, 8: -1, 3: 1}), job(8, {0: 1, 7: 1})]
    dev = IbmDevice("fake_marrakesh", iceberg=iceberg)
    check_spins(dev.run(jobs, 100, seed=1), jobs, 100)
    est = dev.estimate(jobs, 100)
    assert est.circuits == 2 and 0 < est.fidelity < 1 and est.two_qubit_gates > 0


def test_ibm_real_device_needs_submit() -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit.providers.fake_provider import GenericBackendV2

    dev = IbmDevice(GenericBackendV2(20))
    with pytest.raises(RuntimeError):
        dev.run([job(8, {0: 1, 7: 1})], 10, seed=1)


@pytest.mark.parametrize("iceberg", [False, True])
def test_quantinuum_local_emulation(iceberg: bool) -> None:
    pytest.importorskip("pytket.extensions.quantinuum")
    jobs = [job(9, {0: 1, 8: -1, 3: 1}), job(8, {0: 1, 7: 1})]
    dev = QuantinuumDevice(iceberg=iceberg)
    native = {"ZZPhase", "ZZMax", "PhasedX", "Rz", "Measure", "Barrier"}
    for c in dev.compile([program(j, iceberg).circuit for j in jobs]):
        assert {g.op.type.name for g in c} <= native
    check_spins(dev.run(jobs, 100, seed=1), jobs, 100)
    with pytest.raises(RuntimeError):
        QuantinuumDevice(local=False).run(jobs, 10, seed=1)


def test_solve_with_local_quantinuum_proves_optimum() -> None:
    pytest.importorskip("pytket.extensions.quantinuum")
    sampler = QaoaSampler(LabsAngleSchedule(1), QuantinuumDevice(max_size=8))
    r = solve(LabsProblem(10, boundary=2), sampler, Settings(shots=50, seed=1))
    assert r.proven and r.ub == instances.get(10).energy


class Counting(Emulator):
    def __init__(self, max_size: int) -> None:
        super().__init__(max_size=max_size)
        self.seen: list[int] = []

    def run(self, jobs: Sequence[QaoaJob], shots: int, seed: int | None) -> list:
        self.seen += [j.size for j in jobs]
        return super().run(jobs, shots, seed)


def test_qaoa_sampler_rejects_hamiltonians_larger_than_the_backend() -> None:
    backend = Counting(max_size=9)
    sampler = QaoaSampler(LabsAngleSchedule(2), backend)
    problems = [node(12, dict.fromkeys(range(6), 1)), node(9, {})]
    sets = sampler.sample(problems, shots=3, seed=0)
    assert backend.seen == [6, 9]
    assert [s.spins.shape[1] for s in sets] == [6, 9]
    assert all(s.counts.sum() == 3 for s in sets)
    with pytest.raises(SizeError):
        sampler.sample([node(12, {}), node(9, {})], 3, 0)
    assert backend.seen == [6, 9]


def test_skew_restriction_has_zero_odd_autocorrelations() -> None:
    sk = SkewHamiltonian(13)
    seqs = sk.complete(all_states(sk))
    assert np.all(autocorrelations(seqs)[:, ::2] == 0)
    assert np.array_equal(sk.energies(all_states(sk)), diagonal(sk))
    assert diagonal(SkewHamiltonian(21)).min() == instances.get(21).energy


def test_skew_sampler_returns_master_samples() -> None:
    h = NodeHamiltonian.master(21)
    sampler = SkewSampler(QaoaSampler(LabsAngleSchedule(12)), with_master=False)
    s = sampler.sample([h, node(21, {0: 1})], shots=300, seed=2)
    assert s[0].spins.shape[1] == 21 and s[0].counts.sum() == 300
    assert np.array_equal(h.energies(s[0].spins), s[0].energies)
    assert np.all(autocorrelations(h.complete(s[0].spins))[:, ::2] == 0)
    assert s[1].spins.shape[1] == 20


def test_skew_sampler_merges_master_samples() -> None:
    h = NodeHamiltonian.master(19)
    qaoa = QaoaSampler(LabsAngleSchedule(4))
    merged = SkewSampler(qaoa).sample([h], shots=200, seed=3)[0]
    alone = SkewSampler(qaoa, with_master=False).sample([h], shots=200, seed=3)[0]
    assert merged.counts.sum() == 400 and alone.counts.sum() == 200
    assert merged.best <= alone.best
    assert np.any(autocorrelations(h.complete(merged.spins))[:, ::2] != 0)


def test_skew_sampler_with_unsampled_master_returns_skew_samples() -> None:
    h = NodeHamiltonian.master(19)
    inner = QaoaSampler(LabsAngleSchedule(4), Emulator(max_size=10))
    s = SkewSampler(inner, max_size=10).sample([h], shots=200, seed=3)[0]
    assert s.counts.sum() == 200 and s.spins.shape[1] == 19
    assert s.best == diagonal(SkewHamiltonian(19)).min()


def test_solve_with_qaoa_and_unsampled_master_proves_optimum() -> None:
    for n in (11, 14):
        backend = Emulator(max_size=10)
        sampler = QaoaSampler(LabsAngleSchedule(4), backend)
        r = solve(LabsProblem(n, boundary=2), sampler, Settings(shots=100, seed=1))
        assert r.proven and r.ub == instances.get(n).energy
        assert r.trace[0].ub is None
        assert r.calls[0].size == n and not r.calls[0].sampled
        assert all(c.sampled == (c.size <= 10) for c in r.calls)
        assert abs(sum(c.seconds for c in r.calls) - r.sampler_seconds) < 1e-9
        # the emulator counts the circuits and shots a quantum device would run
        assert backend.circuits == sum(c.sampled for c in r.calls)
        assert backend.shots == 100 * backend.circuits


def test_solve_rejects_a_node_below_the_master_larger_than_the_backend() -> None:
    sampler = QaoaSampler(LabsAngleSchedule(4), Emulator(max_size=10))
    with pytest.raises(SizeError):
        solve(LabsProblem(14, boundary=1), sampler, Settings(shots=100, seed=1))
