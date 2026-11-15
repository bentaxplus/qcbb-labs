"""The node loop: proofs, limits, the trace and the primal-dual integral."""

import time
from collections.abc import Sequence

import numpy as np
import pytest
from helpers import EmptySampler, WorstSampler

from qcbb import (
    Hamiltonian,
    Record,
    Result,
    SampleSet,
    Settings,
    certified_lower_bound,
    conflict_values,
    primal_dual_integral,
    solve,
)
from qcbb.labs import AnnealingSampler, LabsProblem, NodeHamiltonian, bound, instances
from qcbb.labs.energy import energy
from qcbb.node_loop import INFINITY


@pytest.mark.parametrize("n", range(2, 15))
@pytest.mark.parametrize("candidates", ["all", "outermost"])
def test_proven_optimum_equals_qoblib(n: int, candidates: str) -> None:
    problem = LabsProblem(n, boundary=3, candidates=candidates)  # type: ignore[arg-type]
    result = solve(problem, AnnealingSampler(sweeps=200), Settings(shots=16, seed=0))
    assert result.proven
    assert result.ub == instances.get(n).energy == energy(result.incumbent)
    assert result.lb == result.ub


class CountingSampler:
    """Simulated annealing that records the size of every call."""

    def __init__(self) -> None:
        self.inner = AnnealingSampler(sweeps=100)
        self.sizes: list[int] = []

    def sample(
        self, problems: Sequence[Hamiltonian], shots: int, seed: int | None = None
    ) -> list[SampleSet]:
        self.sizes.append(len(problems))
        return self.inner.sample(problems, shots, seed)


class CountingProblem(LabsProblem):
    """LABS problem that counts the node Hamiltonians it builds."""

    def __init__(self, n: int, boundary: int) -> None:
        super().__init__(n, boundary=boundary)
        self.hamiltonians = 0

    def hamiltonian(self, node: np.ndarray) -> NodeHamiltonian:
        self.hamiltonians += 1
        return super().hamiltonian(node)


@pytest.mark.parametrize("n", range(15, 21))
def test_every_node_hamiltonian_is_sampled_once(n: int) -> None:
    sampler = CountingSampler()
    problem = CountingProblem(n, boundary=4)
    result = solve(problem, sampler, Settings(shots=16, seed=0))
    assert result.proven
    assert result.ub == instances.get(n).energy == energy(result.incumbent)
    assert sampler.sizes == [1] * result.sampler_calls
    assert problem.hamiltonians == result.sampler_calls
    assert 0 < result.sampler_calls <= result.nodes


def test_master_is_sampled_first_and_its_children_are_the_class_roots() -> None:
    n, m = 18, 4
    result = solve(
        LabsProblem(n, boundary=m), AnnealingSampler(sweeps=100), Settings(16, seed=0)
    )
    sizes = [c.size for c in result.calls]
    assert sizes[0] == n
    assert all(f <= n - 2 * m for f in sizes[1:])


@pytest.mark.parametrize("sampler", [WorstSampler(), EmptySampler()])
@pytest.mark.parametrize("n", [9, 12, 13])
def test_adversarial_and_empty_samplers_still_prove(sampler: object, n: int) -> None:
    result = solve(LabsProblem(n, boundary=3), sampler, Settings(shots=1))  # type: ignore[arg-type]
    assert result.proven
    assert result.ub == instances.get(n).energy


def check_trace(result: Result) -> None:
    """The trace is monotone, records only changes, and ends at the result."""
    trace = result.trace
    assert trace and trace[0].lb is not None
    assert trace[-1].ub == result.ub and trace[-1].lb == result.lb
    pairs = list(zip(trace, trace[1:], strict=False))
    assert all(a.seconds <= b.seconds for a, b in pairs)
    assert all((a.ub, a.lb) != (b.ub, b.lb) for a, b in pairs)
    ubs = [e.ub for e in trace if e.ub is not None]
    assert all(a >= b for a, b in zip(ubs, ubs[1:], strict=False))
    lbs = [e.lb for e in trace if e.lb is not None]
    assert all(a <= b for a, b in zip(lbs, lbs[1:], strict=False))
    assert all(e.lb <= e.ub for e in trace if e.lb is not None and e.ub is not None)
    assert result.gap == trace[-1].gap
    assert 0 <= result.sampler_seconds <= result.seconds
    assert 0 <= result.pdi <= result.seconds


@pytest.mark.parametrize("n", [13, 20, 23])
def test_trace_of_a_proof_closes_the_gap(n: int) -> None:
    problem = LabsProblem(n, boundary=4)
    result = solve(problem, AnnealingSampler(sweeps=100), Settings(shots=16, seed=0))
    check_trace(result)
    assert result.proven and result.trace[-1].gap == 0.0


@pytest.mark.parametrize("n", [14, 16, 22])
def test_node_limit_gives_a_certified_gap_on_the_lattice(n: int) -> None:
    result = solve(LabsProblem(n, boundary=4), WorstSampler(), Settings(node_limit=5))
    check_trace(result)
    assert result.lb <= instances.get(n).energy <= result.ub
    assert (result.ub - result.lb) % LabsProblem(n).lattice_step == 0
    if not result.proven:
        assert result.trace[-1].gap > 0 and result.pdi > 0


class SlowSampler(WorstSampler):
    """Adversarial sampler that takes a fixed time per call."""

    def sample(
        self, problems: Sequence[Hamiltonian], shots: int, seed: int | None = None
    ) -> list[SampleSet]:
        time.sleep(0.02)
        return super().sample(problems, shots, seed)


def test_time_limit_gives_a_certified_gap() -> None:
    n = 20
    result = solve(LabsProblem(n, boundary=2), SlowSampler(), Settings(time_limit=0.2))
    check_trace(result)
    assert not result.proven
    assert result.seconds >= 0.2 and result.nodes < 1000
    assert result.lb <= instances.get(n).energy <= result.ub


def test_limit_before_any_incumbent_raises() -> None:
    with pytest.raises(RuntimeError):
        solve(LabsProblem(12, boundary=2), WorstSampler(), Settings(node_limit=0))


def test_certified_lower_bound_rounds_up_to_the_lattice() -> None:
    problem = LabsProblem(16)  # Delta = 4
    assert certified_lower_bound(problem, None, None) is None
    assert certified_lower_bound(problem, None, 13) == 13
    assert certified_lower_bound(problem, 24, None) == 24
    assert certified_lower_bound(problem, 24, 30) == 24
    assert certified_lower_bound(problem, 24, 20) == 20
    assert certified_lower_bound(problem, 24, 17) == 20
    assert certified_lower_bound(problem, 24, 13) == 16
    assert certified_lower_bound(LabsProblem(18), 25, 10) == 17  # Delta = 8


def test_infinite_threshold_before_the_first_incumbent() -> None:
    rng = np.random.default_rng(5)
    for n in (9, 16, 30):
        node = np.zeros(n, dtype=np.int8)
        node[: n // 3] = rng.choice(np.array([1, -1], dtype=np.int8), size=n // 3)
        assert np.array_equal(bound.propagate(node, INFINITY), node)
        b = bound.budgets(node, INFINITY)
        assert np.all(b > INFINITY - 4 * n * n) and np.all(b <= INFINITY)
        problem = LabsProblem(n)
        samples = WorstSampler().sample([problem.hamiltonian(node)], 1)[0]
        assert np.all(conflict_values(problem, node, samples, INFINITY) == 0)


def test_primal_dual_integral_of_a_step_trace() -> None:
    trace = [
        Record(1.0, 100, None),
        Record(2.0, 80, None),
        Record(3.0, 80, 40),
        Record(5.0, 60, 40),
        Record(6.0, 60, 60),
    ]
    assert [e.gap for e in trace] == [1.0, 1.0, 0.5, pytest.approx(1 / 3), 0.0]
    expected = 1.0 * 1.0 + 1.0 * 1.0 + 1.0 * 1.0 + 0.5 * 2.0 + (1 / 3) * 1.0 + 0.0 * 2.0
    assert primal_dual_integral(trace, 8.0) == pytest.approx(expected)
    assert primal_dual_integral([], 4.0) == 4.0
    assert Record(0.0, -3, -5).gap == pytest.approx(2 / 5)
