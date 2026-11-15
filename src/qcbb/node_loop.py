"""The QCBB node loop (paper, Algorithm 1 and Fig. 1).

Every node is evaluated in the same way:

1. drop the node if it fails the symmetry rule, or if propagation shows that
   no completion has an objective <= T = UB - Delta;
2. compute the node's lower bound and prune if it exceeds T;
3. at a leaf (no free variable) the objective is exact: update UB;
4. otherwise sample the node Hamiltonian with the run's sampler and offer the
   best sample as an incumbent candidate;
5. branch: the children of the master are the class roots, the children of
   every other node fix the variable chosen by the conflict values of the
   node's own samples (QCBB eqs. 5-9).

Nodes are taken best first: the lowest key, then the deeper node, then the
older one. A child enters the queue with the key max(parent bound, own bound)
if that is at most T. The run ends when the lowest key exceeds T, and UB is
then optimal, or at a node or time limit with a certified gap. One sampler
serves every node of a run.

After every node, (time, UB, certified LB) is recorded in the trace whenever
one of the bounds changed. The certified LB is min(UB, lowest key) rounded up
to the lattice. The primal-dual integral of the trace (Berthold 2013) measures
the convergence of a run.
"""

import heapq
import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qcbb.conflict import branching_variable, conflict_values
from qcbb.problem import Problem, Spins
from qcbb.sampler import Sampler, SampleSet

INFINITY = 2**40
"""The threshold before the first incumbent: T = +inf."""


@dataclass
class Settings:
    """Run settings.

    Attributes:
        shots: Samples per node.
        node_limit: Stop after this many nodes, or None.
        time_limit: Stop after this many seconds, or None.
        seed: Random seed; None draws fresh samples on every run.
    """

    shots: int = 1000
    node_limit: int | None = None
    time_limit: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class Record:
    """One point of the trace.

    Attributes:
        seconds: Wall time since the start of the run.
        ub: Upper bound UB at that time, or None before the first incumbent.
        lb: Certified lower bound at that time, or None before the first bound.
    """

    seconds: float
    ub: int | None
    lb: int | None

    @property
    def gap(self) -> float:
        """Primal-dual gap: 1 without both bounds, 0 at UB = LB."""
        if self.lb is None or self.ub is None:
            return 1.0
        if self.ub == self.lb:
            return 0.0
        return (self.ub - self.lb) / max(abs(self.ub), abs(self.lb))


def primal_dual_integral(trace: Sequence[Record], seconds: float) -> float:
    """Return the integral of the primal-dual gap over [0, seconds].

    The gap is 1 until the first record and a step function between records
    (Berthold, Oper. Res. Lett. 41, 2013).

    Args:
        trace: Records in time order.
        seconds: End of the run.

    Returns:
        The primal-dual integral, in seconds.
    """
    total, t, gap = 0.0, 0.0, 1.0
    for record in trace:
        total += gap * (record.seconds - t)
        t, gap = record.seconds, record.gap
    return total + gap * (seconds - t)


@dataclass(frozen=True)
class Call:
    """One sampler call (one node).

    Attributes:
        size: Number of free variables F of the node Hamiltonian.
        sampled: True if the sampler returned at least one sample (a
            Hamiltonian larger than the backend takes returns none).
        seconds: Wall time of the call.
    """

    size: int
    sampled: bool
    seconds: float


@dataclass(frozen=True, eq=False)
class Result:
    """Outcome of a run.

    Attributes:
        incumbent: Best assignment found.
        ub: Its objective, the upper bound UB.
        lb: Certified lower bound on the optimum.
        proven: True if UB is proven optimal.
        nodes: Nodes evaluated.
        seconds: Wall time.
        trace: Records (time, UB, certified LB) in time order.
        calls: The sampler calls in time order.
    """

    incumbent: Spins
    ub: int
    lb: int
    proven: bool
    nodes: int
    seconds: float
    trace: tuple[Record, ...] = ()
    calls: tuple[Call, ...] = ()

    @property
    def sampler_calls(self) -> int:
        """Number of sampler calls."""
        return len(self.calls)

    @property
    def sampler_seconds(self) -> float:
        """Wall time spent in sampler calls."""
        return sum(c.seconds for c in self.calls)

    @property
    def gap(self) -> float:
        """Relative gap (UB - LB) / max(|UB|, |LB|); 0 when proven."""
        return Record(self.seconds, self.ub, self.lb).gap

    @property
    def pdi(self) -> float:
        """Primal-dual integral of the trace over the run."""
        return primal_dual_integral(self.trace, self.seconds)


def certified_lower_bound(
    problem: Problem[Any], ub: int | None, lowest: int | None
) -> int | None:
    """Return min(UB, lowest open key), rounded up to the lattice.

    Every objective value differs from UB by a multiple of Delta (condition
    (iv)), so a bound between two lattice points can be rounded up.

    Args:
        problem: The problem.
        ub: Upper bound, an attained objective value, or None.
        lowest: Lowest key of the open nodes, or None if none is open.

    Returns:
        The certified lower bound, or None if neither bound exists yet.
    """
    if ub is None:
        return lowest
    if lowest is None or lowest >= ub:
        return ub
    return ub - problem.lattice_step * ((ub - lowest) // problem.lattice_step)


@dataclass
class _Run:
    """State of one run: incumbent, clock, trace and sampler calls."""

    problem: Problem[Any]
    sampler: Sampler
    settings: Settings
    start: float = field(default_factory=time.perf_counter)
    incumbent: Spins | None = None
    ub: int | None = None
    trace: list[Record] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Seconds since the start."""
        return time.perf_counter() - self.start

    @property
    def threshold(self) -> int:
        """The current threshold T = UB - Delta, or +inf before an incumbent."""
        if self.ub is None:
            return INFINITY
        return self.ub - self.problem.lattice_step

    def offer(self, x: Spins, value: int) -> None:
        """Take an assignment as incumbent if it improves UB."""
        if self.ub is None or value < self.ub:
            self.incumbent, self.ub = x, value

    def record(self, lb: int | None) -> None:
        """Append (time, UB, LB) to the trace if a bound changed."""
        if not self.trace or (self.trace[-1].ub, self.trace[-1].lb) != (self.ub, lb):
            self.trace.append(Record(self.elapsed, self.ub, lb))

    def sample(self, hamiltonian: Any) -> SampleSet:
        """Sample one node Hamiltonian and record the call."""
        seed = self.settings.seed
        if seed is not None:
            seed += len(self.calls)
        t = time.perf_counter()
        samples = self.sampler.sample([hamiltonian], self.settings.shots, seed)[0]
        seconds = time.perf_counter() - t
        self.calls.append(Call(hamiltonian.size, samples.spins.shape[0] > 0, seconds))
        return samples

    def evaluate(self, node: Any, depth: int) -> tuple[int, list[Any]]:
        """Evaluate one node (steps 1-5 of the module docstring).

        Args:
            node: The node.
            depth: Minus the node's depth; 0 for the master.

        Returns:
            The node's lower bound and its children (none if it is dropped,
            pruned or a leaf).
        """
        problem, threshold = self.problem, self.threshold
        if not problem.passes_symmetry_rule(node):
            return 0, []
        node = problem.propagate(node, threshold)
        if node is None:
            return 0, []
        bound = problem.lower_bound(node)
        if bound > threshold:
            return bound, []
        if problem.free(node).size == 0:
            x = problem.complete(node, np.empty(0, dtype=np.int8))
            self.offer(x, problem.objective(x))
            return bound, []
        samples = self.sample(problem.hamiltonian(node))
        if samples.spins.shape[0]:
            self.offer(problem.complete(node, samples.spins[0]), samples.best)
        if bound > self.threshold:
            return bound, []
        if depth == 0:
            return bound, list(problem.class_roots())
        gamma = conflict_values(problem, node, samples, self.threshold)
        var = branching_variable(problem, node, gamma)
        return bound, [problem.fix(node, var, 1), problem.fix(node, var, -1)]


def solve(
    problem: Problem[Any], sampler: Sampler, settings: Settings | None = None
) -> Result:
    """Run QCBB and return a proven optimum or a certified gap.

    The limits are checked before each node, so a run may overshoot the time
    limit by one node.

    Args:
        problem: The problem.
        sampler: The sampler of every node; correctness does not depend on it.
        settings: Run settings.

    Returns:
        The outcome of the run.

    Raises:
        RuntimeError: If a limit stops the run before any incumbent exists.
    """
    settings = settings or Settings()
    run = _Run(problem, sampler, settings)
    order = itertools.count()
    master = problem.master()
    queue: list[tuple[int, int, int, Any]] = [
        (problem.lower_bound(master), 0, next(order), master)
    ]
    nodes = 0
    while queue and queue[0][0] <= run.threshold:
        if settings.node_limit is not None and nodes >= settings.node_limit:
            break
        if settings.time_limit is not None and run.elapsed >= settings.time_limit:
            break
        _, depth, _, node = heapq.heappop(queue)
        nodes += 1
        parent, children = run.evaluate(node, depth)
        for child in children:
            key = max(parent, problem.lower_bound(child))
            if key <= run.threshold:
                heapq.heappush(queue, (key, depth - 1, next(order), child))
        lowest = queue[0][0] if queue else None
        run.record(certified_lower_bound(problem, run.ub, lowest))

    if run.incumbent is None or run.ub is None:
        raise RuntimeError("a limit was reached before any incumbent")
    lowest = queue[0][0] if queue and queue[0][0] <= run.threshold else None
    lb = certified_lower_bound(problem, run.ub, lowest)
    assert lb is not None
    run.record(lb)
    return Result(
        incumbent=run.incumbent,
        ub=run.ub,
        lb=lb,
        proven=lb == run.ub,
        nodes=nodes,
        seconds=run.elapsed,
        trace=tuple(run.trace),
        calls=tuple(run.calls),
    )
