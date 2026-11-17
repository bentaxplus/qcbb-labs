"""QCBB: quantum-classical hybrid branch and bound.

This package is the problem-agnostic algorithm: the problem contract
(`problem`), the sampler contract (`sampler`), the conflict values
(`conflict`) and the node loop (`node_loop`). `qcbb.labs` is the LABS
instance and `qcbb.qaoa` the fixed-angle QAOA sampler with its backends.
See docs/algorithm.md.
"""

from qcbb.conflict import branching_variable, conflict_values
from qcbb.node_loop import (
    Call,
    Record,
    Result,
    Settings,
    certified_lower_bound,
    primal_dual_integral,
    solve,
)
from qcbb.problem import Hamiltonian, Problem
from qcbb.sampler import Sampler, SampleSet, SizeError, collect

__version__ = "0.1.0"

__all__ = [
    "Call",
    "Hamiltonian",
    "Problem",
    "Record",
    "Result",
    "SampleSet",
    "Sampler",
    "Settings",
    "branching_variable",
    "certified_lower_bound",
    "SizeError",
    "collect",
    "conflict_values",
    "primal_dual_integral",
    "solve",
]
