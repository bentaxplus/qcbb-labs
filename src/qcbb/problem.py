"""What QCBB needs from a problem.

QCBB is stated for a general binary problem min f(x) over x in {-1, +1}^n.
A problem supplies the node Hamiltonians for the sampler, the constraints
for the conflict values (QCBB eqs. 5-9) and the five conditions of the
correctness theorem of the paper:

    (i)   `lower_bound(node)` <= f(x) for every completion x of the node;
    (ii)  `propagate(node, T)` removes only completions with f > T;
    (iii) `class_roots()` and `passes_symmetry_rule` keep a representative
          of every orbit;
    (iv)  all values of f differ by multiples of `lattice_step`;
    (v)   `fix(node, var, +1)` and `fix(node, var, -1)` partition the node.

The sampler enters none of them.
"""

from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

Spins = NDArray[np.int8]
Ints = NDArray[np.int64]
Monomial = tuple[int, ...]

Node = TypeVar("Node")


class Hamiltonian(Protocol):
    """The objective of a node as a function of its free variables."""

    @property
    def size(self) -> int:
        """Number of free variables F, the number of qubits."""
        ...

    def energies(self, z: Spins) -> Ints:
        """Return the exact objective of free-variable assignments, shape (m,)."""
        ...

    def monomials(self) -> tuple[int, dict[Monomial, int]]:
        """Return the constant and the Ising monomials over local indices."""
        ...


class Problem(Protocol[Node]):
    """A minimization problem over +-1 variables, solvable by QCBB.

    Attributes:
        lattice_step: Step Delta of the lattice of objective values, condition (iv).
    """

    lattice_step: int

    def master(self) -> Node:
        """Return the master node, with no variable fixed."""
        ...

    def class_roots(self) -> list[Node]:
        """Return the children of the master that pass the symmetry rule, (iii)."""
        ...

    def hamiltonian(self, node: Node) -> Hamiltonian:
        """Return the node Hamiltonian (QCBB eqs. 10-16)."""
        ...

    def complete(self, node: Node, z: Spins) -> Spins:
        """Return the full assignment with the free variables set to z."""
        ...

    def objective(self, x: Spins) -> int:
        """Return the exact objective f(x) of a full assignment."""
        ...

    def free(self, node: Node) -> Ints:
        """Return the free variables of a node, in local index order."""
        ...

    def fix(self, node: Node, var: int, value: int) -> Node:
        """Return the child with `var` fixed to `value`, condition (v)."""
        ...

    def lower_bound(self, node: Node) -> int:
        """Return a valid lower bound LB on the node, condition (i)."""
        ...

    def propagate(self, node: Node, threshold: int) -> Node | None:
        """Fix forced variables, or return None if no completion has f <= T.

        Condition (ii).
        """
        ...

    def passes_symmetry_rule(self, node: Node) -> bool:
        """Return False only if no completion is an orbit representative."""
        ...

    def violations(self, node: Node, x: Spins, threshold: int) -> NDArray[np.bool_]:
        """Return V of QCBB eq. 5, shape (constraints, m), for full assignments x."""
        ...

    def correspondence(self, node: Node) -> NDArray[np.int8]:
        """Return the correspondence matrix P of QCBB eq. 7, shape (constraints, F)."""
        ...

    def candidates(self, node: Node) -> Ints:
        """Return the local indices of the free variables eligible for eq. 9."""
        ...

    def tie_key(self, node: Node, var: int) -> tuple[int, int]:
        """Return the tie-break key of a variable; the smallest key wins."""
        ...
