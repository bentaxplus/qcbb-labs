"""Iceberg error detection, the [[k+2, k, 2]] code, for QAOA circuits.

He, Amaro, Shaydulin and Pistoia (Commun. Phys. 2025, arXiv:2409.12104) run
QAOA in this code on a trapped-ion device and keep only shots without a
detected error. The code has k data qubits (k even), a top qubit t and a
bottom qubit b, with stabilizers S_X = X^(k+2) and S_Z = Z^(k+2) and logical
operators X_i = X_t X_i and Z_i = Z_i Z_b. Hence:

    logical exp(-i theta Z_S / 2)  = physical exp(-i theta Z_S / 2) if |S| is even,
                                     and on S + {b} if |S| is odd;
    logical exp(-i beta X_i)       = physical exp(-i beta X_t X_i) (rxx(2 beta));
    logical |+>^k                  = the X-basis GHZ state on all k + 2 qubits.

The LABS master has only 2- and 4-spin terms, so its phase layer is the same
physical circuit as without the code. Here the syndrome is read once, at the
end: S_X through one ancilla (register "f") and S_Z as the parity of the final
Z measurement of all k + 2 qubits (register "m"). A shot is kept if both are
+1; its logical bits are m_i XOR m_b. Every logical gate is a Pauli rotation
of even weight, so it commutes with both stabilizers, and a single Pauli
error between logical gates that anticommutes with one is detected. An error
inside a CNOT ladder can spread to two qubits and pass, errors on the ancilla
are not caught, and errors that cancel in pairs pass: this is detection, not
a fault-tolerant scheme. An odd F is padded with one idle logical qubit,
which is dropped when decoding.
"""

from typing import Any

import numpy as np

from qcbb.problem import Spins
from qcbb.qaoa.backend import QaoaJob
from qcbb.qaoa.circuits import Bits, rotate_z


def encode(job: QaoaJob) -> Any:
    """Return the Iceberg-encoded, measured QAOA circuit of a job.

    Qubits 0..k-1 are data (k = F rounded up to even), k is t, k + 1 is b and
    k + 2 is the syndrome ancilla.
    """
    from qiskit import ClassicalRegister, QuantumCircuit

    k = job.size + job.size % 2
    t, b, a = k, k + 1, k + 2
    code = list(range(k + 2))
    qc = QuantumCircuit(k + 3)
    qc.h(t)
    for q in [*range(k), b]:
        qc.cx(t, q)
    qc.h(code)
    for gamma, beta in zip(job.gammas, job.betas, strict=True):
        for mono, w in job.terms:
            rotate_z(qc, [*mono, b] if len(mono) % 2 else list(mono), gamma * w)
        for i in range(job.size):
            qc.rxx(2 * beta, t, i)
    qc.h(a)
    for q in code:
        qc.cx(a, q)
    qc.h(a)
    m, f = ClassicalRegister(k + 2, "m"), ClassicalRegister(1, "f")
    qc.add_register(m)
    qc.add_register(f)
    qc.measure(a, f[0])
    qc.measure(code, m)
    return qc


def decode(m: Bits, f: Bits, size: int) -> Spins:
    """Return the logical spins of the shots with no detected error.

    Args:
        m: Final Z measurement of the k + 2 code qubits, shape (shots, k + 2).
        f: S_X syndrome bit, shape (shots, 1).
        size: Number of logical qubits F.

    Returns:
        Spins of the accepted shots, shape (accepted, F).
    """
    keep = (f[:, 0] == 0) & (m.sum(axis=1) % 2 == 0)
    logical = m[keep, :size] ^ m[keep, -1:]
    return (1 - 2 * logical.astype(np.int8)).astype(np.int8)
