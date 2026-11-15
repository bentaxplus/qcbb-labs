"""QAOA circuits in qiskit form, shared by the gate-based devices.

A Z-string rotation exp(-i theta Z_S / 2) is rz for one qubit, rzz for two,
and for more qubits a CNOT ladder that collects the parity of all but the last
qubit, rzz with the last qubit, and the ladder back. Devices with a native
fractional RZZ (IBM Heron, Quantinuum) run the rzz directly; elsewhere the
compiler turns it into two CNOTs. The mixer is rx(2 beta) on every qubit.

A `Program` is the circuit of one job with its classical registers: "m" for
the measured qubits and, with error detection, "f" for the syndrome. Its
`decode` turns raw register bits into the job's spins and drops rejected
shots. Qubit j is local spin j, and a measured 1 means z_j = -1.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qcbb.problem import Hamiltonian, Spins
from qcbb.qaoa.backend import QaoaJob

Bits = NDArray[np.uint8]


def term_counts(hamiltonian: Hamiltonian) -> dict[int, int]:
    """Return the number of monomials with 1, 2, 3 and 4 spins."""
    counts = dict.fromkeys(range(1, 5), 0)
    for mono in hamiltonian.monomials()[1]:
        counts[len(mono)] += 1
    return counts


def cnots_per_layer(hamiltonian: Hamiltonian) -> int:
    """Return the CNOTs of one phase layer, 2 (|S| - 1) per monomial, no routing."""
    return sum(2 * (k - 1) * c for k, c in term_counts(hamiltonian).items())


def rotate_z(qc: Any, qubits: list[int], angle: float) -> None:
    """Append exp(-i angle Z_S / 2) on the given qubits to a qiskit circuit."""
    if len(qubits) == 1:
        qc.rz(angle, qubits[0])
        return
    ladder = list(zip(qubits[:-2], qubits[1:-1], strict=True))
    for a, b in ladder:
        qc.cx(a, b)
    qc.rzz(angle, qubits[-2], qubits[-1])
    for a, b in reversed(ladder):
        qc.cx(a, b)


def qaoa_circuit(job: QaoaJob) -> Any:
    """Return the QAOA circuit of a job as a qiskit `QuantumCircuit`, unmeasured."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(job.size)
    qc.h(range(job.size))
    for gamma, beta in zip(job.gammas, job.betas, strict=True):
        for mono, w in job.terms:
            rotate_z(qc, list(mono), gamma * w)
        qc.rx(2 * beta, range(job.size))
    return qc


def registers(data: Any, circuit: Any) -> dict[str, Bits]:
    """Return the bits of each register of `circuit` from a qiskit sampler result.

    Args:
        data: The `data` field of one result of a qiskit `SamplerV2`.
        circuit: The circuit that was run (gives the register names).

    Returns:
        Bits per register, column j = classical bit j.
    """
    return {
        r.name: getattr(data, r.name).to_bool_array(order="little").astype(np.uint8)
        for r in circuit.cregs
    }


@dataclass(frozen=True)
class Program:
    """A measured circuit for one job.

    Attributes:
        circuit: qiskit `QuantumCircuit` with register "m" (and "f").
        size: Number of logical qubits F of the job.
        iceberg: Whether the circuit is Iceberg-encoded.
    """

    circuit: Any
    size: int
    iceberg: bool

    def decode(self, registers: dict[str, Bits]) -> Spins:
        """Return the spins of the accepted shots.

        Args:
            registers: Bits per register name, shape (shots, width), column j
                = classical bit j.

        Returns:
            Spins, shape (accepted shots, F).
        """
        if self.iceberg:
            from qcbb.qaoa.iceberg import decode

            return decode(registers["m"], registers["f"], self.size)
        return (1 - 2 * registers["m"].astype(np.int8)).astype(np.int8)


def program(job: QaoaJob, iceberg: bool = False) -> Program:
    """Return the measured circuit of a job, Iceberg-encoded if asked."""
    if iceberg:
        from qcbb.qaoa.iceberg import encode

        return Program(encode(job), job.size, True)
    from qiskit import ClassicalRegister

    qc = qaoa_circuit(job)
    m = ClassicalRegister(job.size, "m")
    qc.add_register(m)
    qc.measure(range(job.size), m)
    return Program(qc, job.size, False)
