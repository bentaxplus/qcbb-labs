"""The run flow shared by the gate-based devices, with circuit packing.

For a batch of jobs: build each job's program (Iceberg-encoded if asked),
pack the programs into as few circuits as the device allows, compile and run
them (device-specific), split the measured registers back to the jobs and
decode. A real device runs only with `submit=True`; before each submission
the resource estimate is logged, so the cost is visible before it is spent.
Local emulation (a fake backend or a noise model) always runs.

Circuit packing (Liaqat et al., arXiv:2610.10772) places independent QAOA
circuits on disjoint qubits of one chip. Node circuits with about 10 qubits
use a small part of a 56- to 156-qubit device, so the programs of one batch
are grouped up to a qubit capacity and composed into one circuit. Register r
of the i-th program in a group is renamed r + str(i), and the measured bits
of each program are read back from its own registers.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from qcbb.problem import Spins
from qcbb.qaoa.backend import Estimate, QaoaJob, log
from qcbb.qaoa.circuits import Bits, Program, program


def groups(programs: Sequence[Program], capacity: int) -> list[list[int]]:
    """Group program indices in order so that each group fits `capacity` qubits.

    A program larger than the capacity gets a group of its own.
    """
    out: list[list[int]] = []
    used = capacity
    for i, p in enumerate(programs):
        width = p.circuit.num_qubits
        if used + width > capacity:
            out.append([])
            used = 0
        out[-1].append(i)
        used += width
    return out


def combine(programs: Sequence[Program]) -> Any:
    """Return one circuit with the programs on disjoint qubits."""
    from qiskit import ClassicalRegister, QuantumCircuit

    qc = QuantumCircuit(sum(p.circuit.num_qubits for p in programs))
    start = 0
    for i, p in enumerate(programs):
        regs = [ClassicalRegister(r.size, f"{r.name}{i}") for r in p.circuit.cregs]
        for r in regs:
            qc.add_register(r)
        width = p.circuit.num_qubits
        qubits = list(range(start, start + width))
        clbits = [bit for r in regs for bit in r]
        qc.compose(p.circuit, qubits=qubits, clbits=clbits, inplace=True)
        start += width
    return qc


def split(programs: Sequence[Program], bits: dict[str, Bits]) -> list[dict[str, Bits]]:
    """Return the registers of each program of a combined circuit, original names."""
    return [
        {r.name: bits[f"{r.name}{i}"] for r in p.circuit.cregs}
        for i, p in enumerate(programs)
    ]


class Device(ABC):
    """A gate-based device: shared encoding, packing and decoding.

    Args:
        max_size: Largest logical F accepted (`QaoaSampler` skips larger ones).
        iceberg: Encode every job in the Iceberg code and post-select.
        pack: Pack several programs into one circuit (real devices only).
        fill: Share of the device's qubits used when packing.
        submit: Allow submission to a real device.
    """

    local: bool

    def __init__(
        self, max_size: int, iceberg: bool, pack: bool, fill: float, submit: bool
    ) -> None:
        self.max_size = max_size
        self.iceberg = iceberg
        self.pack = pack
        self.fill = fill
        self.submit = submit
        self.acceptance: list[float] = []

    @property
    @abstractmethod
    def qubits(self) -> int:
        """Number of physical qubits of the device."""

    @abstractmethod
    def execute(
        self, circuits: Sequence[Any], shots: int, seed: int | None
    ) -> list[dict[str, Bits]]:
        """Compile and run circuits; return the bits of every register per circuit."""

    @abstractmethod
    def resources(self, circuits: Sequence[Any], shots: int) -> Estimate:
        """Compile circuits and return their resource estimate."""

    def estimate(self, jobs: Sequence[QaoaJob], shots: int) -> Estimate:
        """Return the resources of running the jobs, without running them."""
        return self.resources(self._circuits(self._programs(jobs))[1], shots)

    def run(self, jobs: Sequence[QaoaJob], shots: int, seed: int | None) -> list[Spins]:
        """Run the jobs and return the spins of the accepted shots of each.

        Args:
            jobs: QAOA jobs, each with at most `max_size` qubits.
            shots: Shots per circuit.
            seed: Seed of local emulation (ignored by real devices).

        Returns:
            Spins per job, shape (accepted shots, F).

        Raises:
            ValueError: If a job is larger than `max_size`.
            RuntimeError: If the device is real and `submit` is False.
        """
        if any(j.size > self.max_size for j in jobs):
            raise ValueError(f"a job exceeds max_size = {self.max_size}")
        if not self.local and not self.submit:
            raise RuntimeError(
                "submission to a real device is off; check estimate() and pass "
                "submit=True"
            )
        programs = self._programs(jobs)
        index, circuits = self._circuits(programs)
        if not self.local:
            log.warning("submitting: %s", self.resources(circuits, shots))
        out: list[Spins] = []
        for group, bits in zip(index, self.execute(circuits, shots, seed), strict=True):
            members = [programs[i] for i in group]
            for p, regs in zip(members, split(members, bits), strict=True):
                out.append(p.decode(regs))
        self.acceptance = [z.shape[0] / shots for z in out]
        return out

    def _programs(self, jobs: Sequence[QaoaJob]) -> list[Program]:
        """Return the program of each job."""
        return [program(j, self.iceberg) for j in jobs]

    def _circuits(
        self, programs: Sequence[Program]
    ) -> tuple[list[list[int]], list[Any]]:
        """Return the packing groups (job order is kept) and their circuits."""
        if self.pack and not self.local:
            index = groups(programs, int(self.fill * self.qubits))
        else:
            index = [[i] for i in range(len(programs))]
        return index, [combine([programs[i] for i in g]) for g in index]
