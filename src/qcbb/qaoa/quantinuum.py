"""Quantinuum devices (needs the `quantinuum` extra: pytket, qnexus).

Trapped-ion H-series devices connect all qubits and have a native
arbitrary-angle ZZPhase gate, so the 4-spin terms of LABS need no routing.
One choice per hardware problem:

- compilation: the device's own pytket pass (rebase to ZZPhase, PhasedX, Rz
  and optimization), run offline;
- error detection: Iceberg, on by default, the code shown to improve QAOA on
  H2 (He et al. 2025; Jin et al. 2025);
- throughput: spatial circuit packing.

`local=True` (default) compiles offline and emulates the compiled circuit with
Aer and a depolarizing noise model (`Noise`, rough H2-class rates, not a
calibrated device model). `local=False` submits to Nexus (Quantinuum's cloud)
with the login made by `qnexus.login()`; no credentials pass through this
module. Before submission the cost in HQC is queried and logged, and
`max_cost` caps it. Helios-generation systems take Guppy/HUGR programs, not
pytket circuits, and are not supported here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from qcbb.qaoa.backend import Estimate, log
from qcbb.qaoa.circuits import Bits, registers
from qcbb.qaoa.device import Device

TWO_QUBIT = ("ZZPhase", "ZZMax", "TK2")


@dataclass(frozen=True)
class Noise:
    """Depolarizing error rates of the local emulation.

    The defaults are of the order of the published H2 figures; they are a
    rough model for testing, not a calibration.

    Attributes:
        one: Error of a single-qubit gate (PhasedX; Rz is virtual).
        two: Error of a two-qubit gate.
        readout: Probability that a measured bit is flipped.
    """

    one: float = 3e-5
    two: float = 1e-3
    readout: float = 1e-3


class QuantinuumDevice(Device):
    """Run QAOA jobs on a Quantinuum H-series device or a local emulation.

    Args:
        device: Device name, e.g. "H2-1" (QPU) or "H2-1E" (Nexus emulator).
        local: Compile offline and emulate locally instead of using Nexus.
        max_size: Largest logical F accepted.
        iceberg: Encode jobs in the Iceberg code and post-select.
        pack: Pack several circuits into one (Nexus only).
        fill: Share of the device's qubits used when packing.
        noise: Error rates of the local emulation.
        submit: Allow submission to Nexus.
        max_cost: Largest HQC cost of one submission, or None.
        project: Nexus project name.
        optimisation_level: Level of the pytket compilation pass.
    """

    def __init__(
        self,
        device: str = "H2-1",
        *,
        local: bool = True,
        max_size: int = 14,
        iceberg: bool = True,
        pack: bool = True,
        fill: float = 0.9,
        noise: Noise | None = None,
        submit: bool = False,
        max_cost: float | None = None,
        project: str = "qcbb-labs",
        optimisation_level: int = 2,
    ) -> None:
        super().__init__(max_size, iceberg, pack, fill, submit)
        from pytket.extensions.quantinuum import (
            QuantinuumAPIOffline,
            QuantinuumBackend,
        )

        self.device = device
        self.local = local
        self.noise = noise or Noise()
        self.max_cost = max_cost
        self.project = project
        self.optimisation_level = optimisation_level
        self.backend = QuantinuumBackend(
            device.removesuffix("E"), api_handler=QuantinuumAPIOffline()
        )

    @property
    def qubits(self) -> int:
        """Number of physical qubits of the device."""
        return int(self.backend.backend_info.n_nodes)

    def compile(self, circuits: Sequence[Any]) -> list[Any]:
        """Return the circuits as pytket circuits in the device's native gates."""
        from pytket.extensions.qiskit import qiskit_to_tk

        return [
            self.backend.get_compiled_circuit(
                qiskit_to_tk(qc), optimisation_level=self.optimisation_level
            )
            for qc in circuits
        ]

    def resources(self, circuits: Sequence[Any], shots: int) -> Estimate:
        """Compile circuits and return their resources and estimated fidelity."""
        compiled = self.compile(circuits)
        fidelity, total = 1.0, 0
        for c in compiled:
            kinds = [g.op.type.name for g in c]
            two = sum(k in TWO_QUBIT for k in kinds)
            one = kinds.count("PhasedX")
            reads = kinds.count("Measure")
            total += two
            f = (
                (1 - self.noise.two) ** two
                * (1 - self.noise.one) ** one
                * (1 - self.noise.readout) ** reads
            )
            fidelity = min(fidelity, f)
        return Estimate(
            circuits=len(compiled),
            qubits=max(c.n_qubits for c in compiled),
            two_qubit_gates=total,
            depth=max(c.depth() for c in compiled),
            fidelity=fidelity,
            shots=shots,
        )

    def execute(
        self, circuits: Sequence[Any], shots: int, seed: int | None
    ) -> list[dict[str, Bits]]:
        """Compile and run circuits; return the bits of every register per circuit."""
        compiled = self.compile(circuits)
        if self.local:
            return self._emulate(compiled, circuits, shots, seed)
        return self._nexus(compiled, circuits, shots)

    def _emulate(
        self, compiled: list[Any], circuits: Sequence[Any], shots: int, seed: int | None
    ) -> list[dict[str, Bits]]:
        """Run compiled circuits on Aer with the depolarizing noise model."""
        from pytket.extensions.qiskit import tk_to_qiskit
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        from qiskit_aer.primitives import SamplerV2

        model = NoiseModel()
        model.add_all_qubit_quantum_error(depolarizing_error(self.noise.one, 1), ["r"])
        model.add_all_qubit_quantum_error(
            depolarizing_error(self.noise.two, 2), ["rzz"]
        )
        e = self.noise.readout
        model.add_all_qubit_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]))
        options: dict[str, Any] = {"backend_options": {"noise_model": model}}
        if seed is not None:
            options["run_options"] = {"seed": seed}
        native = [tk_to_qiskit(c) for c in compiled]
        result = SamplerV2(options=options).run(native, shots=shots).result()
        return [
            registers(pub.data, qc) for pub, qc in zip(result, circuits, strict=True)
        ]

    def _nexus(
        self, compiled: list[Any], circuits: Sequence[Any], shots: int
    ) -> list[dict[str, Bits]]:
        """Submit compiled circuits to Nexus and wait for the results."""
        import qnexus as qnx
        from pytket.unit_id import Bit

        project = qnx.projects.get_or_create(name=self.project)
        config = qnx.QuantinuumConfig(device_name=self.device)
        refs = [
            qnx.circuits.upload(circuit=c, name=f"qcbb-{i}", project=project)
            for i, c in enumerate(compiled)
        ]
        cost = qnx.circuits.cost(
            refs,
            n_shots=[shots] * len(refs),
            backend_config=config,
            syntax_checker=f"{self.device.removesuffix('E')}SC",
        )
        log.warning("Nexus cost estimate: %s HQC on %s", cost, self.device)
        if self.max_cost is not None and cost is not None and cost > self.max_cost:
            raise RuntimeError(f"cost {cost} HQC exceeds max_cost = {self.max_cost}")
        job = qnx.start_execute_job(
            programs=refs,
            n_shots=[shots] * len(refs),
            backend_config=config,
            name="qcbb-labs",
            project=project,
            max_cost=[] if self.max_cost is None else [self.max_cost] * len(refs),
        )
        qnx.jobs.wait_for(job)
        results = [r.download_result() for r in qnx.jobs.results(job)]
        return [
            {
                r.name: np.asarray(
                    res.get_shots(cbits=[Bit(r.name, j) for j in range(r.size)]),
                    dtype=np.uint8,
                )
                for r in qc.cregs
            }
            for res, qc in zip(results, circuits, strict=True)
        ]
