"""IBM Quantum devices (needs the `ibm` extra, qiskit-ibm-runtime).

One choice per hardware problem, all available in Qiskit Runtime:

- two-qubit gate count: fractional gates on Heron devices (native RZZ and RX,
  `use_fractional_gates=True`), so a ZZ term is one RZZ instead of two CNOTs;
- connectivity and qubit quality: the preset pass manager at level 3, whose
  layout and routing use the device's error rates;
- idle and crosstalk errors: dynamical decoupling with the XY4 sequence;
- readout errors: measurement twirling (TREX). Gate twirling is not used,
  since it does not work with fractional gates. Readout correction (M3) and
  ZNE/PEC are not used: they correct expectation values, not samples, and a
  misread bitstring is still a valid assignment whose energy is recomputed;
- error detection: Iceberg (optional, off by default on heavy-hex, where its
  all-to-all checks cost SWAPs);
- throughput: spatial circuit packing.

A backend name starting with "fake_" (e.g. "fake_marrakesh") selects the
local emulation of that device with its calibrated noise model (Aer); runtime
options do not apply there. Real devices use the account saved with
`QiskitRuntimeService.save_account`; no credentials pass through this module.
"""

from collections.abc import Sequence
from typing import Any

from qcbb.qaoa.backend import Estimate
from qcbb.qaoa.circuits import Bits, registers
from qcbb.qaoa.device import Device


class IbmDevice(Device):
    """Run QAOA jobs on an IBM Quantum device or its local emulation.

    Args:
        backend: Backend name ("ibm_fez", "fake_marrakesh", ...) or a qiskit
            `BackendV2`.
        max_size: Largest logical F accepted.
        iceberg: Encode jobs in the Iceberg code and post-select.
        pack: Pack several circuits into one (real devices only).
        fill: Share of the device's qubits used when packing.
        decoupling: Enable dynamical decoupling (XY4) on real devices.
        submit: Allow submission to a real device.
        optimization_level: Level of the preset pass manager.
    """

    def __init__(
        self,
        backend: str | Any = "fake_marrakesh",
        *,
        max_size: int = 10,
        iceberg: bool = False,
        pack: bool = True,
        fill: float = 0.8,
        decoupling: bool = True,
        submit: bool = False,
        optimization_level: int = 3,
    ) -> None:
        super().__init__(max_size, iceberg, pack, fill, submit)
        self.backend = _backend(backend) if isinstance(backend, str) else backend
        from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

        self.local = isinstance(self.backend, FakeBackendV2)
        self.decoupling = decoupling
        self.optimization_level = optimization_level
        self._pm: Any = None

    @property
    def qubits(self) -> int:
        """Number of physical qubits of the device."""
        return int(self.backend.num_qubits)

    def compile(self, circuits: Sequence[Any]) -> list[Any]:
        """Return the circuits transpiled to the device's gates and layout."""
        if self._pm is None:
            from qiskit.transpiler import generate_preset_pass_manager

            self._pm = generate_preset_pass_manager(
                optimization_level=self.optimization_level, backend=self.backend
            )
        return list(self._pm.run(list(circuits)))

    def resources(self, circuits: Sequence[Any], shots: int) -> Estimate:
        """Compile circuits and return their resources and estimated fidelity."""
        compiled = self.compile(circuits)
        target = self.backend.target
        fidelity, two = 1.0, 0
        for qc in compiled:
            f = 1.0
            for inst in qc.data:
                qargs = tuple(qc.find_bit(q).index for q in inst.qubits)
                two += len(qargs) == 2
                f *= 1 - _error(target, inst.operation.name, qargs)
            fidelity = min(fidelity, f)
        return Estimate(
            circuits=len(compiled),
            qubits=max(len(_active(qc)) for qc in compiled),
            two_qubit_gates=two,
            depth=max(qc.depth() for qc in compiled),
            fidelity=fidelity,
            shots=shots,
        )

    def execute(
        self, circuits: Sequence[Any], shots: int, seed: int | None
    ) -> list[dict[str, Bits]]:
        """Compile and run circuits; return the bits of every register per circuit."""
        from qiskit_ibm_runtime import SamplerV2

        options: dict[str, Any]
        if self.local:
            options = {} if seed is None else {"simulator": {"seed_simulator": seed}}
        else:
            options = {
                "dynamical_decoupling": {
                    "enable": self.decoupling,
                    "sequence_type": "XY4",
                },
                "twirling": {"enable_gates": False, "enable_measure": True},
            }
        sampler = SamplerV2(mode=self.backend, options=options)
        result = sampler.run(self.compile(circuits), shots=shots).result()
        return [
            registers(pub.data, qc) for pub, qc in zip(result, circuits, strict=True)
        ]


def _backend(name: str) -> Any:
    """Return a fake backend for "fake_..." names, else the real device."""
    if name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        return getattr(fake_provider, "Fake" + name[5:].capitalize())()
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService().backend(name, use_fractional_gates=True)


def _error(target: Any, name: str, qargs: tuple[int, ...]) -> float:
    """Return the device's error rate of an instruction, 0 if unknown."""
    props = target[name].get(qargs) if name in target else None
    return 0.0 if props is None or props.error is None else float(props.error)


def _active(qc: Any) -> set[int]:
    """Return the physical qubits that a compiled circuit acts on."""
    return {
        qc.find_bit(q).index
        for inst in qc.data
        if inst.operation.name not in ("barrier", "delay")
        for q in inst.qubits
    }
