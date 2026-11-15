"""Fixed-angle QAOA as the sampler, with the backends that execute its circuits.

`QaoaSampler(angles, backend)` is a `Sampler`. The backends are

- `Emulator`: the exact noiseless emulation of the circuit (numba, CPU);
- `IbmDevice`: IBM Quantum devices or their local noisy emulation;
- `QuantinuumDevice`: Quantinuum H-series via Nexus, or a local emulation.

Shared by the gate-based devices: `circuits` (qiskit QAOA circuits),
`iceberg` (error detection) and `device` (the run flow and circuit packing).
The IBM and Quantinuum SDKs are optional extras, imported only when a device
is created.
"""

from qcbb.qaoa.backend import AngleSchedule, Backend, Estimate, QaoaJob, QaoaSampler
from qcbb.qaoa.circuits import cnots_per_layer, program, qaoa_circuit, term_counts
from qcbb.qaoa.emulator import Emulator, diagonal, qaoa_state
from qcbb.qaoa.ibm import IbmDevice
from qcbb.qaoa.quantinuum import Noise, QuantinuumDevice

__all__ = [
    "AngleSchedule",
    "Backend",
    "Emulator",
    "Estimate",
    "IbmDevice",
    "Noise",
    "QaoaJob",
    "QaoaSampler",
    "QuantinuumDevice",
    "cnots_per_layer",
    "diagonal",
    "program",
    "qaoa_circuit",
    "qaoa_state",
    "term_counts",
]
