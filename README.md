# qcbb-labs

Quantum-classical branch and bound (QCBB) for low autocorrelation binary sequences (LABS,
problem 02 of the QOBLIB benchmark library).

The solver searches a branch and bound tree. Classical bounds prune the tree, and a sampler
(fixed-angle QAOA or simulated annealing) samples every node that is not pruned. The samples give
candidate sequences and choose the next spin to branch on, but they never enter a bound, so the
result is correct for every sampler. A run ends with a proof of optimality, or, if it is stopped
early, with the best sequence found, a certified lower bound and the gap between them.

The method and its proofs are in [docs/algorithm.md](docs/algorithm.md). The QCBB framework is
described in arXiv:2511.19501.

## Install

You need Python 3.12 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The core needs only NumPy and Numba. Optional extras add Qiskit: `.[quantum]` (Qiskit and
Qiskit Aer), `.[ibm]` and `.[quantinuum]` (the IBM and Quantinuum connectors).

Check the install by running the tests:

```bash
.venv/bin/pytest
```

## Solve an instance

```bash
.venv/bin/qcbb labs solve 20 --sampler qaoa --seed 1
```

The output shows the best sequence, its energy E, the lower bound LB, and whether E is proven
optimal:

```
-----+-+++-+--+++--+
E = 26  LB = 26  proven optimal
nodes = 328  sampler calls = 130  0.36 s (sampler 0.19 s)  PDI = 0.29 s
QOBLIB labs020: E = 26 (proven)
```

PDI is the primal-dual integral of the run, a measure of how fast the bounds close. The last line
compares the result with the QOBLIB reference value.

More examples:

```bash
.venv/bin/qcbb labs solve 22 --seed 1                        # simulated annealing (default)
.venv/bin/qcbb labs solve 24 --sampler qaoa --p 1 --seed 1   # QAOA at depth p = 1
.venv/bin/qcbb labs solve 26 --node-limit 1000 --seed 1      # stop early: best E, LB and gap
.venv/bin/qcbb labs instances                                # QOBLIB instances and energies
.venv/bin/qcbb labs qaoa-table 21                            # qubits and gates of the QAOA circuits
```

Main options of `qcbb labs solve n`:

| Option | Meaning | Default |
| --- | --- | --- |
| `--sampler` | `sa` (simulated annealing), `qaoa`, `ibm` or `quantinuum` | `sa` |
| `--p` | QAOA depth | 12 for `qaoa`, 1 for `ibm` and `quantinuum` |
| `--shots` | samples per node | 1000 |
| `--sweeps` | annealing sweeps per sample | 1000 |
| `--node-limit`, `--time-limit` | stop after this many nodes or seconds | none |
| `--seed` | random seed; the same seed gives the same search tree | none |

Run `.venv/bin/qcbb labs solve --help` for all options. The QAOA sampler computes the QAOA state
exactly and handles up to 26 qubits. Only the root node (the full problem) may be larger; it is
then split into its symmetry classes without being sampled, and every other node is sampled.

The QOBLIB submission runs (labs002 to labs030, five seeds each) use:

```bash
.venv/bin/qcbb labs solve 30 --sampler qaoa --p 12 --shots 1000 --seed 1
```

## Use from Python

```python
from qcbb import Settings, solve
from qcbb.labs import LabsAngleSchedule, LabsProblem, SkewSampler
from qcbb.qaoa import Emulator, QaoaSampler

backend = Emulator()
sampler = SkewSampler(QaoaSampler(LabsAngleSchedule(p=12), backend), backend.max_size)
result = solve(LabsProblem(21), sampler, Settings(shots=1000, seed=1))

print(result.ub, result.lb, result.proven)  # best energy, lower bound, proof
print(result.incumbent)  # the best sequence
for record in result.trace:  # the bounds over time
    print(record.seconds, record.ub, record.lb)
```

For simulated annealing use `AnnealingSampler()` from `qcbb.labs` as the sampler. `Settings` also
takes `node_limit` and `time_limit`.

## Layout

```
src/qcbb/          the node loop, sampler interface and command line
src/qcbb/labs/     LABS: energy, bounds, propagation, symmetry, annealing, QAOA angles
src/qcbb/qaoa/     the QAOA sampler, its exact state computation and device connectors
tests/             tests
docs/algorithm.md  the algorithm and its proofs
```

## Credits

This is an independent implementation of the QCBB algorithm described in arXiv:2511.19501. It
contains no code from the QCBB reference implementation for binary linear programs
(https://github.com/david-sipos/qcbb, GPL-3.0).

The fixed QAOA angles in `src/qcbb/labs/data/qaoa_parameters.json` come from QOKit
(https://github.com/jpmorganchase/QOKit) and remain under QOKit's Apache-2.0 license.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Bentax Plus.
