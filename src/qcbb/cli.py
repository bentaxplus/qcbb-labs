"""Command line interface.

qcbb labs instances          list the QOBLIB reference table
qcbb labs solve N            proven optimum or certified gap
qcbb labs qaoa-table N       qubits, Ising terms and CNOTs per QAOA layer

The sampler is simulated annealing (`--sampler sa`) or fixed-angle QAOA on a
backend: the exact noiseless emulation (`qaoa`), IBM Quantum (`ibm`) or
Quantinuum (`quantinuum`). IBM devices named "fake_..." and Quantinuum
without `--remote` are local noisy emulations; a real device runs only with
`--submit`. One sampler serves every node of a run, with 1,000 shots per node
by default; a Hamiltonian with more qubits than the backend takes is not
sampled. For odd n the master is also sampled through its skew-symmetric
restriction (`--no-skew` turns this off). `solve` prints the sampler time
and the primal-dual integral of the run.
"""

import argparse

from qcbb import Sampler, Settings, solve
from qcbb.labs import (
    AnnealingSampler,
    LabsAngleSchedule,
    LabsProblem,
    NodeHamiltonian,
    SkewHamiltonian,
    SkewSampler,
    instances,
)
from qcbb.problem import Hamiltonian, Spins
from qcbb.qaoa import (
    Backend,
    Emulator,
    IbmDevice,
    QaoaSampler,
    QuantinuumDevice,
    cnots_per_layer,
    term_counts,
)

SAMPLERS = ("sa", "qaoa", "ibm", "quantinuum")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run a command."""
    parser = argparse.ArgumentParser(prog="qcbb")
    problems = parser.add_subparsers(dest="problem", required=True)
    labs = problems.add_parser("labs", help="low autocorrelation binary sequences")
    sub = labs.add_subparsers(dest="command", required=True)
    sub.add_parser("instances", help="list QOBLIB LABS instances")
    table = sub.add_parser("qaoa-table", help="QAOA resources of the Hamiltonians")
    table.add_argument("n", type=int, help="sequence length")
    table.add_argument("--boundary", type=int, default=7)
    cmd = sub.add_parser("solve")
    cmd.add_argument("n", type=int, help="sequence length")
    cmd.add_argument("--sampler", choices=SAMPLERS, default="sa")
    cmd.add_argument("--shots", type=int, default=1000, help="shots per node")
    cmd.add_argument("--sweeps", type=int, default=1000)
    cmd.add_argument("--p", type=int, default=None, help="12 qaoa, 1 devices")
    cmd.add_argument("--device", default=None, help="IBM or Quantinuum device")
    cmd.add_argument("--remote", action="store_true", help="Quantinuum via Nexus")
    cmd.add_argument("--submit", action="store_true", help="allow a real device")
    cmd.add_argument("--iceberg", action=argparse.BooleanOptionalAction, default=None)
    cmd.add_argument(
        "--no-skew", action="store_true", help="QAOA, odd n: no skew restriction"
    )
    cmd.add_argument("--boundary", type=int, default=7)
    cmd.add_argument("--node-limit", type=int, default=None)
    cmd.add_argument("--time-limit", type=float, default=None, help="seconds")
    cmd.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command == "instances":
        for inst in instances.load().values():
            print(f"{inst.name}  n={inst.n:3d}  E={inst.energy:4d}  {_status(inst)}")
        return
    problem = LabsProblem(args.n, boundary=args.boundary)
    if args.command == "qaoa-table":
        _qaoa_table(problem)
        return

    r = solve(
        problem,
        _sampler(args),
        Settings(args.shots, args.node_limit, args.time_limit, args.seed),
    )
    print(_signs(r.incumbent))
    status = "proven optimal" if r.proven else f"gap {r.gap:.1%}"
    print(f"E = {r.ub}  LB = {r.lb}  {status}")
    print(
        f"nodes = {r.nodes}  sampler calls = {r.sampler_calls}  {r.seconds:.2f} s"
        f" (sampler {r.sampler_seconds:.2f} s)  PDI = {r.pdi:.2f} s"
    )
    ref = instances.load().get(args.n)
    if ref is not None:
        print(f"QOBLIB {ref.name}: E = {ref.energy} ({_status(ref)})")


def _sampler(args: argparse.Namespace) -> Sampler:
    """Return the sampler chosen on the command line."""
    if args.sampler == "sa":
        return AnnealingSampler(sweeps=args.sweeps)
    backend = _backend(args)
    p = args.p or (12 if args.sampler == "qaoa" else 1)
    qaoa = QaoaSampler(LabsAngleSchedule(p), backend)
    if args.no_skew:
        return qaoa
    return SkewSampler(qaoa, backend.max_size)


def _backend(args: argparse.Namespace) -> Backend:
    """Return the QAOA backend chosen on the command line."""
    options = {} if args.iceberg is None else {"iceberg": args.iceberg}
    if args.sampler == "ibm":
        return IbmDevice(args.device or "fake_marrakesh", submit=args.submit, **options)
    if args.sampler == "quantinuum":
        return QuantinuumDevice(
            args.device or "H2-1",
            local=not args.remote,
            submit=args.submit,
            **options,
        )
    return Emulator()


def _qaoa_table(problem: LabsProblem) -> None:
    """Print qubits, Ising term counts and CNOTs per QAOA layer."""
    rows: list[tuple[str, Hamiltonian]] = [
        ("master", NodeHamiltonian.master(problem.n))
    ]
    if problem.n % 2:
        rows.append(("skew", SkewHamiltonian(problem.n)))
    roots = problem.class_roots()
    rows.append((f"class root (1 of {len(roots)})", problem.hamiltonian(roots[0])))
    print("Hamiltonian              qubits      Z     ZZ    ZZZ   ZZZZ CNOT/layer")
    for name, h in rows:
        c = term_counts(h)
        print(
            f"{name:24s} {h.size:6d} {c[1]:6d} {c[2]:6d} {c[3]:6d} {c[4]:6d}"
            f" {cnots_per_layer(h):10d}"
        )


def _signs(x: Spins) -> str:
    """Return a sequence as a string of + and -."""
    return "".join("+" if v > 0 else "-" for v in x.tolist())


def _status(inst: instances.Instance) -> str:
    """Return the status word of a reference energy."""
    return "proven" if inst.proven else "best known"
