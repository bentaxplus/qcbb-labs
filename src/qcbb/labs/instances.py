"""QOBLIB LABS instances and their reference energies.

The table lists labs002 to labs100: the length n, the best known energy, and
whether that energy is proven optimal. Source: QOBLIB problem 02,
https://zib-aopt.github.io/QOBLIB/problem/02/.
"""

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class Instance:
    """One QOBLIB LABS instance.

    Attributes:
        name: QOBLIB name, e.g. "labs030".
        n: Sequence length.
        energy: Best known energy.
        proven: True if the energy is proven optimal.
    """

    name: str
    n: int
    energy: int
    proven: bool


@cache
def load() -> dict[int, Instance]:
    """Return all instances keyed by n."""
    text = resources.files("qcbb.labs").joinpath("data/qoblib_labs.csv").read_text()
    rows = csv.DictReader(text.splitlines())
    table = (
        Instance(r["name"], int(r["n"]), int(r["energy"]), r["proven"] == "1")
        for r in rows
    )
    return {inst.n: inst for inst in table}


def get(key: int | str) -> Instance:
    """Return one instance by length or by name.

    Args:
        key: The length n, or a name such as "labs030".

    Returns:
        The instance.

    Raises:
        KeyError: If no such instance exists.
    """
    n = int(key[4:]) if isinstance(key, str) else key
    return load()[n]
