import io
import re
from collections import namedtuple
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms


def str_to_arr(string: str, dtype=np.float64) -> np.ndarray:
    io_string = io.StringIO(string)
    return np.loadtxt(io_string, dtype=dtype)


# Parses *.interp file from orca NEB calculation
class ORCANEBParser:
    _re_split_iter = re.compile(r"Iteration:.+\n")
    _re_split_interp = re.compile(r".+:.+\n")
    NEBPath = namedtuple("NEBPath", ["pathway", "distance", "energy"])

    def __init__(self, filepath: str, units: Literal["au", "ev"] = "au"):
        self.filepath = Path(filepath)
        self.units = units
        self.raw_path = None
        self.interp_path = None
        self._parse_interp_file()

    def _parse_interp_file(self):
        def _parse_pathway(block: str):
            pathway, distance, energy = str_to_arr(block.strip(), dtype=np.float64).T
            if self.units == "ev":
                distance = distance * 0.52917721092
                energy = energy * 27.2114
            return self.NEBPath(pathway, distance, energy)

        data = self.filepath.read_text().strip()
        iter_blocks = self._re_split_iter.split(data)[1:]
        raw_path, interp_path = [], []

        for block in iter_blocks:
            _, raw, interp = self._re_split_interp.split(block)
            raw = _parse_pathway(raw)
            raw_path.append(raw)
            interp = _parse_pathway(interp)
            interp_path.append(interp)

        self.raw_path = raw_path
        self.interp_path = interp_path

    def plot(self, idx: int = -1, ax=None, **kwargs):
        color = kwargs.pop("color", "k")
        if ax is None:
            fig, ax = plt.subplots()
        raw_path = self.raw_path[idx]
        interp_path = self.interp_path[idx]
        ax.set_xlabel("Reaction Coordinate")
        ax.set_ylabel("Energy ({})".format("eV" if self.units == "ev" else "Ha"))
        ax.plot(raw_path.pathway, raw_path.energy, "o", color=color, **kwargs)
        ax.plot(interp_path.pathway, interp_path.energy, "-", color=color, **kwargs)
        return ax


def parse_matrix(s: str):
    lines = (line.strip() for line in s.splitlines())
    try:
        n_rows = int(next(lines))
    except StopIteration as e:
        raise ValueError("Empty file") from e
    except ValueError as e:
        raise ValueError("First line is not an integer") from e

    n_processed_cols = 0
    matrix = []
    while n_processed_cols < n_rows:
        n_cols = len(next(lines).split())
        matrix_fragment = []
        for _ in range(n_rows):
            matrix_elems = next(lines).split()[1:]
            matrix_elems = [float(i) for i in matrix_elems]
            matrix_fragment.append(matrix_elems)
        matrix_fragment = np.array(matrix_fragment)
        matrix.append(matrix_fragment)
        n_processed_cols += n_cols
    matrix = np.hstack(matrix)
    return matrix


def parse_geom_from_out(txt) -> Atoms:
    lines = txt.strip().splitlines()
    symbols = []
    positions = []
    for line in lines[2:-2]:
        if "----" in line or not line:
            continue
        s, x, y, z = line.strip().split()
        symbols.append(s)
        positions.append([float(x), float(y), float(z)])

    return Atoms(symbols=symbols, positions=positions)
