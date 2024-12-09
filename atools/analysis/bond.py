import itertools
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from matscipy.neighbours import neighbour_list


class NeighborList(NamedTuple):
    i: np.ndarray  # (n_pairs,)
    j: np.ndarray  # (n_pairs,)
    d: np.ndarray  # (n_pairs,)
    D: np.ndarray  # (n_pairs, 3)
    S: np.ndarray  # (n_pairs, 3)

    @classmethod
    def from_atoms(cls, atoms: Atoms, cutoff: float | dict[tuple[str, str], float]) -> "NeighborList":
        if np.linalg.det(atoms.get_cell().array) < 1e-8:
            centroid = atoms.get_positions().mean(axis=0)
            radius = np.linalg.norm(atoms.get_positions() - centroid, axis=1).max()
            celldim = 10 * radius
            cell = np.eye(3) * celldim
            atoms.cell = cell
        i, j, d, D, S = neighbour_list("ijdDS", atoms, cutoff)
        return cls(i, j, d, D, S)

    def apply_mask(self, mask: np.ndarray) -> "NeighborList":
        return NeighborList(self.i[mask], self.j[mask], self.d[mask], self.D[mask], self.S[mask])

    def to_directed(self) -> "NeighborList":
        mask = self.i < self.j
        return self.apply_mask(mask)


class BondAnalyzer:
    def __init__(self, atoms: Atoms, bond_tol: float = 0.2, extra_cutoff: dict[tuple[str, str], float] = None):
        self.atoms = atoms
        self.bond_tol = bond_tol
        self.nbrs = NeighborList.from_atoms(atoms, self._get_covalent_cutoffs(extra_cutoff))

    @property
    def n_bonds(self):
        "Number of bonds."
        return len(self.nbrs.i / 2)

    def get_species_bonds(self, s1: str, s2: str, return_only_ij: bool = False):
        """Find all bonds between two species."""
        z1 = atomic_numbers[s1]
        z2 = atomic_numbers[s2]
        nbrs = self.nbrs.to_directed()
        mask_1 = np.logical_and(self.atoms.numbers[nbrs.i] == z1, self.atoms.numbers[nbrs.j] == z2)
        mask_2 = np.logical_and(self.atoms.numbers[nbrs.i] == z2, self.atoms.numbers[nbrs.j] == z1)
        mask = np.logical_or(mask_1, mask_2)
        nbrs = nbrs.apply_mask(mask)
        if return_only_ij:
            return [(i, j) for i, j in zip(nbrs.i, nbrs.j, strict=True)]
        return nbrs

    def is_bonded(self, i: int, j: int) -> bool:
        """Check if two atoms are bonded."""
        if i > j:
            i, j = j, i
        return (i, j) in zip(self.nbrs.i, self.nbrs.j, strict=True)

    def get_bond_length(self, i: int, j: int) -> float:
        """Get bond lengths between atom i and a list of atoms js."""
        if not self.is_bonded(i, j):
            return np.nan
        mask = np.logical_and(self.nbrs.i == i, self.nbrs.j == j)
        return self.nbrs.d[mask].item()

    def _get_covalent_cutoffs(self, extra_cutoff: dict[tuple[str, str], float] = None) -> dict[tuple[str, str], float]:
        extra_cutoff = extra_cutoff or {}
        unique_elems = list(set(self.atoms.get_chemical_symbols()))
        unique_elems.sort(key=lambda x: atomic_numbers[x])
        cutoffs = {}
        for e_i, e_j in itertools.combinations_with_replacement(unique_elems, 2):
            z_i = atomic_numbers[e_i]
            z_j = atomic_numbers[e_j]
            if z_i > z_j:
                pair = (e_j, e_i)
            else:
                pair = (e_i, e_j)
            cutoffs[pair] = (covalent_radii[z_i] + covalent_radii[z_j]) * (1 + self.bond_tol)
        cutoffs.update(extra_cutoff)
        return cutoffs
