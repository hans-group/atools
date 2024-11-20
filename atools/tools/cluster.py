import itertools
from typing import Set

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from matscipy.neighbours import neighbour_list
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components


class MolecularCluster:
    """
    A class to extract molecular clusters from periodic systems based on covalent bonds.

    Parameters
    ----------
    atoms : ase.Atoms
        The periodic system to analyze
    tol : float, optional
        Tolerance for covalent bond detection (default: 0.15)
    """

    def __init__(self, atoms: Atoms, tol: float = 0.15):
        self.atoms = atoms.copy()
        self.tol = tol
        self._build_adjacency_matrix()
        self._find_molecules()
        self._unwrap_molecules()

    def _unwrap_molecules(self) -> None:
        """
        Unwrap molecules that might be split across periodic boundaries.
        This modifies self.atoms in-place.
        """
        positions = self.atoms.get_positions()
        cell = self.atoms.get_cell()

        # Get bonded pairs and their shifts
        i, j, S = neighbour_list("ijS", self.atoms, self.covalent_bond_lengths)

        # Process each molecule
        for mol_idx in np.unique(self.molecule_idx):
            mol_mask = self.molecule_idx == mol_idx
            mol_atoms = np.where(mol_mask)[0]

            # Start from first atom in molecule
            start_atom = mol_atoms[0]
            processed = {start_atom}
            to_process = [(start_atom, np.zeros(3))]  # (atom_idx, accumulated_shift)

            # Store accumulated shifts for each atom
            shifts = np.zeros((len(self.atoms), 3))

            # Breadth-first search through bonds
            while to_process:
                current, current_shift = to_process.pop(0)

                # Find all bonds involving current atom
                bonds_i = i == current
                bonds_j = j == current

                # Process neighbors through both i->j and j->i bonds
                for idx, shift in zip(j[bonds_i], S[bonds_i]):
                    if idx not in processed and self.molecule_idx[idx] == mol_idx:
                        new_shift = current_shift + shift
                        shifts[idx] = new_shift
                        processed.add(idx)
                        to_process.append((idx, new_shift))

                for idx, shift in zip(i[bonds_j], -S[bonds_j]):
                    if idx not in processed and self.molecule_idx[idx] == mol_idx:
                        new_shift = current_shift + shift
                        shifts[idx] = new_shift
                        processed.add(idx)
                        to_process.append((idx, new_shift))

            # Apply accumulated shifts to molecule atoms
            positions[mol_mask] += np.dot(shifts[mol_mask], cell)

        self.atoms.set_positions(positions)

    def _build_adjacency_matrix(self) -> None:
        """Build adjacency matrix based on covalent bonds."""
        # Create dictionary of covalent bond lengths
        unique_z = set(self.atoms.get_atomic_numbers())
        self.covalent_bond_lengths = {}
        for z1, z2 in itertools.product(unique_z, unique_z):
            self.covalent_bond_lengths[(z1, z2)] = (covalent_radii[z1] + covalent_radii[z2]) * (1 + self.tol)

        # Get bonded pairs
        i, j = neighbour_list("ij", self.atoms, self.covalent_bond_lengths)
        N = len(self.atoms)
        self.adj = coo_array((np.ones_like(i), (i, j)), shape=(N, N))

    def _find_molecules(self) -> None:
        """Identify molecules using connected components analysis."""
        self.n_molecules, self.molecule_idx = connected_components(self.adj)

    def _get_molecule_atoms(self, mol_idx: int) -> Set[int]:
        """Get atom indices belonging to a specific molecule."""
        return set(np.where(self.molecule_idx == mol_idx)[0])

    def _get_molecules_within_cutoff(self, center_idx: int, cutoff: float) -> Set[int]:
        """
        Find molecules that have at least one atom within the cutoff distance.

        Parameters
        ----------
        center_idx : int
            Index of the center atom
        cutoff : float
            Cutoff radius for including molecules

        Returns
        -------
        Set[int]
            Set of molecule indices that should be included
        """
        # Get neighbors within cutoff
        i, j, S = neighbour_list("ijS", self.atoms, cutoff)
        mask = i == center_idx
        neighbor_indices = j[mask]

        # Find molecules containing these atoms
        molecules_to_include = set()
        for idx in neighbor_indices:
            molecules_to_include.add(self.molecule_idx[idx])

        return molecules_to_include

    def _adjust_molecular_positions(self, mol_indices: Set[int], center_idx: int) -> Atoms:
        """
        Adjust positions of molecules to maintain molecular integrity across PBC.

        Parameters
        ----------
        mol_indices : Set[int]
            Set of molecule indices to include in the cluster
        center_idx : int
            Index of the center atom to use as reference

        Returns
        -------
        ase.Atoms
            New Atoms object with adjusted positions
        """
        # Create a new Atoms object for the cluster
        cluster_atoms = self.atoms.copy()
        cluster_atoms.set_pbc(False)
        positions = cluster_atoms.get_positions()
        cell = self.atoms.get_cell()

        # Get position of center atom as reference
        center_pos = positions[center_idx]

        # First adjust positions within each molecule
        for mol_idx in mol_indices:
            mol_atoms = self._get_molecule_atoms(mol_idx)

            # Get bonded pairs within the molecule
            i, j, S = neighbour_list("ijS", self.atoms, self.covalent_bond_lengths)
            mask = np.isin(i, list(mol_atoms)) & np.isin(j, list(mol_atoms))
            mol_i, mol_j, mol_S = i[mask], j[mask], S[mask]

            # Adjust positions to keep molecule together
            for idx_i, idx_j, offset in zip(mol_i, mol_j, mol_S):
                displacement = np.dot(offset, cell)
                if np.any(offset != 0):
                    positions[idx_j] += displacement

        # Then adjust entire molecules relative to center
        for mol_idx in mol_indices:
            mol_atoms = list(self._get_molecule_atoms(mol_idx))
            mol_center = positions[mol_atoms].mean(axis=0)

            # Find the periodic image of the molecule closest to the center atom
            diff = mol_center - center_pos
            shifts = -np.round(diff / cell.diagonal()) @ cell

            # Apply shift to all atoms in the molecule
            positions[mol_atoms] += shifts

        cluster_atoms.set_positions(positions)
        return cluster_atoms

    def get_cluster(self, center_idx: int, cutoff: float) -> Atoms:
        """
        Extract molecular cluster around a center atom.

        Parameters
        ----------
        center_idx : int
            Index of the center atom
        cutoff : float
            Cutoff radius for including molecules

        Returns
        -------
        ase.Atoms
            Cluster of molecules within cutoff distance
        """
        # Find molecules to include
        molecules_to_include = self._get_molecules_within_cutoff(center_idx, cutoff)

        # Adjust positions to maintain molecular integrity
        cluster = self._adjust_molecular_positions(molecules_to_include, center_idx)

        # Create mask for atoms to keep
        mask = np.zeros(len(self.atoms), dtype=bool)
        for mol_idx in molecules_to_include:
            mask |= self.molecule_idx == mol_idx

        # Return only the relevant atoms
        return cluster[mask]
