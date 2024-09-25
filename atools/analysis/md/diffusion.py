import warnings
from typing import List, Tuple, Union

import numpy as np
from ase import Atoms, units
from numba import jit


@jit(nopython=True)
def apply_mic(dr, cell, inv_cell):
    """
    Apply minimum image convention to a displacement vector.

    :param dr: Displacement vector
    :param cell: Cell matrix
    :param inv_cell: Inverse of the cell matrix
    :return: Displacement vector after applying minimum image convention
    """
    s = np.dot(dr, inv_cell)
    s -= np.round(s)
    return np.dot(s, cell)


@jit(nopython=True)
def unwrap_positions(positions: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    """
    Unwrap positions accounting for periodic boundary conditions in non-cubic cells.

    :param positions: Array of positions (n_frames, n_atoms, 3)
    :param cell: Cell matrix (3, 3)
    :param inv_cell: Inverse of the cell matrix (3, 3)
    :return: Unwrapped positions
    """
    n_frames, n_atoms, _ = positions.shape
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]

    for i in range(1, n_frames):
        diff = positions[i] - positions[i - 1]
        for j in range(n_atoms):
            diff[j] = apply_mic(diff[j], cell, inv_cell)
        unwrapped[i] = unwrapped[i - 1] + diff

    return unwrapped


@jit(nopython=True)
def calculate_msd_array(unwrapped: np.ndarray, max_delta: int) -> np.ndarray:
    """
    Calculate MSD array using multiple time origins.

    :param unwrapped: Unwrapped positions
    :param max_delta: Maximum time difference to consider
    :return: MSD array
    """
    n_frames, n_atoms, _ = unwrapped.shape
    msd = np.zeros(max_delta)
    count = np.zeros(max_delta)

    for i in range(n_frames):
        for j in range(i + 1, min(i + max_delta + 1, n_frames)):
            dt = j - i
            disp = unwrapped[j] - unwrapped[i]
            msd[dt] += np.sum(disp**2)
            count[dt] += n_atoms

    return msd / count


def calculate_msd(
    trajectory: List[Atoms],
    timestep: float = 1.0,
    elements: Union[str, List[str]] = None,
    max_delta: int = None,
    dimensions: Union[str, List[int]] = "xyz",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Mean Square Displacement (MSD) for given elements in the trajectory.

    :param trajectory: List of ASE Atoms objects representing the MD trajectory
    :param elements: Symbol(s) of the element(s) to calculate MSD for (if None, calculate for all atoms)
    :param max_delta: Maximum time difference to consider (if None, use all frames)
    :param dimensions: Dimensions to include in MSD calculation ('x', 'y', 'z', or list of indices)
    :return: Tuple of (time differences, MSD values)
    """
    n_frames = len(trajectory)
    n_atoms = len(trajectory[0])

    if not all(len(atoms) == n_atoms for atoms in trajectory):
        raise ValueError("All frames must have the same number of atoms.")

    # Get cell and its inverse for the first frame
    cell = trajectory[0].get_cell().array
    inv_cell = np.linalg.inv(cell)

    # Check if cell changes over time
    if not all(np.allclose(atoms.get_cell(), cell) for atoms in trajectory):
        warnings.warn("Cell changes over time. Using the initial cell for unwrapping.", stacklevel=2)

    # Determine which atoms to include
    if elements is None:
        atom_indices = slice(None)
    else:
        if isinstance(elements, str):
            elements = [elements]
        atom_indices = [atom.index for atom in trajectory[0] if atom.symbol in elements]

    # Determine which dimensions to include
    if isinstance(dimensions, str):
        dim_indices = [i for i, d in enumerate("xyz") if d in dimensions.lower()]
    else:
        dim_indices = dimensions

    # Extract positions
    positions = np.array([atoms.get_positions() for atoms in trajectory])

    # Unwrap trajectories
    unwrapped_positions = unwrap_positions(positions, cell, inv_cell)

    # Set maximum time difference
    if max_delta is None:
        max_delta = n_frames - 1
    else:
        max_delta = min(max_delta, n_frames - 1)

    # Calculate MSD
    msd = calculate_msd_array(unwrapped_positions[:, atom_indices][:, :, dim_indices], max_delta)
    msd[0] = 0
    time_diffs = np.arange(max_delta)

    return time_diffs * timestep, msd


def calculate_diffusion_coefficient(time, msd, skip_steps=0):
    """
    Calculate the diffusion coefficient from the MSD curve.

    :param time: Time array
    :param msd: MSD array
    :param skip_steps: Number of initial steps to skip
    :return: Diffusion coefficient (m^2/s)
    """
    time = time[skip_steps:]
    msd = msd[skip_steps:]
    slope = np.polyfit(time, msd, 1)[0]  # Å^2/ps
    return slope / 6 * 1e-8  # Å^2/ps to m^2/s


def calculate_ionic_conductivity(
    atoms,
    element,
    temperature,
    ionic_charge,
    *,
    time=None,
    msd=None,
    skip_steps=0,
    diffusion_coefficient=None,
):
    """
    Calculate the ionic conductivity from the MSD curve.

    :param atoms: ASE Atoms object
    :param time: Time array
    :param msd: MSD array
    :param temperature: Temperature (K)
    :param element: Element symbol
    :param ionic_charge: Ionic charge
    :return: Ionic conductivity (mS/cm)
    """
    has_msd = time is not None and msd is not None
    has_diffusion = diffusion_coefficient is not None
    if not (has_msd or has_diffusion):
        raise ValueError("Either time and msd or diffusion_coefficient must be provided.")
    if has_msd:
        if len(time) != len(msd):
            raise ValueError("Time and MSD arrays must have the same length.")
        if not (time is not None and msd is not None):
            raise ValueError("Time and MSD arrays must be provided together.")

    N = atoms.symbols.count(element)
    q = ionic_charge * 1.602176634 * 10**-19  # C
    V = atoms.get_volume() * 1e-30  # m^3
    if has_msd:
        D = calculate_diffusion_coefficient(time, msd, skip_steps)  # m^2/s
    else:
        D = diffusion_coefficient
    kB = units.kB / units.J  # Boltzmann constant (J/K)
    sigma = (N * q**2) / (V * kB * temperature) * D  # S/m
    return sigma * 10  # mS/cm
