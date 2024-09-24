"""Calculate mean square displacement (MSD) of a given element in a trajectory.
Use torch.vmap to accelerate computation.
"""

from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers

from ..space import PeriodicSpace


def _check_images(images: List[Atoms]) -> None:
    """Check if images is a list of ase.Atoms objects of same system.

    Args:
        images (List[Atoms]): List of ase.Atoms objects.

    Raises:
        TypeError: If images is not a list of ase.Atoms objects.
        ValueError: If images do not contain same model system.
    """
    _type_err_msg = "images must be a list of ase.Atoms objects"
    if not isinstance(images, list):
        raise TypeError(_type_err_msg)
    if not all(isinstance(image, Atoms) for image in images):
        raise TypeError(_type_err_msg)

    atoms_0 = images[0]
    if not all(len(image) == len(atoms_0) for image in images):
        raise ValueError("All images must have the same number of atoms")

    cond1 = all(np.allclose(a.numbers, atoms_0.numbers) for a in images)
    cond2 = all(np.allclose(a.cell.array, atoms_0.cell.array) for a in images)
    if not (cond1 and cond2):
        raise ValueError("All images must contain same model system")


def mean_square_displacement(images: List[Atoms], element: str, dt: float) -> np.ndarray:
    """Calculate mean square displacement of a given element in a trajectory.

    Args:
        images (List[Atoms]): List of ase.Atoms objects.
        element (str): Element symbol.
        dt (float): Time step between snapshots.

    Raises:
        TypeError: If images is not a list of ase.Atoms objects.
        ValueError: If images do not contain same model system.

    Returns:
        (np.ndarray, np.ndarray): MSD of the element.
    """
    _check_images(images)

    # Get positions of selected element
    elem_idx = np.where(images[0].numbers == atomic_numbers[element])[0]
    all_pos_elem = []
    for atoms in images:
        all_pos_elem.append(atoms.get_scaled_positions()[elem_idx])
    all_pos_elem = torch.from_numpy(np.stack(all_pos_elem))  # (n_images, n_elem, 3)

    # R(i) - R(i-1) for i = 1, ..., n_images
    all_pos_elem_i = all_pos_elem[:-1]
    all_pos_elem_j = all_pos_elem[1:]
    space = PeriodicSpace(
        torch.from_numpy(images[0].cell.array),
        fractional_coordinates=True,
    )
    pairwise_displacement_fn = torch.func.vmap(torch.func.vmap(space.displacement))
    disp_ij = pairwise_displacement_fn(all_pos_elem_i, all_pos_elem_j)
    # Disp(i) = sum_{j=1}^{i} (R(j) - R(j-1))
    disp_t = torch.cumsum(disp_ij, dim=0)
    # Compute average over elements
    msd = torch.mean(disp_t.norm(dim=-1) ** 2, dim=-1)
    msd = torch.cat([torch.zeros((1,)), msd])

    # construct time array
    time = np.arange(len(images)) * dt
    return time, msd.numpy()
