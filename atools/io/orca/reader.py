import io
import re
import warnings
from pathlib import Path

import ase.io
import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator

from .parsers import parse_geom_from_out, parse_matrix


def find_all_between(text, re_start, re_end):
    """
    find all text between two regular expressions
    """

    def find_between(text, re_start, re_end):
        """
        find text between two regular expressions
        """
        start = re.search(re_start, text).start()
        end = re.search(re_end, text).end()
        return start, end

    while True:
        try:
            # print(text)
            start, end = find_between(text, re_start, re_end)
            yield text[start:end].strip()
            text = text[end:]

        except AttributeError:
            break


def read_orca_hessian(hess_file: str, symmetrize: bool = False):
    with open(hess_file, "r") as f:
        s = f.readlines()
        s = [i.strip() for i in s]
    start_line = s.index("$hessian")
    s = "\n".join(s[start_line + 1 :])  # noqa
    hess = parse_matrix(s)
    if symmetrize:
        # check max error and raise error if larger than 1e-3
        max_error = np.max(np.abs(hess - hess.T))
        if max_error > 1e-2:
            raise ValueError(f"Symmetrization error: max error = {max_error}")
        hess = (hess + hess.T) / 2
    return hess


def read_orca_energy(out_txt: str):
    re_energy = r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+.?\d+)"
    energy = float(re.search(re_energy, out_txt).group(1)) * units.Ha
    return energy


def read_orca_force(out_txt: str):
    re_start = r"ORCA SCF GRADIENT CALCULATION"
    re_end = r"Norm of the cartesian gradient"
    chunks = find_all_between(out_txt, re_start, re_end)
    forces_list = []
    for chunk in chunks:
        re_arr = re.compile(r"\d\s+\w+\s+:\s+([-+]?\d*\.?\d+|\d+)\s+([-+]?\d*\.?\d+|\d+)\s+([-+]?\d*\.?\d+|\d+)")
        grad_str = re_arr.findall(chunk)
        grad = []
        for i in grad_str:
            grad.append([float(j) for j in i])
        grad = np.array(grad)
        forces = -np.array(grad) * units.Hartree / units.Bohr
        forces_list.append(forces)
    if len(forces_list) == 0:
        raise ValueError("No forces found")
    if len(forces_list) == 1:
        return forces_list[0]
    return forces_list


def read_orca_geom(out_txt: str):
    re_start = r"CARTESIAN COORDINATES \(ANGSTROEM\)"
    re_end = r"CARTESIAN COORDINATES \(A\.U\.\)"

    geom_chunks = find_all_between(out_txt, re_start, re_end)

    atoms_list = [parse_geom_from_out(chunk) for chunk in geom_chunks]
    if len(atoms_list) == 0:
        raise ValueError("No geometry found")
    if len(atoms_list) == 1:
        return atoms_list[0]
    return atoms_list


def read_orca_singlepoint(calc_dir, calc_name: str = "orca", atomref_energies: dict[str, float] = None):
    outfile = Path(calc_dir) / f"{calc_name}.out"
    hessfile = Path(calc_dir) / f"{calc_name}.hess"
    out_txt = outfile.read_text()
    atoms = read_orca_geom(out_txt)
    if isinstance(atoms, list):
        raise RuntimeError("It seems that this is an optimization calculation.")
    energy = read_orca_energy(out_txt)
    if atomref_energies is not None:
        for s in atoms.get_chemical_symbols():
            energy -= atomref_energies[s]  # eV
    try:
        forces = read_orca_force(out_txt)[:len(atoms)]
    except ValueError:
        forces = None
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.set_calculator(calc)
    if hessfile.is_file():
        hessian = read_orca_hessian(hessfile) * units.Hartree / (units.Bohr**2)
        atoms.info["hessian"] = hessian
    return atoms


def read_orca_opt(calc_dir, calc_name: str = "orca", atomref_energies: dict[str, float] = None):
    re_start = r"GEOMETRY OPTIMIZATION CYCLE"
    re_end = r"Geometry convergence"

    with open(Path(calc_dir) / f"{calc_name}.out", mode="r", encoding="utf-8") as fd:
        text = fd.read()
    if "THE OPTIMIZATION HAS CONVERGED" not in text:
        warnings.warn("Optimization did not converge.", stacklevel=1)

    chunks = find_all_between(text, re_start, re_end)
    atoms_list = []
    for chunk in chunks:
        try:
            atoms = read_orca_geom(chunk)
        except ValueError:
            warnings.warn("No geometry found.", stacklevel=1)
            continue
        energy = read_orca_energy(chunk)
        if atomref_energies is not None:
            for s in atoms.get_chemical_symbols():
                energy -= atomref_energies[s]  # eV
        forces = read_orca_force(chunk)
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.set_calculator(calc)
        atoms_list.append(atoms)

    stationary_eval_start = text.find("FINAL ENERGY EVALUATION AT THE STATIONARY POINT")
    if stationary_eval_start == -1:
        warnings.warn("Geometry optimization did not converge.", stacklevel=1)
    text = text[stationary_eval_start:]
    try:
        atoms = read_orca_geom(text)
    except ValueError:
        warnings.warn("No geometry found.", stacklevel=1)
        return atoms_list
    energy = read_orca_energy(text)
    calc = SinglePointCalculator(atoms, energy=energy, forces=None)
    atoms.set_calculator(calc)
    if (hess_file := Path(calc_dir) / f"{calc_name}.hess").is_file():
        atoms.info["hessian"] = read_orca_hessian(hess_file) * units.Hartree / (units.Bohr**2)
    atoms_list.append(atoms)
    return atoms_list


def read_orca_xyz(xyzfile, subs_key=r"E\s+", atomref_energies: dict[str, float] = None):
    xyz_txt = Path(xyzfile).read_text()
    re_energy = r"({})([-+]?\d+.?\d+)".format(subs_key)
    # substitute E\s+ with energy=
    xyz_txt = re.sub(re_energy, r"energy=\2", xyz_txt)
    s = io.StringIO(xyz_txt)
    images = ase.io.read(s, ":", format="extxyz")
    if not isinstance(images, list):
        images = [images]
    for i in range(len(images)):
        energy = images[i].get_potential_energy() * units.Hartree
        if atomref_energies is not None:
            for s in images[i].get_chemical_symbols():
                energy -= atomref_energies[s]
        calc = SinglePointCalculator(images[i], energy=energy)
        images[i].set_calculator(calc)
    return images


def read_orca_md(calc_dir, trajectory_file: str = "trajectory.xyz", force_file: str | None = "forces.xyz"):
    trajectory_file = Path(calc_dir) / trajectory_file
    images = read_orca_xyz(trajectory_file, "E_Pot=")[1:]
    if force_file is not None:
        force_file = Path(calc_dir) / force_file
        force_images = ase.io.read(force_file, "1:")
        for i in range(len(images)):
            # Distance is already in Angstrom in the xyz file
            F = force_images[i].get_positions() * units.Hartree
            calc = SinglePointCalculator(images[i], energy=images[i].get_potential_energy(), forces=F)
            images[i].set_calculator(calc)
    return images


def read_orca_neb(calc_dir, calc_name: str = "orca"):
    mep_traj = Path(calc_dir) / f"{calc_name}_MEP_trj.xyz"
    mep_images = read_orca_xyz(mep_traj)

    return mep_images


def read_orca(calc_dir, calc_name: str = "orca", calc_type: str = "singlepoint", **kwargs):
    if calc_type == "singlepoint":
        return read_orca_singlepoint(calc_dir, calc_name, **kwargs)
    elif calc_type == "opt":
        return read_orca_opt(calc_dir, calc_name, **kwargs)
    elif calc_type == "md":
        return read_orca_md(calc_dir, calc_name, **kwargs)
    elif calc_type == "neb":
        return read_orca_neb(calc_dir, calc_name)
    else:
        raise ValueError(f"Unknown calculation type: {calc_type}")
