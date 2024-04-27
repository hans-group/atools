import numpy as np
from typing import Sequence, Union


class ElectroCatalysis:
    def __init__(
        self,
        steps: Sequence[str],  # Name of reaction steps
        binding_energies: Union[np.ndarray, Sequence[float]],  # Binding energies
        n_electrons: List[int] = None,  # number of involved electrons per step
        set_zero: str = "last",  # Which step to set energy as zero
        use_e0=False,  # If True, use E0 instead of free energy (G)
    ):
        self.steps = steps
        key = "E0" if use_e0 else "G"
        self.free_energies = get_values(binding_energies, "G")
        self.n_electrons = np.asarray(n_electrons) if n_electrons is not None else np.zeros(len(steps))
        self.set_zero = set_zero

        self._validate_inputs()

    def _validate_inputs(self):
        if len(self.steps) != len(self.free_energies):
            raise ValueError("steps and free_energies must have the same length")

        if len(self.steps) != len(self.n_electrons):
            raise ValueError("steps and n_electrons must have the same length")

        if self.set_zero not in ["last", "first"]:
            raise ValueError("set_zero must be either 'last' or 'first'")

    def shifted_free_energies(self, U=0.0):
        shift = self.free_energies[-1] if self.set_zero == "last" else 0
        energies = self.free_energies - shift
        energies += -self.n_electrons * U
        return energies

    @property
    def equilibrium_potential(self):
        ne_diff = self.n_electrons[0] - self.n_electrons[-1]
        energy_diff = self.free_energies[0] - self.free_energies[-1]
        return energy_diff / ne_diff

    def limiting_potential(self, direction="forward"):
        if direction == "forward":
            ne_diff = self.n_electrons[:-1] - self.n_electrons[1:]
            energy_diff = self.free_energies[:-1] - self.free_energies[1:]
            return np.min(energy_diff / ne_diff)
        elif direction == "backward":
            ne_diff = self.n_electrons[1:] - self.n_electrons[:-1]
            energy_diff = self.free_energies[1:] - self.free_energies[:-1]
            return np.max(energy_diff / ne_diff)

    def draw_energy_diagram(self, colors=None, annotate=True, ax=None):
        if ax is None:
            ax = plt.gca()
        if colors is None:
            colors = ["k", "r", "b", "g"]
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Free energy (eV)")
        ax.set_xticks(range(len(self.steps)))
        ax.set_xticklabels(self.steps)
        ax.set_xlim(-0.5, len(self.steps) - 0.5)

        G_0 = self.shifted_free_energies()
        E_eq = self.equilibrium_potential
        E_lim_fwd = self.limiting_potential(direction="forward")
        E_lim_bwd = self.limiting_potential(direction="backward")
        self.eta_fwd = E_eq - E_lim_fwd
        self.eta_bwd = E_lim_bwd - E_eq

        G_lim_fwd = self.shifted_free_energies(U=E_lim_fwd)
        G_eq = self.shifted_free_energies(U=E_eq)
        G_lim_bwd = self.shifted_free_energies(U=E_lim_bwd)

        draw_energy_diagram(G_0, ax=ax, color=colors[0], label="U = 0 V")
        draw_energy_diagram(
            G_lim_fwd, ax=ax, color=colors[1], label=f"U = {E_lim_fwd:.2f} V ($\eta$ = {self.eta_fwd:.2f} V)"
        )
        draw_energy_diagram(G_eq, ax=ax, color=colors[2], label=f"U = {E_eq:.2f} V")
        draw_energy_diagram(
            G_lim_bwd, ax=ax, color=colors[3], label=f"U = {E_lim_bwd:.2f} V ($\eta$ = {self.eta_bwd:.2f} V)"
        )

        if annotate:
            ymin, ymax = ax.get_ylim()
            offset = 0.02 * (ymax - ymin)
            for i in range(len(self.steps)):
                ax.text(i, G_0[i] - offset, f"{G_0[i]:.2f}", ha="center", va="top", color=colors[0])
                ax.text(i, G_lim_fwd[i] - offset, f"{G_lim_fwd[i]:.2f}", ha="center", va="top", color=colors[1])
                ax.text(i, G_eq[i] - offset, f"{G_eq[i]:.2f}", ha="center", va="top", color=colors[2])
                ax.text(i, G_lim_bwd[i] - offset, f"{G_lim_bwd[i]:.2f}", ha="center", va="top", color=colors[3])
            ax.set_ylim(ymin - offset * 2, ymax)
