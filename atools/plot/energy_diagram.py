import matplotlib.pyplot as plt


def draw_energy_diagram(
    energies,
    ax=None,
    color="k",
    width=0.5,
    thickness=3,
    connect_style="--",
    lw=0.8,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    label = kwargs.pop("label", None)
    shift = 0.5 * width
    for i, e in enumerate(energies):
        xs = [i - shift, i + shift]
        ys = [e, e]
        _label = label if i == 0 else None
        ax.plot(
            xs,
            ys,
            "-",
            c=color,
            lw=thickness,
            label=_label,
            solid_capstyle="round",
            **kwargs,
        )
    for i in range(len(energies) - 1):
        xs = [i + shift, i + 1 - shift]
        ys = [energies[i], energies[i + 1]]
        ax.plot(xs, ys, c=color, ls=connect_style, lw=lw, **kwargs)
    return ax
