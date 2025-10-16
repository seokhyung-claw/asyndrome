import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.lines as mlines
from matplotlib import pyplot as plt

import asyndrome


def evaluate(testcase: str, nshots: int = 1000000):
    qecc = asyndrome.CSSCode.from_file(f"qecc/{testcase}.json")

    data = {}
    for method in ("alpha", "trivial", "google"):
        if method == "alpha":
            full_name = "alpha-pymatching"
        else:
            full_name = method

        schedule = asyndrome.Schedule.from_file(f"qecc/{testcase}/{full_name}.json")
        data[method] = schedule.evaluate(
            qecc, "pymatching", asyndrome.Brisbane(), nshots
        )
    return data


if __name__ == "__main__":
    testcases = [
        "surface-3x3",
        "surface-5x5",
        "surface-7x7",
        "surface-9x9",
        "surface-5x9",
    ]

    rename = {"alpha": "AlphaSyndrome", "google": "Google", "trivial": "Trivial"}
    numberof = {"alpha": 0, "google": 2, "trivial": 1}

    colors = mpl.colormaps["Dark2"].colors  # type: ignore

    plt.rcParams["font.size"] = 26
    plt.rcParams["font.family"] = "serif"
    fig, _ = plt.subplots(
        2,
        3,
        figsize=(15, 12),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )

    handles, labels = [], []

    for i, case in enumerate(testcases):
        ax = plt.subplot(2, 3, i + 1)
        print(f"  Case {case}")
        data = evaluate(case)
        for j, (name, (xrate, zrate)) in enumerate(data.items()):
            ax.scatter(
                xrate,
                zrate,
                label=rename[name],
                marker="P" if name == "alpha" else "o",
                color=colors[numberof[name]],
                zorder=10 if name == "alpha" else 3,
                s=800 if name == "alpha" else 600,
                alpha=0.8,
            )

        n = case.split("-")[1]
        d = min(map(int, n.split("x")))
        ax.set_title(f"$[[{n},1,{d}]]$", y=1.12)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        minimum = min(xmin, ymin)
        maximum = max(xmax, ymax)

        variation = (maximum - minimum) / 10
        minimum -= variation
        maximum += variation

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            color="gray",
            alpha=0.3,
        )

        ax.set_xlim(minimum, maximum)
        ax.set_ylim(minimum, maximum)

        # contour
        x = np.linspace(minimum, maximum, 50)
        y = np.linspace(minimum, maximum, 50)
        X, Y = np.meshgrid(x, y)
        Z = 1 - (1 - X) * (1 - Y)
        contour_plot = ax.pcolormesh(X, Y, Z, shading="auto", alpha=0.4, cmap="binary")

        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        if i == 3:
            ax.set_xlabel("logical $X$ error rate", labelpad=25, loc="left")
            ax.set_ylabel("logical $Z$ error rate")

        # ax.set_title(f"{testcase}-{decoder}")
        ax.grid()

    handle1 = mlines.Line2D(
        [], [], color=colors[0], marker="P", markersize=20, linestyle="None"
    )
    handle2 = mlines.Line2D(
        [], [], color=colors[1], marker="o", markersize=20, linestyle="None"
    )
    handle3 = mlines.Line2D(
        [], [], color=colors[2], marker="o", markersize=20, linestyle="None"
    )
    handles = [handle1, handle2, handle3]
    labels = ["AlphaSyndrome", "Trivial", "Google"]

    ax = plt.subplot(2, 3, i + 2)  # type: ignore
    ax.axis("off")
    ax.legend(handles, labels, ncol=1, loc="center")
    fig.savefig("results/surface.pdf")
