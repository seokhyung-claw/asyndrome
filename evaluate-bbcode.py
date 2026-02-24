import sys
import asyndrome
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import matplotlib as mpl

print("Evaluate bivariate bicycle code n = 72")

TESTCASE = "bbcode-72"
code = asyndrome.CSSCode.from_file("qecc/bbcode-72.json")

alpha_bp_osd = asyndrome.Schedule.from_file("qecc/bbcode-72/alpha-bp_osd.json")
alpha_unionfind = asyndrome.Schedule.from_file(
    "qecc/bbcode-72/alpha-hypergraph_union_find.json"
)

results = {
    "BP-OSD": [
        # AlphaSyndrome + BP-OSD
        (
            "AlphaSyndrome",
            *alpha_bp_osd.evaluate(code, "bp_osd", asyndrome.Brisbane(), 100000),
        ),
        # IBM + BP-OSD
        (
            "IBM",
            *asyndrome.IBMEvaluator.evaluate(
                72, "bp_osd", asyndrome.Brisbane(), 100000
            ),
        ),
    ],
    "Unionfind": [
        (
            "AlphaSyndrome",
            *alpha_unionfind.evaluate(
                code, "hypergraph_union_find", asyndrome.Brisbane(), 100000
            ),
        ),
        # IBM + BP-OSD
        (
            "IBM",
            *asyndrome.IBMEvaluator.evaluate(
                72, "hypergraph_union_find", asyndrome.Brisbane(), 100000
            ),
        ),
    ],
}

# matplotlib configs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 26
plt.rcParams["font.family"] = "serif"
fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
colors = mpl.colormaps["Dark2"].colors  # type: ignore

for i, decoder in enumerate(("BP-OSD", "Unionfind")):
    print(f"  {decoder}")
    ax = plt.subplot(1, 3, i + 1)

    data = results[decoder]

    rec = {}
    for i, (method, xrate, zrate) in enumerate(data):
        ax.scatter(
            xrate,
            zrate,
            label=method,
            marker="P" if method == "AlphaSyndrome" else "o",
            color=colors[0 if method == "AlphaSyndrome" else 3],
            zorder=10 if method == "AlphaSyndrome" else 3,
            s=800 if method == "AlphaSyndrome" else 600,
            alpha=0.8,
        )

        rec[method] = 1 - (1 - xrate) * (1 - zrate)

    print((rec["IBM"] - rec["AlphaSyndrome"]) / rec["AlphaSyndrome"] * 100)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    minimum = min(xmin, ymin, 0)
    maximum = max(xmax, ymax)

    variation = (maximum - minimum) / 15
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
    x = np.linspace(minimum, maximum, 40)
    y = np.linspace(minimum, maximum, 40)
    X, Y = np.meshgrid(x, y)
    Z = 1 - (1 - X) * (1 - Y)
    contour_plot = ax.pcolormesh(X, Y, Z, shading="auto", alpha=0.4, cmap="binary")

    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    if i == 0:
        ax.set_xlabel("logical $X$ error rate", loc="left", labelpad=25)
        ax.set_ylabel("logical $Z$ error rate")

    ax.set_title(decoder, y=1.12)
    ax.grid()


handles = []
labels = []
for handle, label in zip(*axes[0].get_legend_handles_labels()):
    handles.append(handle)
    labels.append(label)

ax = plt.subplot(1, 3, 3)
ax.axis("off")
ax.legend(handles[:3], labels[:3], loc="center", ncol=1)
fig.savefig("results/bbcode.pdf")
