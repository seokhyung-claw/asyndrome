from statistics import mean, stdev
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches
from tqdm import tqdm

import asyndrome


action = sys.argv[1]

if action == "schedule":
    print("Scheduling syndrome measurement for non-uniform Brisbane error model")
    for distance in (3, 5, 7):
        print(f"  Distance {distance}")

        code = asyndrome.CSSCode.from_file(f"qecc/surface-{distance}x{distance}.json")
        error_model = asyndrome.NonUniformBrisbane(code.n, code.ancillas, 2)

        scheduler = asyndrome.AlphaScheduler(iters_per_step=8000, nshots=40000)
        scheduler.schedule(code, "pymatching", error_model).to_file(
            f"results/nonuniform/surface-{distance}x{distance}.json"
        )
elif action == "evaluate":
    print("Evaluating for non-uniform Brisbane error model")

    rename = {"mcts": "AlphaSyndrome", "google": "Google"}
    colors = mpl.colormaps["Dark2"].colors  # type: ignore
    fig = plt.figure(figsize=(18, 6), dpi=300, constrained_layout=True)
    plt.rcParams["font.size"] = 28
    plt.rcParams["font.family"] = "serif"

    for i, distance in enumerate((3, 5, 7)):
        print(f"  Distance {distance}...")
        code = asyndrome.CSSCode.from_file(f"qecc/surface-{distance}x{distance}.json")
        error_model = asyndrome.NonUniformBrisbane(code.n, code.ancillas, 2)

        google_schedule = asyndrome.Schedule.from_file(
            f"qecc/surface-{distance}x{distance}/google.json"
        )
        alpha_schedule = asyndrome.Schedule.from_file(
            f"results/nonuniform/surface-{distance}x{distance}.json"
        )

        nshots = 10000000

        google_datas = []
        alpha_datas = []

        for _ in tqdm(range(10)):
            xrate, zrate = google_schedule.evaluate(
                code, "pymatching", error_model, nshots
            )
            google_data = 1 - (1 - xrate) * (1 - zrate)
            google_datas.append(google_data)

            xrate, zrate = alpha_schedule.evaluate(
                code, "pymatching", error_model, nshots
            )
            alpha_data = 1 - (1 - xrate) * (1 - zrate)
            alpha_datas.append(alpha_data)

        alpha_data = mean(alpha_datas)
        google_data = mean(google_datas)

        print("Plotting")

        ax = plt.subplot(1, 3, i + 1)
        ax.set_ylabel("overall logical error rate")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_xticklabels([])

        ax.bar(
            ["AlphaSyndrome", "Google"],
            [alpha_data, google_data],
            yerr=[stdev(alpha_datas), stdev(google_datas)],
            color=[colors[0], colors[2]],
            ecolor="black",
            error_kw={"capsize": 10},
        )

        ax.text(
            -0.4,
            (alpha_data * 0.3 + google_data * 0.7),
            f"$\\downarrow${(google_data - alpha_data) * 100 / google_data:.2f}%",
        )

    fig.savefig("results/nonuniform/nonuniform.png")
elif action == "plot":
    print("Plotting nonuniform error model")

    fig = plt.figure(figsize=(18, 5), dpi=300, constrained_layout=True)

    for i, distance in enumerate((3, 5, 7)):
        print(f"  Distance {distance}")

        ax = plt.subplot(1, 3, i + 1)

        code = asyndrome.CSSCode.from_file(f"qecc/surface-{distance}x{distance}.json")
        error_model = asyndrome.NonUniformBrisbane(code.n, code.ancillas, 2)

        def get_err_scale(index: int):
            idle_rate = error_model.error_data[index][0]
            scale = idle_rate / 0.005243978963702009
            return scale

        def get_coord(index: int):
            if index < code.n:
                # this is a data qubit
                row = index // distance
                y = row * 5

                col = index % distance
                x = col * 5

                return (x, y)
            else:
                # this is a ancilla qubit
                sindex = index - code.n
                stabilizer = (code.x_stabilizers + code.z_stabilizers)[sindex]
                data_index = [i for i, p in enumerate(stabilizer) if p != "I"]
                data_coords = list(map(get_coord, data_index))
                x_coords, y_coords = zip(*data_coords)
                average_x = sum(x_coords) / len(x_coords)
                average_y = sum(y_coords) / len(y_coords)

                if average_x == 0:
                    average_x -= 2.5

                if average_x == (distance - 1) * 5.0:
                    average_x += 2.5

                if average_y == 0:
                    average_y -= 2.5

                if average_y == (distance - 1) * 5.0:
                    average_y += 2.5

                return (average_x, average_y)

        radius = 1

        for i in range(code.n + code.ancillas):
            x, y = get_coord(i)
            if i < code.n:
                circle = patches.Circle(
                    (x, y), radius, facecolor="white", edgecolor="gray", linewidth=2
                )
            else:
                scale = (get_err_scale(i) + 0.5) / 2.5
                circle = patches.Circle(
                    (x, y), radius, facecolor=plt.get_cmap("Reds")(scale)
                )
            ax.add_patch(circle)
        ax.set_xlim(-radius - 2.5, 5 * (distance - 1) + radius + 2.5)
        ax.set_ylim(-radius - 2.5, 5 * (distance - 1) + radius + 2.5)
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig("results/nonuniform/error_model.png")
