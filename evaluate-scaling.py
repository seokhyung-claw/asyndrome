import sys

from matplotlib import pyplot as plt
import matplotlib as mpl
import asyndrome

TESTCASES = ("color-hex-3", "hypercolor-24-8-4-4", "hypersurface-30-8-3-3")
DECODERS = ("bp_osd", "hypergraph_union_find", "pymatching")
CNOT_ERR = 0.01
IDLE_ERR = 0.01

action = sys.argv[1]

if action == "schedule":
    print("Schedule for scaling experiment")
    for decoder, testcase in zip(DECODERS, TESTCASES):
        print(f"  {testcase}")

        code = asyndrome.CSSCode.from_file(f"qecc/{testcase}.json")

        scheduler = asyndrome.AlphaScheduler(iters_per_step=8000, nshots=20000)

        for scale in (1, 0.1, 0.01, 0.001):
            print(f"    Scale {scale}")
            error_model = asyndrome.TrivialModel(IDLE_ERR * scale, CNOT_ERR * scale)

            scheduler.schedule(code, decoder, error_model).to_file(
                f"results/scaling/{testcase}-{scale}.json"
            )
elif action == "evaluate":
    print("Evaluate scaling experiment")

    plt.rcParams["font.size"] = 26
    plt.rcParams["font.family"] = "serif"
    colors = mpl.colormaps["Dark2"].colors  # type: ignore

    fig = plt.figure(figsize=(30, 5), dpi=300, constrained_layout=True)

    for i, (decoder, testcase) in enumerate(zip(DECODERS, TESTCASES)):
        print(f"  {testcase}")

        code = asyndrome.CSSCode.from_file(f"qecc/{testcase}.json")
        baseline_schedule = asyndrome.Schedule.from_file(
            f"qecc/{testcase}/baseline.json"
        )

        y_baseline: list[float] = []
        y_alpha: list[float] = []
        x = (1, 0.1, 0.01, 0.001)

        for scale in x:
            print(f"    Scale {scale}")
            error_model = asyndrome.TrivialModel(IDLE_ERR * scale, CNOT_ERR * scale)
            alpha_schedule = asyndrome.Schedule.from_file(
                f"results/scaling/{testcase}-{scale}.json"
            )

            baseline = baseline_schedule.evaluate_overall(
                code, decoder, error_model, 1000000
            )

            alpha = alpha_schedule.evaluate_overall(code, decoder, error_model, 1000000)

            print(alpha, baseline)

            y_baseline.append(baseline)
            y_alpha.append(alpha)

        x = tuple(i * 0.01 for i in x)
        ax = plt.subplot(1, 3, i + 1)
        ax.plot(x, y_alpha, markersize=20, marker="P", label="AlphaSyndrome", color=colors[0])
        ax.plot(x, y_baseline, markersize=20, marker="o", label="Lowest Depth", color=colors[1])
        ax.set_xlabel("physical error rate")
        ax.set_ylabel("logical error rate")
        ax.grid()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"[[{code.n},{code.k},{code.d}]] {code.family}", pad=20)
    
    fig.savefig("results/scaling/scaling.png")