import sys
import os
import shutil


def table2():
    print("Expect duration: ~10mins")
    for case in (
        "color-hex",
        "color-oct",
        "hypercolor",
        "hypersurface",
        "defect",
    ):
        decoders = "bp_osd,hypergraph_union_find"
        if case == "hypercolor":
            decoders = "hypergraph_union_find"
        elif case in ("hypersurface", "xzzx", "defect"):
            decoders = "pymatching"

        os.system(f"python3 experiment-general.py {case} -d {decoders}")


def table3():
    print("Expect duration: ~5mins")
    os.system("python3 experiment-crossdec.py")


def figure12():
    print("Expect duration: ~2mins")
    os.system("python3 experiment-surface.py")


def figure13():
    print("Expect duration: ~2mins")
    os.system("python3 experiment-bbcode.py")


def figure14():
    print("Expect duration: ~2mins")
    for code in ("color-hex-3", "hypercolor-24-8-4-4", "hypersurface-30-8-3-3"):
        os.system(f"python3 experiment-scaling.py evaluate {code}")


def figure15():
    print("Expect duration: ~15mins")
    for action in ("evaluate", "plot"):
        os.system(f"python3 experiment-nonuniform.py {action}")


if len(sys.argv) != 2:
    print(
        "This is the artifact evaluation script, to use, input the result (table, figure) you want to reproduce."
    )

    print(
        "The script will run the evaluation and output the result in the corresponding folder."
    )

asset = sys.argv[1]

if asset not in ("table2", "table3", "figure12", "figure13", "figure14", "figure15"):
    print("The asset you want to reproduce does not exist.")
    exit(-1)

print(
    f"Reproducing asset: {asset}. The result will be output to the folder 'asset' with the same name."
)

if os.path.exists(f"./asset/{asset}"):
    print("Previous asset exists, deleting.")
    shutil.rmtree(f"./asset/{asset}")

os.makedirs(f"./asset/{asset}", exist_ok=True)

globals()[asset]()
