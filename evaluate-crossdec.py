import re
import asyndrome


def to_latex_sci(number: float, bold: bool):
    """convert number to scientific form in latex"""

    s = f"{number:.2e}"
    match = re.match(r"(.+)e([+-]\d+)", s)
    base, exponent = float(match.group(1)), int(match.group(2))  # type: ignore
    latex_string = f"{base}\\times 10^{{{exponent}}}"
    if bold:
        latex_string = f"\\mathbf{{{latex_string}}}"
    return f"${latex_string}$"


def test_crossdec(nshots: int = 500000):
    output = open("results/crossdec.tex", "w")

    def tex_output(tex: str):
        print(tex, file=output)

    for code_name in ("color-hex", "color-oct"):
        family_name = {"hex": "Hexagonal", "oct": "Square-Octagonal"}[
            code_name.split("-")[1]
        ] + " Color Code"

        print(f"Testing code family '{family_name}'")

        # tex output
        tex_output(r"\hline\multicolumn{7}{|c|}{\textbf{" + family_name + r"}}\\\hline")

        for d in (3, 5, 7, 9):
            print(f"  Code distance = {d}")
            code = asyndrome.CSSCode.from_file(f"qecc/{code_name}-{d}.json")
            data = []
            for test_decoder in ("bp_osd", "hypergraph_union_find"):
                for train_decoder in ("bp_osd", "hypergraph_union_find"):
                    # load train
                    schedule = asyndrome.load_all_schedules(
                        f"qecc/{code_name}-{d}", train_decoder
                    )["alpha"]

                    xrate, zrate = schedule.evaluate(
                        code, test_decoder, asyndrome.Brisbane(), nshots
                    )
                    data.append(1 - (1 - xrate) * (1 - zrate))

            def reduction(to: float, f: float):
                red = (f - to) / f * 100
                return f"${red:.2f}\\%$"

            [bb, bu, ub, uu] = data
            tex_output(
                f"$[[{code.n},{code.k},{code.d}]]$ & {to_latex_sci(bb, bb <= bu)} & {to_latex_sci(bu, bu <= bb)} & {reduction(bb, bu)} & {to_latex_sci(ub, ub <= uu)} & {to_latex_sci(uu, uu <= ub)} & {reduction(uu, ub)}\\\\\\hline"
            )

    output.close()

if __name__ == "__main__":
    test_crossdec()