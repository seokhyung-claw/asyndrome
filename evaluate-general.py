import argparse
import glob
import os
import re
from typing import TextIO
import asyndrome


parser = argparse.ArgumentParser(
    description="General evaluation script to test different code families and decoders"
)

parser.add_argument(
    "code_family",
    type=str,
    help="code family to be tested, should have qecc/{code_family}-*.json",
)

parser.add_argument(
    "-d",
    "--decoders",
    type=lambda s: s.split(","),
    help="decoders to be used for evaluation",
    default="bp_osd,hypergraph_union_find",
)


DECODER_TAG = {
    "hypergraph_union_find": "Unionfind",
    "pymatching": "MWPM",
    "bp_osd": "BP-OSD",
}


def report(
    file: TextIO,
    qecc: asyndrome.CSSCode,
    decoder: str,
    data: dict[str, tuple[float, float, float]],
    first_decoder: bool,
    ndecoders: int,
):
    def to_latex_sci(number: float, bold: bool):
        s = f"{number:.2e}"
        match = re.match(r"(.+)e([+-]\d+)", s)
        base, exponent = float(match.group(1)), int(match.group(2))  # type: ignore
        latex_string = f"{base}\\times 10^{{{exponent}}}"
        if bold:
            latex_string = f"\\mathbf{{{latex_string}}}"
        return f"${latex_string}$"

    def format(
        alpha_data: tuple[float, float, float],
        baseline_data: tuple[float, float, float],
    ):
        mctsx, mctsz, mctsd = alpha_data
        fasx, fasz, fastd = baseline_data

        mctso = 1 - (1 - mctsx) * (1 - mctsz)
        faso = 1 - (1 - fasx) * (1 - fasz)
        improve = (faso - mctso) / faso
        return f"{to_latex_sci(mctsx, mctsx <= fasx)} & {to_latex_sci(mctsz, mctsz <= fasz)} & {to_latex_sci(mctso, mctso <= faso)} & {mctsd + 1} & {to_latex_sci(fasx, mctsx >= fasx)} & {to_latex_sci(fasz, mctsz >= fasz)} & {to_latex_sci(faso, mctso >= faso)} & {fastd + 1} & {improve * 100:.2f}\\%"

    if first_decoder:
        nkd = f"\\multirow{{{ndecoders}}}{{*}}{{$[[{qecc.n},{qecc.k},{qecc.d}]]$}}"
        if ndecoders > 1:
            next_line = "\\cline{2-11}"
        else:
            next_line = "\\hline"
    else:
        nkd = ""
        next_line = "\\hline"

    print(
        f"{nkd} & {DECODER_TAG[decoder]} & {format(data['alpha'], data['baseline'])} \\\\{next_line}",
        file=file,
    )


def test_code_family(code_family: str, decoders: list[str], nshots=1000000):
    # search for all codes
    search_pattern = os.path.join("qecc", f"{code_family}-*.json")
    all_codes = [os.path.splitext(file)[0] for file in glob.glob(search_pattern)]
    all_codes.sort()

    error_model = asyndrome.Brisbane()

    print(
        f"Evaluting code family '{code_family}' ({len(all_codes)}) with {', '.join(decoders)}"
    )

    with open(f"results/{code_family}.tex", "w") as file:
        # code family line
        print(
            "\\multicolumn{11}{|c|}{\\textbf{"
            + asyndrome.CSSCode.from_file(all_codes[0] + ".json").family
            + "}} \\\\\\hline",
            file=file,
        )

        for code_file in all_codes:
            code = asyndrome.CSSCode.from_file(code_file + ".json")
            print(f"  [[{code.n},{code.k},{code.d}]]")

            baseline_schedule = asyndrome.Schedule.from_file(
                code_file + "/baseline.json"
            )

            # reporting variable
            first_decoder = True

            for decoder in decoders:
                print(f"    {decoder}")

                alpha_schedule = asyndrome.Schedule.from_file(
                    code_file + f"/alpha-{decoder}.json"
                )

                report(
                    file,
                    code,
                    decoder,
                    {
                        "alpha": (
                            *alpha_schedule.evaluate(
                                code, decoder, error_model, nshots
                            ),
                            alpha_schedule.max_tick,
                        ),
                        "baseline": (
                            *baseline_schedule.evaluate(
                                code, decoder, error_model, nshots
                            ),
                            baseline_schedule.max_tick,
                        ),
                    },
                    first_decoder,
                    len(decoders),
                )

                # reporting variable, not the first line anymore
                first_decoder = False


if __name__ == "__main__":
    args = parser.parse_args()
    test_code_family(args.code_family, args.decoders)
