import json
import os
from pathlib import Path
from time import time
import asyndrome
import argparse

parser = argparse.ArgumentParser(
    "main.py", description="Scheduling syndrome measurement for QECCs"
)

parser.add_argument(
    "qecc",
    type=str,
    help="location of the QECC to be scheduled",
    metavar="QECC",
)

parser.add_argument(
    "-d",
    "--decoder",
    type=str,
    default=None,
    help="decoder to use of the corresponding code",
)

parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="alpha",
    help="searching method, default to 'alpha', could be 'baseline' 'alpha' 'google'",
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="output filename, default will be automatically selected",
)


def schedule(qecc: str, decoder: str | None, method: str, output: str | None):
    print("Summary:")
    print(f"    QECC: {qecc}")
    print(f"    Decoder: {decoder}")
    print(f"    Method: {method}")

    code = asyndrome.CSSCode.from_file(qecc)

    start_time = time()

    scheduler: asyndrome.Scheduler = {
        "baseline": asyndrome.BaselineScheduler(logpath=f"{qecc}.pulp.log"),
        "alpha": asyndrome.AlphaScheduler(iters_per_step=8000, nshots=20000),
        "google": asyndrome.GoogleScheduler(code.d, code.n // code.d),
        "trivial": asyndrome.TrivialScheduler(code.d, code.n // code.d)
    }[method]

    if decoder is None and method == "alpha":
        raise RuntimeError("unable to schedule with alpha without decoder")
    
    if decoder is None:
        decoder = ""

    schedule = scheduler.schedule(code, decoder, asyndrome.Brisbane())

    end_time = time()

    print(f"Schedule completed in {end_time - start_time:.2f}s")
    print(f"Max tick: {schedule.max_tick}")

    if output is None:
        output_folder = str(Path(qecc).with_suffix(""))
        os.makedirs(output_folder, exist_ok=True)
        if decoder == "":
            output = output_folder + f"/{method}.json"
        else:
            output = output_folder + f"/{method}-{decoder}.json"

    print(f"Output to {output}")

    with open(output, "w") as output_file:
        json.dump(schedule.to_serializable(), output_file, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    schedule(args.qecc, args.decoder, args.method, args.output)
