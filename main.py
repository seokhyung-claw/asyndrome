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
    required=True,
    help="decoder to use of the corresponding code",
)


parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="output filename, default will be automatically selected",
)


def schedule(qecc: str, decoder: str, output: str | None):
    print("Summary:")
    print(f"    QECC: {qecc}")
    print(f"    Decoder: {decoder}")

    code = asyndrome.QECCode.from_file(qecc)

    start_time = time()

    scheduler: asyndrome.Scheduler = asyndrome.AlphaScheduler(
        iters_per_step=8000, nshots=10000
    )

    if decoder is None:
        decoder = ""

    schedule = scheduler.schedule(code, decoder, asyndrome.Brisbane())

    end_time = time()

    print(f"Schedule completed in {end_time - start_time:.2f}s")
    print(f"Max tick: {schedule.max_tick}")

    if output is None:
        output_folder = str(Path(qecc).with_suffix(""))
        os.makedirs(output_folder, exist_ok=True)
        output = output_folder + f"/alphasyndrome-{decoder}.json"

    print(f"Output to {output}")

    with open(output, "w") as output_file:
        json.dump(schedule.to_serializable(), output_file, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    schedule(args.qecc, args.decoder, args.output)
