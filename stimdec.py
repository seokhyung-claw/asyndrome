import os
import pickle
import sys
import numpy as np
import multiprocessing as mp


def count_logic_error(decode, detection_events):
    requested_processes = int(os.environ.get("ASYNDROME_DECODER_PROCESSES", "4"))
    available_cpus = os.cpu_count() or 1
    shot_count = len(detection_events)
    process = max(1, min(requested_processes, available_cpus, shot_count))

    if process == 1:
        return decode(detection_events)

    chunks = np.array_split(detection_events, process)

    with mp.get_context("fork").Pool(process) as p:
        results = p.map(decode, chunks)

        return np.concatenate(results)


if __name__ == "__main__":
    while True:
        length_bytes = sys.stdin.buffer.read(4)
        if not length_bytes: # stdin closed or no more data
            break

        data_length = int.from_bytes(length_bytes, 'big')

        input = sys.stdin.buffer.read(data_length)
        decode, detection_events = pickle.loads(input)
        result = count_logic_error(decode, detection_events)

        response = pickle.dumps(result)
        sys.stdout.buffer.write(len(response).to_bytes(4, 'big'))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()