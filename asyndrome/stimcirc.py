from contextlib import contextmanager
from functools import partial
from pathlib import Path
import pickle
from typing import Any, Iterable
import numpy as np
import stim
import sinter
import subprocess as sub
import sys
from typing import IO
import stimbposd
import relay_bp
import relay_bp.stim


class StimMeasurement:
    def __init__(self) -> None:
        self._index = 0

    def _update(self):
        self._index -= 1


class StimCircuit:
    def __init__(self) -> None:
        self._circuit = stim.Circuit()
        self._measurements = list[StimMeasurement]()

    def clear(self):
        self._circuit.clear()
        self._measurements.clear()

    def gate(
        self,
        gate: str,
        targets: Any | list[Any],
        arg: float | Iterable[float] | None = None,
    ):
        self._circuit.append(gate, targets, arg)  # type: ignore

    def measures(self, gate: str, targets: list[Any]) -> list[StimMeasurement]:
        self._circuit.append(gate, targets)  # type: ignore
        meas: list[StimMeasurement] = []
        for _ in range(len(targets)):
            meas.append(self._new_measure())
        return meas

    def measure_pauli(self, pauli: str) -> StimMeasurement:
        self._circuit.append(
            "MPP",
            [*stim.target_combined_paulis(stim.PauliString(pauli))],  # type: ignore
        )
        return self._new_measure()

    def concat(self, circuit: "StimCircuit"):
        self._circuit += circuit._circuit
        for m in self._measurements:
            m._index -= len(circuit._measurements)
        self._measurements.extend(circuit._measurements)

    def detector(self, measures: list[StimMeasurement], index: int):
        self._circuit.append(
            "DETECTOR", [stim.target_rec(m._index) for m in measures], index
        )

    def observable(self, measures: list[StimMeasurement], index: int):
        self._circuit.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(m._index) for m in measures], index
        )

    def _new_measure(self):
        meas = StimMeasurement()
        self._measurements.append(meas)
        for m in self._measurements:
            m._update()
        return meas


def _sinter_predict_observable(detection_events, decoder, custom_decoders, dem):
    return sinter.predict_observables(
        dem=dem, decoder=decoder, custom_decoders=custom_decoders, dets=detection_events
    )


RELAY_PARAMS = {
    "gamma0": 0.1,
    "pre_iter": 80,
    "num_sets": 100,
    "set_max_iter": 60,
    "gamma_dist_interval": (-0.24, 0.66),
    "stop_nconv": 2,
}

BPLSD_PARAMS = {
    "max_iter": 5,
    "bp_method": "min_sum",
    "ms_scaling_factor": 0.5,
    "schedule": "parallel",
    "lsd_method": "lsd_e",
    "lsd_order": 3,
}


class DecoderAgent:
    def __init__(self, decoder: str, nkd: tuple[int, int, int]) -> None:
        self._process = sub.Popen(
            [sys.executable, (Path(__file__).parent.parent / "stimdec.py").resolve()],
            stdin=sub.PIPE,
            stdout=sub.PIPE,
        )

        self._stdin: IO[bytes] = self._process.stdin  # type: ignore
        self._stdout: IO[bytes] = self._process.stdout  # type: ignore
        self._decoder = decoder
        self._nkd = nkd

    def simulate(
        self,
        circuit: StimCircuit,
        nshots: int = 10000,
    ):
        custom_decoders = {
            "bp_osd": stimbposd.SinterDecoder_BPOSD(
                max_bp_iters=self._nkd[0]  # , osd_order=self._nkd[2]
            ),
            **relay_bp.stim.sinter_decoders(**RELAY_PARAMS),
        }

        # some stim stuffs here
        sampler = circuit._circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            nshots, separate_observables=True
        )
        dem = circuit._circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True
        )

        # worker function to be distributed to the stimdec.py worker
        decode = partial(
            _sinter_predict_observable,
            dem=dem,
            decoder=self._decoder,
            custom_decoders=custom_decoders,
        )

        # if decode by mwpm, then force serial decoding, otherwise send to the worker
        if self._decoder in ("pymatching", "fusion_blossom"):
            predictions = decode(detection_events)
        else:
            predictions = self._request(decode, detection_events)

        # count the number of flips
        flips = np.sum(np.any(predictions != observable_flips, axis=1))
        return flips

    def _request(self, decode, detection_events):
        input = pickle.dumps((decode, detection_events))
        self._stdin.write(len(input).to_bytes(4, "big"))
        self._stdin.write(input)
        self._stdin.flush()

        length_bytes = self._stdout.read(4)
        response_length = int.from_bytes(length_bytes, "big")
        response = self._stdout.read(response_length)
        return pickle.loads(response)

    def _close(self):
        self._stdin.close()
        self._process.wait()


@contextmanager
def decoder_agent(decoder: str, nkd: tuple[int, int, int]):
    decode_worker = DecoderAgent(decoder, nkd)
    try:
        yield decode_worker
    finally:
        decode_worker._close()


class ErrorModel:
    def idling(self, targets: int | list[int], circuit: StimCircuit): ...

    def cnot(self, targets: int | list[int], circuit: StimCircuit): ...
