import json
from random import Random
from asyndrome.csscode import CSSCode, PauliCheck
from asyndrome.scheduler import Scheduler, Schedule, load_all_schedules
from asyndrome.baseline import BaselineScheduler
from asyndrome.stimcirc import ErrorModel, StimCircuit
from asyndrome.treesearch import AlphaScheduler
try:
    from asyndrome.special import GoogleScheduler, TrivialScheduler, IBMEvaluator
except ImportError:
    GoogleScheduler = None
    TrivialScheduler = None
    IBMEvaluator = None


class Brisbane(ErrorModel):
    def idling(self, targets: int | list[int], circuit: StimCircuit):
        circuit.gate("DEPOLARIZE1", targets, 0.005243978963702009)

    def cnot(self, targets: int | list[int], circuit: StimCircuit):
        circuit.gate("DEPOLARIZE1", targets, 0.007432674432642006)


class SD6NoiseModel(ErrorModel):
    circuit_level = True

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def idling(self, targets: int | list[int], circuit: StimCircuit):
        circuit.gate("DEPOLARIZE1", targets, self.p)

    def after_single_qubit_gate(
        self, gate: str, targets: int | list[int], circuit: StimCircuit
    ):
        circuit.gate("DEPOLARIZE1", targets, self.p)

    def after_two_qubit_gate(
        self, gate: str, targets: int | list[int], circuit: StimCircuit
    ):
        circuit.gate("DEPOLARIZE2", targets, self.p)

    def before_measurement(
        self, gate: str, targets: int | list[int], circuit: StimCircuit
    ):
        if gate == "MZ":
            circuit.gate("X_ERROR", targets, self.p)
        elif gate == "MX":
            circuit.gate("Z_ERROR", targets, self.p)

    def after_reset(
        self, gate: str, targets: int | list[int], circuit: StimCircuit
    ):
        if gate == "RZ":
            circuit.gate("X_ERROR", targets, self.p)
        elif gate == "RX":
            circuit.gate("Z_ERROR", targets, self.p)


class NonUniformModel(ErrorModel):
    def __init__(self, error_data: dict[int, tuple[float, float]]) -> None:
        super().__init__()
        self.error_data = error_data

    def _apply_error(
        self, targets: int | list[int], circuit: StimCircuit, error_kind: int
    ):
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            error_rate = self.error_data.get(target, (0, 0))[error_kind]
            circuit.gate("DEPOLARIZE1", target, error_rate)

    def idling(self, targets: int | list[int], circuit: StimCircuit):
        self._apply_error(targets, circuit, 0)

    def cnot(self, targets: int | list[int], circuit: StimCircuit):
        self._apply_error(targets, circuit, 1)

    def save_model(self, path: str):
        with open(path, "w") as file:
            json.dump(self.error_data, file)


class TrivialModel(ErrorModel):
    def __init__(self, idle_err: float, cnot_err: float) -> None:
        super().__init__()
        self.idle_err = idle_err
        self.cnot_err = cnot_err

    def idling(self, targets: int | list[int], circuit: StimCircuit):
        circuit.gate("DEPOLARIZE1", targets, self.idle_err)

    def cnot(self, targets: int | list[int], circuit: StimCircuit):
        circuit.gate("DEPOLARIZE1", targets, self.cnot_err)


def make_brisbane_nonuniform(ndata: int, nancilla: int, seed: int, scale: float):
    idle_err = 0.005243978963702009
    cnot_err = 0.007432674432642006

    rnd = Random(seed)

    error_data = {}

    for i in range(nancilla):
        scaling = rnd.uniform(0, scale)
        idle_biased = scaling * idle_err
        cnot_biased = scaling * cnot_err
        error_data[i + ndata] = (idle_biased, cnot_biased)

    return error_data


class NonUniformBrisbane(NonUniformModel):
    def __init__(self, ndata: int, nancilla: int, scale: float, seed: int = 42) -> None:
        super().__init__(make_brisbane_nonuniform(ndata, nancilla, seed, scale))


__all__ = [
    "CSSCode",
    "BaselineScheduler",
    "AlphaScheduler",
    "PauliCheck",
    "Scheduler",
    "Schedule",
    "ErrorModel",
    "Brisbane",
    "SD6NoiseModel",
    "NonUniformBrisbane",
    "GoogleScheduler",
    "load_all_schedules",
    "TrivialModel",
    "TrivialScheduler",
    "IBMEvaluator",
]
