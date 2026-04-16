from dataclasses import dataclass
import itertools
import os
import json
from typing import Iterable
from asyndrome.csscode import CSSCode, PauliCheck
from asyndrome.stimcirc import ErrorModel, StimCircuit, StimMeasurement, decoder_agent


@dataclass
class Schedule:
    checks: list[list[PauliCheck]]

    def to_serializable(self):
        return [
            [
                {"data": chk.data, "ancilla": chk.ancilla, "pauli": chk.pauli}
                for chk in tick
            ]
            for tick in self.checks
        ]

    def to_file(self, filename: str):
        with open(filename, "w") as json_file:
            json.dump(self.to_serializable(), json_file)

    @staticmethod
    def from_file(filename: str):
        with open(filename, "r") as json_file:
            return Schedule.from_serializable(json.load(json_file))

    @staticmethod
    def from_serializable(obj):
        checks = [[PauliCheck(**chk) for chk in tick] for tick in obj]
        return Schedule(checks)

    @property
    def max_tick(self):
        return len(self.checks)

    def checks_at_tick(self, tick: int):
        return self.checks[tick]

    def evaluation_circuit(self, code: CSSCode, error_model: ErrorModel):
        z_circuit = evaluate_circuit(
            code,
            self,
            code.x_stabilizers + code.z_stabilizers,
            code.logical_xs,
            error_model,
        )

        x_circuit = evaluate_circuit(
            code,
            self,
            code.x_stabilizers + code.z_stabilizers,
            code.logical_zs,
            error_model,
        )

        return z_circuit, x_circuit

    def evaluate(
        self, code: CSSCode, decoder: str, error_model: ErrorModel, nshots: int
    ):
        with decoder_agent(decoder, (code.n, code.k, code.d)) as agent:
            z_circuit, x_circuit = self.evaluation_circuit(code, error_model)

            z_flips = agent.simulate(z_circuit, nshots)
            x_flips = agent.simulate(x_circuit, nshots)

            return x_flips / nshots, z_flips / nshots

    def distance(self, code: CSSCode, error_model: ErrorModel):
        z_circuit, x_circuit = self.evaluation_circuit(code, error_model)
        return min(
            len(
                z_circuit._circuit.search_for_undetectable_logical_errors(
                    dont_explore_detection_event_sets_with_size_above=code.d + 2,
                    dont_explore_edges_with_degree_above=code.d + 2,
                    dont_explore_edges_increasing_symptom_degree=True,
                )
            ),
            len(
                x_circuit._circuit.search_for_undetectable_logical_errors(
                    dont_explore_detection_event_sets_with_size_above=code.d + 2,
                    dont_explore_edges_with_degree_above=code.d + 2,
                    dont_explore_edges_increasing_symptom_degree=True,
                )
            ),
        )

    def evaluate_overall(
        self, code: CSSCode, decoder: str, error_model: ErrorModel, nshots: int
    ):
        xrate, zrate = self.evaluate(code, decoder, error_model, nshots)
        return 1 - (1 - xrate) * (1 - zrate)


def load_all_schedules(folder: str, decoder: str):
    all_schedules = os.listdir(folder)
    schedules: dict[str, Schedule] = {}
    for file in all_schedules:
        if file.startswith("alpha"):
            [_method, file_decoder] = os.path.splitext(file)[0].split("-")
            if file_decoder == decoder:
                method = _method
            else:
                continue
        else:
            method = os.path.splitext(file)[0]

        with open(f"{folder}/{file}", "r") as schedule_file:
            schedule = Schedule.from_serializable(json.load(schedule_file))
            schedules[method] = schedule

    return schedules


class Scheduler:
    def _sort_schedule(self, schedule: Iterable[tuple[PauliCheck, int]]):
        schedule = sorted(schedule, key=lambda x: x[1])
        schedule_grouped = list[list[PauliCheck]]()
        for _key, group in itertools.groupby(schedule, key=lambda ct: ct[1]):
            schedule_grouped.append([ct[0] for ct in group])
        return Schedule(schedule_grouped)

    def schedule(
        self, code: CSSCode, decoder: str, error_model: ErrorModel
    ) -> Schedule:
        raise NotImplementedError()


def _ideal_measurement(
    circuit: StimCircuit, stabilizers: list[str], logicals: list[str]
):
    stabms = list[StimMeasurement]()
    logms = list[StimMeasurement]()
    for stabilizer in stabilizers:
        stabms.append(circuit.measure_pauli(stabilizer))
    for logical in logicals:
        logms.append(circuit.measure_pauli(logical))
    return stabms, logms


def evaluate_circuit(
    code: CSSCode,
    schedule: Schedule,
    stabilizers: list[str],
    logicals: list[str],
    error_model: ErrorModel,
):
    circuit = StimCircuit()

    first_round, first_ls = _ideal_measurement(circuit, stabilizers, logicals)

    if error_model.circuit_level:
        all_qubits = set(range(code.n + code.ancillas))

        for checks in schedule.checks:
            used_qubits: set[int] = set()

            for chk in checks:
                used_qubits.add(chk.data)
                used_qubits.add(chk.ancilla)

                if chk.pauli == "X":
                    circuit.gate("H", chk.ancilla)
                    error_model.after_single_qubit_gate("H", chk.ancilla, circuit)
                    circuit.gate("CNOT", [chk.ancilla, chk.data])
                    error_model.after_two_qubit_gate(
                        "CNOT", [chk.ancilla, chk.data], circuit
                    )
                    circuit.gate("H", chk.ancilla)
                    error_model.after_single_qubit_gate("H", chk.ancilla, circuit)
                else:
                    circuit.gate("CNOT", [chk.data, chk.ancilla])
                    error_model.after_two_qubit_gate(
                        "CNOT", [chk.data, chk.ancilla], circuit
                    )

            idle_qubits = sorted(all_qubits - used_qubits)
            if idle_qubits:
                error_model.idling(idle_qubits, circuit)
    else:
        for checks in schedule.checks:
            idle_ancillas = [True for _ in range(code.ancillas)]

            for chk in checks:
                idle_ancillas[chk.ancilla - code.n] = False

                if chk.pauli == "X":
                    circuit.gate("H", chk.ancilla)
                    circuit.gate("CNOT", [chk.ancilla, chk.data])
                    circuit.gate("H", chk.ancilla)
                else:
                    circuit.gate("CNOT", [chk.data, chk.ancilla])

            for i, is_idle in enumerate(idle_ancillas):
                if is_idle:
                    error_model.idling(i + code.n, circuit)
                else:
                    error_model.cnot(i + code.n, circuit)

    ancilla_targets = [ancilla + code.n for ancilla in range(code.ancillas)]
    if error_model.circuit_level:
        error_model.before_measurement("MZ", ancilla_targets, circuit)
    circuit.measures("MZ", ancilla_targets)

    second_round, second_ls = _ideal_measurement(circuit, stabilizers, logicals)

    # stabilizer verification
    for i, (a, b) in enumerate(zip(first_round, second_round)):
        circuit.detector([a, b], i)

    # corresponding observable
    for i, (first_l, second_l) in enumerate(zip(first_ls, second_ls)):
        circuit.observable([first_l, second_l], i)

    return circuit
