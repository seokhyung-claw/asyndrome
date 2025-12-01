import json
import stim


def schedule_to_stim(circuit: stim.Circuit, schedule_path: str):
    with open(schedule_path, "r") as file:
        schedule: list[list[dict]] = json.load(file)

    for tick in schedule:
        for check in tick:
            data: int = check["data"]
            ancilla: int = check["ancilla"]
            pauli: str = check["pauli"]

            if pauli == "Z":
                circuit.append("CNOT", [data, ancilla])  # type: ignore
            elif pauli == "X":
                circuit.append("H", ancilla)  # type: ignore
                circuit.append("CNOT", [ancilla, data])  # type: ignore
                circuit.append("H", ancilla)  # type: ignore

    return circuit