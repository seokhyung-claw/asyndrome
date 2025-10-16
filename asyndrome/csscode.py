import dataclasses
import json
from typing import Literal


@dataclasses.dataclass
class PauliCheck:
    data: int
    ancilla: int
    pauli: Literal["X", "Z"]

    def __str__(self) -> str:
        return f"{self.pauli}{self.data} checked by {self.ancilla}"

    @staticmethod
    def from_stabilizer(stabilizer: str, ancilla: int):
        return [
            PauliCheck(data, ancilla, pauli)
            for data, pauli in enumerate(stabilizer)
            if pauli == "X" or pauli == "Z"
        ]


@dataclasses.dataclass
class CSSCode:
    family: str
    n: int
    k: int
    d: int

    x_stabilizers: list[str]
    z_stabilizers: list[str]
    logical_xs: list[str]
    logical_zs: list[str]

    @property
    def ancillas(self):
        return len(self.x_stabilizers) + len(self.z_stabilizers)

    @staticmethod
    def from_file(filename: str):
        with open(filename, "r") as file:
            return CSSCode(**json.load(file))

    @staticmethod
    def from_string(string: str):
        return CSSCode(**json.loads(string))

    def to_string(self):
        return json.dumps(dataclasses.asdict(self))

    def to_file(self, filename: str):
        with open(filename, "w") as file:
            json.dump(dataclasses.asdict(self), file)

    def x_checks(self):
        checks = list[PauliCheck]()
        for ancilla_base_id, stabilizer in enumerate(self.x_stabilizers):
            checks.extend(
                PauliCheck.from_stabilizer(stabilizer, ancilla_base_id + self.n)
            )
        return checks

    def z_checks(self):
        checks = list[PauliCheck]()
        for ancilla_base_id, stabilizer in enumerate(self.z_stabilizers):
            checks.extend(
                PauliCheck.from_stabilizer(
                    stabilizer, ancilla_base_id + self.n + len(self.x_stabilizers)
                )
            )
        return checks
