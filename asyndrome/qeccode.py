from collections import defaultdict
import dataclasses
import itertools
import json
from typing import Any, Literal
import networkx as nx


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
class QECCode:
    family: str
    n: int
    k: int
    d: int

    stabilizers: list[str]
    stabilizer_partition: list[list[str]]

    logical_xs: list[str]
    logical_zs: list[str]

    @property
    def ancillas(self):
        return len(self.stabilizers)

    @staticmethod
    def from_file(filename: str):
        with open(filename, "r") as file:
            object = json.load(file)

        if "x_stabilizers" in object:
            # CSS code
            return QECCode(
                object["family"],
                object["n"],
                object["k"],
                object["d"],
                object["x_stabilizers"] + object["z_stabilizers"],
                [object["x_stabilizers"], object["z_stabilizers"]],
                object["logical_xs"],
                object["logical_zs"],
            )
        else:
            partition_graph = nx.Graph()

            for i, j in itertools.combinations(object["stabilizers"], 2):
                conflict = False
                for pi, pj in zip(i, j):
                    if pi != "I" and pj != "I" and pi != pj:
                        conflict = True
                        break
                if conflict:
                    partition_graph.add_edge(i, j)

            coloring = nx.algorithms.greedy_color(partition_graph)

            partition: dict[Any, list[str]] = defaultdict(list)
            for stabilizer, color in coloring.items():
                partition[color].append(stabilizer)

            return QECCode(
                object["family"],
                object["n"],
                object["k"],
                object["d"],
                object["stabilizers"],
                list(partition.values()),
                object["logical_xs"],
                object["logical_zs"],
            )

    @staticmethod
    def from_string(string: str):
        return QECCode(**json.loads(string))

    def checks(self, parition_id: int):
        checks = list[PauliCheck]()

        offset = self.n
        for i in range(parition_id):
            offset += len(self.stabilizer_partition[i])

        for ancilla_base_id, stabilizer in enumerate(
            self.stabilizer_partition[parition_id]
        ):
            checks.extend(
                PauliCheck.from_stabilizer(stabilizer, ancilla_base_id + offset)
            )
        return checks
