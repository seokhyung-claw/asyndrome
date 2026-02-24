from asyndrome.bbcodeibm import ibm_syndrome_measurement
from asyndrome.csscode import CSSCode
from asyndrome.scheduler import Schedule, Scheduler
from asyndrome.stimcirc import ErrorModel, StimCircuit, decoder_agent


def find_index(string: str, char: str):
    return [pos for pos, c in enumerate(string) if c == char]


class GoogleScheduler(Scheduler):
    def __init__(self, nrow: int, ncol: int) -> None:
        super().__init__()
        self._nrow = nrow
        self._ncol = ncol

    def schedule(
        self, code: CSSCode, decoder: str, error_model: ErrorModel
    ) -> Schedule:
        assert self._nrow * self._ncol == code.n
        assert code.k == 1
        assert code.d == min(self._nrow, self._ncol)

        patches: list[tuple[str, list[int]]] = []

        # sort the patches
        for stabilizer in code.x_stabilizers + code.z_stabilizers:
            if "X" in stabilizer:
                pauli = "X"
            else:
                pauli = "Z"

            data_qubits = find_index(stabilizer, pauli)
            data_qubits.sort()

            if len(data_qubits) == 2:
                [a, b] = data_qubits

                # first, top edge
                if a < self._ncol and b < self._ncol:
                    data_qubits = [-1, -1, a, b]
                elif a % self._ncol == 0 and b % self._ncol == 0:
                    data_qubits = [-1, a, -1, b]
                elif (a + 1) % self._ncol == 0 and (b + 1) % self._ncol == 0:
                    data_qubits = [a, -1, b, -1]
                else:
                    data_qubits = [a, b, -1, -1]

            patches.append((pauli, data_qubits))

        ticks = []

        reorder_table = {"X": [2, 1, 3, 0], "Z": [0, 1, 3, 2]}

        checks = code.x_checks() + code.z_checks()

        for meas in checks:
            patch_id = meas.ancilla - code.n
            pauli, patch = patches[patch_id]

            index_in_patch = patch.index(meas.data)

            tick = reorder_table[pauli][index_in_patch]

            ticks.append((meas, tick))

        return self._sort_schedule(ticks)


class TrivialScheduler(Scheduler):
    def __init__(self, nrow: int, ncol: int) -> None:
        super().__init__()
        self._nrow = nrow
        self._ncol = ncol

    def schedule(
        self, code: CSSCode, decoder: str, error_model: ErrorModel
    ) -> Schedule:
        assert self._nrow * self._ncol == code.n
        assert code.k == 1
        assert code.d == min(self._nrow, self._ncol)

        patches: list[tuple[str, list[int]]] = []

        # sort the patches
        for stabilizer in code.x_stabilizers + code.z_stabilizers:
            if "X" in stabilizer:
                pauli = "X"
            else:
                pauli = "Z"

            data_qubits = find_index(stabilizer, pauli)
            data_qubits.sort()

            if len(data_qubits) == 2:
                [a, b] = data_qubits

                # first, top edge
                if a < self._ncol and b < self._ncol:
                    data_qubits = [-1, -1, a, b]
                elif a % self._ncol == 0 and b % self._ncol == 0:
                    data_qubits = [-1, a, -1, b]
                elif (a + 1) % self._ncol == 0 and (b + 1) % self._ncol == 0:
                    data_qubits = [a, -1, b, -1]
                else:
                    data_qubits = [a, b, -1, -1]

            patches.append((pauli, data_qubits))

        ticks = []

        reorder_table = {"X": [1, 2, 4, 3], "Z": [1, 2, 4, 3]}

        checks = code.x_checks() + code.z_checks()

        for meas in checks:
            patch_id = meas.ancilla - code.n
            pauli, patch = patches[patch_id]

            index_in_patch = patch.index(meas.data)

            tick = reorder_table[pauli][index_in_patch]

            ticks.append((meas, tick))

        return self._sort_schedule(ticks)


def bbcode_parameter(bbcode_n: int):
    if bbcode_n == 144:
        k, d = 12, 12
    elif bbcode_n == 784:
        k, d = 24, 24
    elif bbcode_n == 72:
        k, d = 12, 6
    elif bbcode_n == 90:
        k, d = 8, 10
    elif bbcode_n == 108:
        k, d = 8, 10
    elif bbcode_n == 288:
        k, d = 12, 18
    else:
        raise ValueError(f"bbcode-{bbcode_n} does not exist")

    return bbcode_n, k, d


class IBMEvaluator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def evaluate(n: int, decoder: str, error_model: ErrorModel, nshots: int):
        x_circuit, z_circuit = StimCircuit(), StimCircuit()
        x_circuit = ibm_syndrome_measurement(n, x_circuit, error_model, "X")
        z_circuit = ibm_syndrome_measurement(n, z_circuit, error_model, "Z")

        with decoder_agent(decoder, bbcode_parameter(n)) as agent:
            xflips = agent.simulate(x_circuit, nshots)
            zflips = agent.simulate(z_circuit, nshots)

        return xflips / nshots, zflips / nshots
