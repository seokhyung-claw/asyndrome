import numpy as np
from mqt.qecc.codes.css_code import CSSCode
from asyndrome.stimcirc import StimCircuit, StimMeasurement, ErrorModel


def ideal_measures(csscode: CSSCode, circuit: StimCircuit):
    measures: list[StimMeasurement] = []
    for stabilizer in csscode.stabs_as_pauli_strings():
        measures.append(circuit.measure_pauli(stabilizer))
    return measures


# Takes as input a binary square matrix A
# Returns the rank of A over the binary field F_2
def rank2(A):
    rows, n = A.shape
    X = np.identity(n, dtype=int)

    for i in range(rows):
        y = np.dot(A[i, :], X) % 2
        not_y = (y + 1) % 2
        good = X[:, np.nonzero(not_y)]
        good = good[:, 0, :]
        bad = X[:, np.nonzero(y)]
        bad = bad[:, 0, :]
        if bad.shape[1] > 0:
            bad = np.add(bad, np.roll(bad, 1, axis=1))
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]


def ibm_syndrome_measurement(
    bbcode_n: int, circuit: StimCircuit, error_model: ErrorModel, logic: str
):
    if bbcode_n == 144:
        k, d = 12, 12
        ell, m = 12, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
    elif bbcode_n == 784:
        k, d = 24, 24
        ell, m = 28, 14
        a1, a2, a3 = 26, 6, 8
        b1, b2, b3 = 7, 9, 20
    elif bbcode_n == 72:
        k, d = 12, 6
        ell, m = 6, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
    elif bbcode_n == 90:
        k, d = 8, 10
        ell, m = 15, 3
        a1, a2, a3 = 9, 1, 2
        b1, b2, b3 = 0, 2, 7
    elif bbcode_n == 108:
        k, d = 8, 10
        ell, m = 9, 6
        a1, a2, a3 = 3, 1, 2
        b1, b2, b3 = 3, 1, 2
    elif bbcode_n == 288:
        k, d = 12, 18
        ell, m = 12, 12
        a1, a2, a3 = 3, 2, 7
        b1, b2, b3 = 3, 1, 2
    else:
        raise ValueError(f"bbcode-{bbcode_n} does not exist")

    # code length
    n = 2 * m * ell
    n2 = m * ell

    sX = ["idle", 1, 4, 3, 5, 0, 2]
    sZ = [3, 5, 0, 1, 2, 4, "idle"]

    # Compute check matrices of X- and Z-checks

    # cyclic shift matrices
    I_ell = np.identity(ell, dtype=int)
    I_m = np.identity(m, dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for i in range(m):
        y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))

    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2

    A1 = x[a1]
    A2 = y[a2]
    A3 = y[a3]
    B1 = y[b1]
    B2 = x[b2]
    B3 = x[b3]

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A, B))
    hz = np.hstack((BT, AT))

    # number of logical qubits
    k = n - rank2(hx) - rank2(hz)

    qcode = CSSCode(d, hx, hz)

    lin_order = {}
    data_qubits = []
    Xchecks = []
    Zchecks = []

    cnt = 0
    for i in range(n2):
        node_name = ("Xcheck", i)
        Xchecks.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1

    for i in range(n2):
        node_name = ("data_left", i)
        data_qubits.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1
    for i in range(n2):
        node_name = ("data_right", i)
        data_qubits.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1

    for i in range(n2):
        node_name = ("Zcheck", i)
        Zchecks.append(node_name)
        lin_order[node_name] = cnt
        cnt += 1

    # compute the list of neighbors of each check qubit in the Tanner graph
    nbs = {}
    # iterate over X checks
    for i in range(n2):
        check_name = ("Xcheck", i)
        # left data qubits
        nbs[(check_name, 0)] = ("data_left", np.nonzero(A1[i, :])[0][0])
        nbs[(check_name, 1)] = ("data_left", np.nonzero(A2[i, :])[0][0])
        nbs[(check_name, 2)] = ("data_left", np.nonzero(A3[i, :])[0][0])
        # right data qubits
        nbs[(check_name, 3)] = ("data_right", np.nonzero(B1[i, :])[0][0])
        nbs[(check_name, 4)] = ("data_right", np.nonzero(B2[i, :])[0][0])
        nbs[(check_name, 5)] = ("data_right", np.nonzero(B3[i, :])[0][0])

    # iterate over Z checks
    for i in range(n2):
        check_name = ("Zcheck", i)
        # left data qubits
        nbs[(check_name, 0)] = ("data_left", np.nonzero(B1[:, i])[0][0])
        nbs[(check_name, 1)] = ("data_left", np.nonzero(B2[:, i])[0][0])
        nbs[(check_name, 2)] = ("data_left", np.nonzero(B3[:, i])[0][0])
        # right data qubits
        nbs[(check_name, 3)] = ("data_right", np.nonzero(A1[:, i])[0][0])
        nbs[(check_name, 4)] = ("data_right", np.nonzero(A2[:, i])[0][0])
        nbs[(check_name, 5)] = ("data_right", np.nonzero(A3[:, i])[0][0])

    def target_to_index(target: tuple[str, int]):
        type, index = target
        if type == "data_left":
            return int(index)
        elif type == "data_right":
            return int(index + n2)
        elif type == "Xcheck":
            return int(index + n)
        elif type == "Zcheck":
            return int(index + n + n2)
        else:
            return -1

    def check_to_pauli(check: tuple[str, int]):
        targets = []
        for i in range(6):
            targets.append(target_to_index(nbs[(check, i)]))
        pauli = ["I" for _ in range(n)]
        for t in targets:
            pauli[t] = check[0][0]
        return "".join(pauli)

    stabilizers = []
    for c in Xchecks:
        stabilizers.append(check_to_pauli(c))
    for c in Zchecks:
        stabilizers.append(check_to_pauli(c))

    assert qcode.stabs_as_pauli_strings() == stabilizers

    # syndrome measurement cycle as a list of operations
    cycle = []
    U = np.identity(2 * n, dtype=int)
    # round 0: prep xchecks, CNOT zchecks and data
    t = 0
    for q in Xchecks:
        cycle.append(("PrepX", q))
    data_qubits_cnoted_in_this_round = []
    assert not (sZ[t] == "idle")
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target, direction)]
        U[lin_order[target], :] = (
            U[lin_order[target], :] + U[lin_order[control], :]
        ) % 2
        data_qubits_cnoted_in_this_round.append(control)
        cycle.append(("CNOT", control, target))
    for q in data_qubits:
        if not (q in data_qubits_cnoted_in_this_round):
            cycle.append(("IDLE", q))

    # round 1-5: CNOT xchecks and data, CNOT zchecks and data
    for t in range(1, 6):
        assert not (sX[t] == "idle")
        for control in Xchecks:
            direction = sX[t]
            target = nbs[(control, direction)]
            U[lin_order[target], :] = (
                U[lin_order[target], :] + U[lin_order[control], :]
            ) % 2
            cycle.append(("CNOT", control, target))
        assert not (sZ[t] == "idle")
        for target in Zchecks:
            direction = sZ[t]
            control = nbs[(target, direction)]
            U[lin_order[target], :] = (
                U[lin_order[target], :] + U[lin_order[control], :]
            ) % 2
            cycle.append(("CNOT", control, target))

    # round 6: CNOT xchecks and data, measure Z checks
    t = 6
    for q in Zchecks:
        cycle.append(("MeasZ", q))
    assert not (sX[t] == "idle")
    data_qubits_cnoted_in_this_round = []
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control, direction)]
        U[lin_order[target], :] = (
            U[lin_order[target], :] + U[lin_order[control], :]
        ) % 2
        cycle.append(("CNOT", control, target))
        data_qubits_cnoted_in_this_round.append(target)
    for q in data_qubits:
        if not (q in data_qubits_cnoted_in_this_round):
            cycle.append(("IDLE", q))

    # round 7: all data qubits are idle, Prep Z checks, Meas X checks
    for q in data_qubits:
        cycle.append(("IDLE", q))
    for q in Xchecks:
        cycle.append(("MeasX", q))
    for q in Zchecks:
        cycle.append(("PrepZ", q))

    logicals = (
        qcode.x_logicals_as_pauli_strings()
        if logic == "X"
        else qcode.z_logicals_as_pauli_strings()
    )

    first_round = ideal_measures(qcode, circuit)
    first_ls = []
    for string in logicals:
        first_ls.append(circuit.measure_pauli(string))

    for gate, *targets in cycle:
        if gate == "PrepX":
            t = target_to_index(targets[0])
            circuit.gate("RX", t)
        elif gate == "PrepZ":
            t = target_to_index(targets[0])
            circuit.gate("RZ", t)
        elif gate == "CNOT":
            gate_c = target_to_index(targets[0])
            gate_t = target_to_index(targets[1])
            circuit.gate("CNOT", [gate_c, gate_t])

            if gate_c >= n:
                error_model.cnot(gate_c, circuit)
            if gate_t >= n:
                error_model.cnot(gate_t, circuit)
        elif gate == "IDLE":
            t = target_to_index(targets[0])
            circuit.gate("I", t)
            if t >= n:
                error_model.idling(t, circuit)
        elif gate == "MeasZ":
            t = target_to_index(targets[0])
            circuit.measures("MZ", [t])
        elif gate == "MeasX":
            t = target_to_index(targets[0])
            circuit.measures("MX", [t])
        else:
            raise ValueError(f"unknown gate {gate}")

    second_round = ideal_measures(qcode, circuit)
    second_ls = []
    for string in logicals:
        second_ls.append(circuit.measure_pauli(string))

    for i, (a, b) in enumerate(zip(first_round, second_round)):
        circuit.detector([a, b], i)

    # corresponding observable
    for i, (first_l, second_l) in enumerate(zip(first_ls, second_ls)):
        circuit.observable([first_l, second_l], i)

    return circuit
