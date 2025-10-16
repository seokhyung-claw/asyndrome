from itertools import combinations
import itertools
import pulp

from asyndrome.csscode import PauliCheck, CSSCode
from asyndrome.scheduler import Schedule, Scheduler
from asyndrome.stimcirc import ErrorModel


class CheckTick:
    def __init__(self, check: PauliCheck) -> None:
        self._check = check
        self._name = f"{check.data}{check.pauli}{check.ancilla}"
        self._var = pulp.LpVariable(self._name, cat="Integer", lowBound=0)

    def conflict(self, other: "CheckTick"):
        return (
            self._check.ancilla == other._check.ancilla
            or self._check.data == other._check.data
        )

    @property
    def tick(self) -> int:
        return int(self._var.varValue)  # type: ignore


def pulp_neq(
    prob: pulp.LpProblem, name: str, x: pulp.LpVariable, y: pulp.LpVariable, m: int
):
    delta = pulp.LpVariable(name, cat="Binary")  # Create the binary variable
    prob += x - y >= 1 + m * (delta - 1)
    prob += y - x >= 1 + m * (-delta)


def pulp_production(
    prob: pulp.LpProblem,
    name: str,
    tx: list[pulp.LpVariable],
    tz: list[pulp.LpVariable],
    m: int,
):
    n = len(tx)
    Z = [pulp.LpVariable(f"{name}_z{i}", cat=pulp.LpBinary) for i in range(n)]
    k = pulp.LpVariable(f"{name}_k", lowBound=0, cat=pulp.LpInteger)
    for i in range(n):
        factor = tx[i] - tz[i]
        prob += factor + m * Z[i] >= 1
        prob += factor - m * (1 - Z[i]) <= -1
    prob += pulp.lpSum(Z) == 2 * k


class CheckVars:
    def __init__(self, stabilizers: list[str], ancilla_offset: int) -> None:
        self._stabilizers = stabilizers
        self._checks: list[dict[int, CheckTick]] = []
        for i, stabilizer in enumerate(stabilizers):
            self._checks.append(
                {
                    chk.data: CheckTick(chk)
                    for chk in PauliCheck.from_stabilizer(
                        stabilizer, i + ancilla_offset
                    )
                }
            )

    def iterate(self):
        for stabilizer, checks in zip(self._stabilizers, self._checks):
            yield (stabilizer, checks)

    @property
    def all_checks(self):
        return list(itertools.chain.from_iterable([x.values() for x in self._checks]))


class BaselineScheduler(Scheduler):
    def __init__(
        self, logpath: str | None = None, timeout: int = 24 * 60 * 60, M: int = 1000000
    ) -> None:
        super().__init__()
        self._logpath = logpath
        self._M = M
        self._timeout = timeout

    def schedule(
        self, code: CSSCode, decoder: str, error_model: ErrorModel
    ) -> Schedule:
        problem = pulp.LpProblem()

        # just a super large upper bound for ticks
        maxtick = pulp.LpVariable("maxtick", cat="Integer", lowBound=0, upBound=self._M)

        # tick variables
        x_checkvars = CheckVars(code.x_stabilizers, code.n)
        z_checkvars = CheckVars(code.z_stabilizers, code.n + len(code.x_stabilizers))

        # TODO: add the commute relationship
        for xi, (x_stab, x_chk) in enumerate(x_checkvars.iterate()):
            for zi, (z_stab, z_chk) in enumerate(z_checkvars.iterate()):
                # common, overlapped data qubits
                overlap_indexes = [
                    i
                    for i, (px, pz) in enumerate(zip(x_stab, z_stab))
                    if px != "I" and pz != "I"
                ]

                # no overlap, nothing happens
                if len(overlap_indexes) == 0:
                    continue

                # otherwise:
                # Prod(tx(q) - tz(q), q in overlap) > 0
                pulp_production(
                    problem,
                    f"x{xi}overlap{zi}",
                    [x_chk[dq]._var for dq in overlap_indexes],
                    [z_chk[dq]._var for dq in overlap_indexes],
                    self._M,
                )

        # constraints on all Pauli checks
        all_checkvars = x_checkvars.all_checks + z_checkvars.all_checks

        # non conflict conditions, two checks with the same data/ancilla cannot be the same tick
        for c1, c2 in combinations(all_checkvars, 2):
            if c1.conflict(c2):
                pulp_neq(problem, f"{c1._name}<>{c2._name}", c1._var, c2._var, self._M)

        # scope conditions, we want to minimize the max tick for the lowest depth
        for chk in all_checkvars:
            problem += chk._var <= maxtick
        problem += maxtick

        # solve the problem
        problem.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                logPath=self._logpath,
                timeLimit=self._timeout,
                threads=64,
            )
        )

        # extract the result
        return self._sort_schedule([(ct._check, ct.tick) for ct in all_checkvars])
