import math
import random
from time import time
from rich.progress import track
from asyndrome.qeccode import QECCode, PauliCheck
from asyndrome.scheduler import Schedule, Scheduler, evaluate_circuit

from dataclasses import dataclass
import numpy as np

from asyndrome.stimcirc import ErrorModel, decoder_agent


@dataclass(slots=True)
class TreeState:
    schedule: np.ndarray
    maxticks: np.ndarray

    @staticmethod
    def initial_state(nchecks: int, nqubits: int):
        return TreeState(np.repeat(-1, nchecks), np.repeat(-1, nqubits))

    def shift(self, checks: list[PauliCheck], meas_index: int):
        chk = checks[meas_index]
        new_tick = max(self.maxticks[chk.data], self.maxticks[chk.ancilla]) + 1

        new_schedule = self.schedule.copy()
        new_maxticks = self.maxticks.copy()

        new_maxticks[chk.data] = new_tick
        new_maxticks[chk.ancilla] = new_tick
        new_schedule[meas_index] = new_tick

        return TreeState(new_schedule, new_maxticks)

    def transitions(self) -> list[int]:
        states = []
        for meas_index, tick in enumerate(self.schedule):
            if tick == -1:  # unmeasured syndrome measurement
                states.append(meas_index)
        return states

    def is_terminal(self) -> bool:
        return min(self.schedule) != -1

    def percentage(self) -> tuple[int, int]:
        total = len(self.schedule)
        return (total - np.count_nonzero(self.schedule == -1)), total


class TreeNode:
    def __init__(self, state: TreeState, parent: "TreeNode | None" = None):
        self.state = state

        self.parent = parent
        self.children: list["TreeNode"] = []

        self.visits = 0
        self.value = 0.0

        self.unvisited = state.transitions()

    def is_fully_expanded(self):
        return len(self.unvisited) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def expand(self, checks: list[PauliCheck]):
        next_state = self.state.shift(checks, self.unvisited.pop())
        child_node = TreeNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight=1.4):
        def ucb_score(child):
            if child.visits == 0:
                return float("inf")
            return child.value / child.visits + exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )

        return max(self.children, key=ucb_score)

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

    def simulate_schedule(self, checks: list[PauliCheck]):
        current_state = self.state
        while not current_state.is_terminal():
            current_state = current_state.shift(
                checks, random.choice(current_state.transitions())
            )
        return current_state.schedule

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()


class AlphaScheduler(Scheduler):
    def __init__(self, *, iters_per_step: int, nshots: int) -> None:
        super().__init__()
        self._iters_per_step = iters_per_step
        self._nshots = nshots

    def _tree_step(
        self,
        root: TreeNode,
        code: QECCode,
        checks: list[PauliCheck],
        stabilizers: list[str],
        error_model: ErrorModel,
    ):
        iterations = max(0, self._iters_per_step - root.visits)

        depth, total = root.state.percentage()

        for _ in track(
            range(iterations),
            description=f"{iterations} iters at {depth} / {total} step",
            transient=True,
        ):
            node = root

            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            if not node.is_terminal():
                node = node.expand(checks)

            schedule = node.simulate_schedule(checks)

            x_circuit = evaluate_circuit(
                code,
                self._sort_schedule(zip(checks, schedule)),
                stabilizers,
                code.logical_xs,
                error_model,
            )

            z_circuit = evaluate_circuit(
                code,
                self._sort_schedule(zip(checks, schedule)),
                stabilizers,
                code.logical_zs,
                error_model,
            )

            x_result = self._decoder_agent.simulate(x_circuit, self._nshots)  # type: ignore
            z_result = self._decoder_agent.simulate(z_circuit, self._nshots)  # type: ignore

            node.backpropagate(self._nshots / (max(x_result, z_result) + 1))

        return root.best_child(exploration_weight=0)

    def _tree_search(
        self,
        initial_state: TreeState,
        code: QECCode,
        checks: list[PauliCheck],
        stabilizers: list[str],
        error_model: ErrorModel,
    ):
        node = TreeNode(initial_state)

        starting_time = time()

        print("Tree search summary:")
        print(f"    Iteration per step: {self._iters_per_step}")
        print(f"    Total steps: {len(checks)}")
        print(f"    Shots: {self._nshots}")
        print("Tree search starts:")

        while not node.is_terminal():
            single_step_start = time()
            node = self._tree_step(node, code, checks, stabilizers, error_model)
            single_step_end = time()
            depth, total = node.state.percentage()
            print(
                f"    {depth} / {total} done in {single_step_end - single_step_start:.3f}s ({single_step_end - starting_time:.3f}s)"
            )

        return node.state

    def _schedule_check(self, parition_id: int, code: QECCode, error_model: ErrorModel):
        # Pauli checks to be scheduled and necessary stabilizers, logicals
        checks = code.checks(parition_id)
        stabilizers = code.stabilizer_partition[parition_id]

        schedule_state = self._tree_search(
            TreeState.initial_state(len(checks), code.n + code.ancillas),
            code,
            checks,
            stabilizers,
            error_model,
        )

        return schedule_state.schedule

    def schedule(
        self, code: QECCode, decoder: str, error_model: ErrorModel
    ) -> Schedule:
        all_ticks = []
        all_checks = []
        offset = 0

        with decoder_agent(decoder, (code.n, code.k, code.d)) as agent:
            self._decoder_agent = agent
            for i, _ in enumerate(code.stabilizer_partition):
                print(
                    f"Scheduling partition {i + 1}/{len(code.stabilizer_partition)} partitions"
                )

                t = self._schedule_check(i, code, error_model)
                t += offset
                offset = max(t)

                all_ticks += t.astype(int).tolist()
                all_checks += code.checks(i)

        return self._sort_schedule(list(zip(all_checks, all_ticks)))
