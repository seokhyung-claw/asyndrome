import math
import pickle
import random
from pathlib import Path
from time import time
from typing import Literal
from rich.progress import track
from asyndrome.csscode import CSSCode, PauliCheck
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
    def __init__(
        self,
        *,
        iters_per_step: int,
        nshots: int,
        checkpoint_path: str | Path | None = None,
        checkpoint_every_steps: int = 1,
    ) -> None:
        super().__init__()
        self._iters_per_step = iters_per_step
        self._nshots = nshots
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self._checkpoint_every_steps = max(1, checkpoint_every_steps)

    def _tree_step(
        self,
        root: TreeNode,
        code: CSSCode,
        checks: list[PauliCheck],
        stabilizers: list[str],
        logicals: list[str],
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

            circuit = evaluate_circuit(
                code,
                self._sort_schedule(zip(checks, schedule)),
                stabilizers,
                logicals,
                error_model,
            )

            result = self._decoder_agent.simulate(circuit, self._nshots)  # type: ignore

            node.backpropagate(self._nshots / (result + 1))

        return root.best_child(exploration_weight=0)

    def _tree_search(
        self,
        initial_state: TreeState,
        code: CSSCode,
        checks: list[PauliCheck],
        stabilizers: list[str],
        logicals: list[str],
        error_model: ErrorModel,
        phase: Literal["X", "Z"],
        resume_node: TreeNode | None = None,
        x_ticks: np.ndarray | None = None,
    ):
        node = resume_node if resume_node is not None else TreeNode(initial_state)

        starting_time = time()

        print("Tree search summary:")
        print(f"    Iteration per step: {self._iters_per_step}")
        print(f"    Total steps: {len(checks)}")
        print(f"    Shots: {self._nshots}")
        print("Tree search starts:")

        while not node.is_terminal():
            single_step_start = time()
            node = self._tree_step(
                node, code, checks, stabilizers, logicals, error_model
            )
            node.parent = None
            single_step_end = time()
            depth, total = node.state.percentage()
            print(
                f"    {depth} / {total} done in {single_step_end - single_step_start:.3f}s ({single_step_end - starting_time:.3f}s)"
            )
            if self._checkpoint_path is not None and (
                depth % self._checkpoint_every_steps == 0 or node.is_terminal()
            ):
                self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._checkpoint_path, "wb") as checkpoint_file:
                    pickle.dump(
                        {
                            "phase": phase,
                            "node": node,
                            "random_state": random.getstate(),
                            "x_ticks": x_ticks,
                        },
                        checkpoint_file,
                    )

        return node.state

    def _schedule_check(
        self,
        check_pauli: Literal["X", "Z"],
        code: CSSCode,
        error_model: ErrorModel,
        resume_node: TreeNode | None = None,
        x_ticks: np.ndarray | None = None,
    ):
        # Pauli checks to be scheduled and necessary stabilizers, logicals
        checks = code.x_checks() if check_pauli == "X" else code.z_checks()
        stabilizers = code.z_stabilizers if check_pauli == "X" else code.x_stabilizers
        logicals = code.logical_zs if check_pauli == "X" else code.logical_xs

        schedule_state = self._tree_search(
            TreeState.initial_state(len(checks), code.n + code.ancillas),
            code,
            checks,
            stabilizers,
            logicals,
            error_model,
            check_pauli,
            resume_node,
            x_ticks,
        )

        return schedule_state.schedule

    def schedule(
        self, code: CSSCode, decoder: str, error_model: ErrorModel
    ) -> Schedule:
        checkpoint = None
        if self._checkpoint_path is not None and self._checkpoint_path.exists():
            with open(self._checkpoint_path, "rb") as checkpoint_file:
                checkpoint = pickle.load(checkpoint_file)

        with decoder_agent(decoder, (code.n, code.k, code.d)) as agent:
            self._decoder_agent = agent
            # first, schedule x checks
            if checkpoint is not None and checkpoint["phase"] == "Z":
                x_ticks = checkpoint["x_ticks"]
                x_ticks_max: int = np.max(x_ticks)
            else:
                print("Scheduling X checks")
                if checkpoint is not None and checkpoint["phase"] == "X":
                    random.setstate(checkpoint["random_state"])
                x_ticks = self._schedule_check(
                    "X",
                    code,
                    error_model,
                    checkpoint["node"] if checkpoint is not None and checkpoint["phase"] == "X" else None,
                )
                x_ticks_max = np.max(x_ticks)
                if self._checkpoint_path is not None:
                    self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._checkpoint_path, "wb") as checkpoint_file:
                        pickle.dump(
                            {
                                "phase": "Z",
                                "node": None,
                                "random_state": random.getstate(),
                                "x_ticks": x_ticks,
                            },
                            checkpoint_file,
                        )

            print("Scheduling Z checks")
            if checkpoint is not None and checkpoint["phase"] == "Z":
                random.setstate(checkpoint["random_state"])
            z_ticks = self._schedule_check(
                "Z",
                code,
                error_model,
                checkpoint["node"] if checkpoint is not None and checkpoint["phase"] == "Z" else None,
                x_ticks,
            )
            z_ticks += (
                x_ticks_max + 1
            )  # advance the ticks by x_max + 1 to avoid conflict, maybe there's a better way to merge?
            self._decoder_agent = None

        if self._checkpoint_path is not None and self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

        all_ticks = x_ticks.astype(int).tolist() + z_ticks.astype(int).tolist()
        all_checks = code.x_checks() + code.z_checks()

        return self._sort_schedule(list(zip(all_checks, all_ticks)))
