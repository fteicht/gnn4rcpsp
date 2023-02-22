import dataclasses
import os
from collections import defaultdict, namedtuple
from copy import deepcopy
from enum import Enum
from time import perf_counter
from typing import Dict, Hashable, List, Optional, Set, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import uniform
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws,
    tree,
)
from discrete_optimization.rcpsp.rcpsp_parser import parse_file
from discrete_optimization.rcpsp.rcpsp_utils import compute_nice_resource_consumption
from graph import Graph
from infer_schedules import build_rcpsp_model, build_rcpsp_model_skdecide
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pp
from models import ResTransformer
from ortools.sat.python import cp_model

CPSAT_TIME_LIMIT = 10  # 10 s
MAX_DEVIATION = 0.8

Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")


class Scheduler(Enum):
    CPSAT = 1
    SGS = 2


class ExecutionMode(Enum):
    REACTIVE_AVERAGE = 1
    REACTIVE_WORST = 2
    REACTIVE_BEST = 3
    HINDSIGHT_LEX = 4  # Select first most appearing top tasks, then break tie by lowest average makespan
    HINDSIGHT_DBP = 5  # Extract first potential best tasks, then reevaluate each of them on each scenario and select top tasks by lowest average makespan


@dataclasses.dataclass
class CPSatSpecificParams:
    do_minimization: bool
    warm_start_with_gnn: bool
    time_limit_seconds: float
    num_workers: int

    @staticmethod
    def default():
        return CPSatSpecificParams(
            do_minimization=False,
            warm_start_with_gnn=True,
            time_limit_seconds=1,
            num_workers=os.cpu_count(),
        )

    # good for experimentation
    @staticmethod
    def default_cp_reactive():
        return CPSatSpecificParams(
            do_minimization=True,
            warm_start_with_gnn=False,
            time_limit_seconds=2,
            num_workers=os.cpu_count(),
        )


class ParamsRemainingRCPSP(Enum):
    EXACT_REMAINING = 0
    KEEP_FULL_RCPSP = 1


class MarkovRCPSP:
    def __init__(self, base_rcpsp_model, max_deviation=0.8) -> None:
        self._base_rcpsp_model = base_rcpsp_model
        self._max_deviation = max_deviation
        self._sampled_rcpsps = {}

    def create_uncertain_rcpsp_model(self, launched_tasks: Set, rcpsp: RCPSPModel):
        assert set(launched_tasks).issubset(set(self._base_rcpsp_model.tasks_list))
        fet = frozenset(launched_tasks)
        if fet in self._sampled_rcpsps:
            poisson_laws = self._sampled_rcpsps[fet]
        else:
            # Compute standard deviation of remaining task durations based on the executed or running (aka launched) tasks
            deviation = uniform.rvs(loc=0, scale=self._max_deviation)
            # Create UncertainRCPSP with random durations only for the very next lanchable tasks
            front_tasks = self.get_front_tasks(launched_tasks=launched_tasks)
            poisson_laws = self.create_poisson_laws(
                front_tasks,
                range_per_around_mean_duration=deviation,
                range_per_around_mean_resource=0,
                do_uncertain_range_duration=False,
                do_uncertain_range_resource=False,
                do_uncertain_duration=True,
                do_uncertain_resource=False,
            )
            self._sampled_rcpsps[fet] = poisson_laws
        uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=rcpsp,
            poisson_laws={
                task: laws
                for task, laws in poisson_laws.items()
                if task in rcpsp.mode_details
            },
            uniform_law=True,
        )
        return uncertain_rcpsp

    def get_front_tasks(self, launched_tasks):
        if len(launched_tasks) == 0:
            front_tasks = set()
            for succ in self._base_rcpsp_model.successors.values():
                front_tasks = front_tasks | set(succ)
            front_tasks = set(self._base_rcpsp_model.mode_details.keys()) - front_tasks
        else:
            front_tasks = set()
            for task in launched_tasks:
                front_tasks = front_tasks | set(self._base_rcpsp_model.successors[task])
            front_tasks = front_tasks - set(launched_tasks)
        return front_tasks

    def create_poisson_laws_duration(
        self, tasks, range_per_around_mean=0.5, do_uncertain_range=True
    ):
        poisson_dict = {}
        source = self._base_rcpsp_model.source_task
        sink = self._base_rcpsp_model.sink_task
        for job in self._base_rcpsp_model.mode_details:
            if job in tasks:
                poisson_dict[job] = {}
                for mode in self._base_rcpsp_model.mode_details[job]:
                    poisson_dict[job][mode] = {}
                    duration = self._base_rcpsp_model.mode_details[job][mode][
                        "duration"
                    ]
                    if job in {source, sink}:
                        poisson_dict[job][mode]["duration"] = (
                            duration,
                            duration,
                            duration,
                        )
                    else:
                        rng_per_around_mean = (
                            uniform.rvs(loc=0, scale=range_per_around_mean)
                            if do_uncertain_range
                            else range_per_around_mean
                        )
                        min_duration = max(
                            1, int(round((1.0 - rng_per_around_mean) * duration))
                        )
                        max_duration = int(
                            round((1.0 + rng_per_around_mean) * duration)
                        )
                        poisson_dict[job][mode]["duration"] = (
                            min_duration,
                            duration,
                            max_duration,
                        )
        return poisson_dict

    def create_poisson_laws_resource(
        self, tasks, range_per_around_mean=0.5, do_uncertain_range=True
    ):
        poisson_dict = {}
        source = self._base_rcpsp_model.source_task
        sink = self._base_rcpsp_model.sink_task
        limit_resource = self._base_rcpsp_model.resources
        resources_non_renewable = self._base_rcpsp_model.non_renewable_resources
        for job in self._base_rcpsp_model.mode_details:
            if job in tasks:
                poisson_dict[job] = {}
                for mode in self._base_rcpsp_model.mode_details[job]:
                    poisson_dict[job][mode] = {}
                    for resource in self._base_rcpsp_model.mode_details[job][mode]:
                        if resource == "duration":
                            continue
                        if resource in resources_non_renewable:
                            continue
                        resource_consumption = self._base_rcpsp_model.mode_details[job][
                            mode
                        ][resource]
                        if job in {source, sink}:
                            poisson_dict[job][mode][resource] = (
                                resource_consumption,
                                resource_consumption,
                                resource_consumption,
                            )
                        else:
                            rng_per_around_mean = (
                                uniform.rvs(loc=0, scale=range_per_around_mean)
                                if do_uncertain_range
                                else range_per_around_mean
                            )
                            min_rc = max(
                                0,
                                int(
                                    round(
                                        (1.0 - rng_per_around_mean)
                                        * resource_consumption
                                    )
                                ),
                            )
                            max_rc = min(
                                int(
                                    round(
                                        (1.0 + rng_per_around_mean)
                                        * resource_consumption
                                    )
                                ),
                                limit_resource[resource],
                            )
                            poisson_dict[job][mode][resource] = (
                                min_rc,
                                resource_consumption,
                                max_rc,
                            )
        return poisson_dict

    def create_poisson_laws(
        self,
        tasks,
        range_per_around_mean_resource: float = 0.5,
        range_per_around_mean_duration: float = 0.5,
        do_uncertain_range_resource: bool = True,
        do_uncertain_range_duration: bool = True,
        do_uncertain_resource: bool = True,
        do_uncertain_duration: bool = True,
    ):
        poisson_laws = tree()
        if do_uncertain_duration:
            poisson_laws_duration = self.create_poisson_laws_duration(
                tasks,
                range_per_around_mean=range_per_around_mean_duration,
                do_uncertain_range=do_uncertain_range_duration,
            )
            for job in poisson_laws_duration:
                for mode in poisson_laws_duration[job]:
                    for res in poisson_laws_duration[job][mode]:
                        poisson_laws[job][mode][res] = poisson_laws_duration[job][mode][
                            res
                        ]
        if do_uncertain_resource:
            poisson_laws_resource = self.create_poisson_laws_resource(
                tasks,
                range_around_mean=range_per_around_mean_resource,
                do_uncertain_range=do_uncertain_range_resource,
            )
            for job in poisson_laws_resource:
                for mode in poisson_laws_resource[job]:
                    for res in poisson_laws_resource[job][mode]:
                        poisson_laws[job][mode][res] = poisson_laws_resource[job][mode][
                            res
                        ]
        return poisson_laws


class SchedulingExecutor:
    def __init__(
        self,
        rcpsp: RCPSPModel,
        model: torch.nn.Module,
        device: torch.device,
        scheduler: Scheduler,
        mode: ExecutionMode,
        samples: int = 100,
        deadline: int = None,
        params_cp: Optional[CPSatSpecificParams] = None,
        params_remaining_rcpsp: Optional[ParamsRemainingRCPSP] = None,
        debug_logs=False,
    ):
        self._debug_logs = debug_logs
        self._rcpsp = rcpsp
        self._model = model
        self._device = device
        self._scheduler = scheduler
        self._mode = mode
        self._simulated_rcpsp = None
        self._executed_schedule = None
        self._current_time = None
        self._samples = samples
        self._deadline = deadline
        self._params_cp = params_cp
        if params_cp is None:
            self._params_cp = CPSatSpecificParams.default_cp_reactive()
        self._params_remaining_rcpsp = params_remaining_rcpsp
        if params_remaining_rcpsp is None:
            self._params_remaining_rcpsp = ParamsRemainingRCPSP.EXACT_REMAINING

    def reset(
        self, sim_rcpsp: Optional[MarkovRCPSP] = None
    ) -> Tuple[int, RCPSPSolution]:
        """Samples a new RCPSP (uncertain task durations and resource capacities)
        and returns an empty schedule and current time set to 0"""
        if sim_rcpsp is None:
            self._simulated_rcpsp = MarkovRCPSP(
                base_rcpsp_model=self._rcpsp, max_deviation=MAX_DEVIATION
            )
        else:
            self._simulated_rcpsp = sim_rcpsp
        self._executed_schedule = RCPSPSolution(
            problem=deepcopy(self._rcpsp),
            rcpsp_permutation=[],
            standardised_permutation=[],
            rcpsp_schedule={},
        )
        self._current_time = 0
        return self._executed_schedule, self._current_time

    def progress(
        self, next_tasks: Set[int], next_start: int, expected_schedule: RCPSPSolution
    ) -> Tuple[int, RCPSPSolution, bool]:
        """Takes the next tasks to execute, their starting date and the expected schedule,
        and returns the new current time and updated executed schedule
        and a boolean indicating whether all the tasks have been scheduled or not"""

        # Add next_task to the executed schedule if it is scheduled to start now
        if self._debug_logs:
            print(
                "next start , ",
                next_start,
                " next tasks ",
                next_tasks,
                " current time ",
                self._current_time,
            )
        if next_start == self._current_time:
            sampled_rcpsp = self._simulated_rcpsp.create_uncertain_rcpsp_model(
                launched_tasks=set(self._executed_schedule.rcpsp_schedule.keys()),
                rcpsp=self._rcpsp,
            ).create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )
            self._executed_schedule.rcpsp_schedule.update(
                {
                    next_task: {
                        "start_time": next_start,
                        "end_time": next_start
                        + int(  # in case the max() below returns a numpy type
                            max(
                                mode["duration"]
                                for mode in sampled_rcpsp.mode_details[
                                    next_task
                                ].values()
                            )
                        ),
                    }
                    for next_task in next_tasks
                }
            )
            # Update the mode details of the started tasks
            for task in next_tasks:
                self._executed_schedule.problem.mode_details[
                    task
                ] = sampled_rcpsp.mode_details[task]
                self._executed_schedule.problem.update_functions()
        # Compute the next event, i.e. smallest executed task ending date or expected task starting date
        filtered_starts = [
            sched["start_time"]
            for sched in expected_schedule.rcpsp_schedule.values()
            if sched["start_time"] > self._current_time
        ]
        filtered_ends = [
            sched["end_time"]
            for sched in self._executed_schedule.rcpsp_schedule.values()
            if sched["end_time"] > self._current_time
        ]
        if len(filtered_starts) > 0 and len(filtered_ends) > 0:
            self._current_time = min(min(filtered_starts), min(filtered_ends))
        elif len(filtered_starts) > 0:
            self._current_time = min(filtered_starts)
        elif len(filtered_ends) > 0:
            self._current_time = min(filtered_ends)
        else:
            return (self._current_time, self._executed_schedule, True)

        # Return the new current time and updated executed schedule
        return (
            self._current_time,
            self._executed_schedule,
            len(self._executed_schedule.rcpsp_schedule)
            == len(self._executed_schedule.problem.mode_details),
        )

    def next_tasks(
        self, executed_schedule: RCPSPSolution, current_time: int
    ) -> Tuple[Set[int], int, int, RCPSPSolution]:
        """
        Takes a schedule which has been executed (i.e. list of committed start dates for some tasks),
        the current time, and returns the next tasks to start, their starting date, the expected makespan
        and the full updated schedule"""
        running_tasks = set()
        remaining_rcpsp = None
        executed_rcpsp = None
        if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING:
            (
                executed_rcpsp,
                remaining_rcpsp,
                running_tasks,
            ) = self.compute_remaining_rcpsp(current_time, executed_schedule)
        if self._params_remaining_rcpsp == ParamsRemainingRCPSP.KEEP_FULL_RCPSP:
            running_tasks = set(
                task
                for task, sched in executed_schedule.rcpsp_schedule.items()
                if sched["start_time"] <= current_time
                and sched["end_time"] > current_time
            )
            executed_rcpsp = executed_schedule.problem
            remaining_rcpsp = executed_rcpsp

        rcpsp = self._simulated_rcpsp.create_uncertain_rcpsp_model(
            launched_tasks=set(executed_schedule.rcpsp_schedule.keys()) | running_tasks,
            rcpsp=remaining_rcpsp,
        )
        # print("initialisation rcpsp, ", perf_counter()-tt)
        best_tasks = None
        best_next_start = None
        best_makespan = None
        worst_expected_schedule = None
        if (
            self._mode == ExecutionMode.HINDSIGHT_LEX
            or self._mode == ExecutionMode.HINDSIGHT_DBP
        ):
            (
                best_tasks,
                best_next_start,
                best_makespan,
                worst_expected_schedule,
            ) = self.next_tasks_hindsight(
                rcpsp, running_tasks, executed_schedule, current_time
            )
        elif (
            self._mode == ExecutionMode.REACTIVE_AVERAGE
            or self._mode == ExecutionMode.REACTIVE_WORST
            or self._mode == ExecutionMode.REACTIVE_BEST
        ):
            (
                best_tasks,
                best_next_start,
                best_makespan,
                worst_expected_schedule,
            ) = self.next_tasks_reactive(
                rcpsp, running_tasks, executed_schedule, current_time
            )

        # Return the best next tasks and the worst schedule in term of makespan
        # among the scenario schedules that feature those best next tasks to start next
        shift = (
            current_time
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING
            else 0
        )
        return (
            best_tasks - running_tasks,
            best_next_start + current_time
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING
            else best_next_start,
            best_makespan,
            RCPSPSolution(
                problem=executed_rcpsp,
                rcpsp_schedule={
                    task: {
                        "start_time": worst_expected_schedule[task] + shift,
                        "end_time": worst_expected_schedule[task]
                        + shift
                        + int(  # in case the max() below returns a numpy type
                            max(
                                mode["duration"]
                                for mode in remaining_rcpsp.mode_details[task].values()
                            )
                        ),
                    }
                    for task in remaining_rcpsp.mode_details.keys()
                    if task != "source" and task not in running_tasks
                },
                rcpsp_permutation=[],
                standardised_permutation=[],
            ),
        )

    def compute_remaining_rcpsp(self, current_time, executed_schedule):
        executed_rcpsp = executed_schedule.problem
        if len(executed_schedule.rcpsp_schedule) == 0:
            running_tasks = set()
            front_tasks = set()
            for succ in executed_rcpsp.successors.values():
                front_tasks = front_tasks | set(succ)
            front_tasks = set(executed_rcpsp.mode_details.keys()) - front_tasks
        else:
            running_tasks = set(
                task
                for task, sched in executed_schedule.rcpsp_schedule.items()
                if sched["start_time"] <= current_time
                and sched["end_time"] > current_time
            )
            front_tasks = set()
            for task in executed_schedule.rcpsp_schedule:
                front_tasks = front_tasks | set(executed_rcpsp.successors[task])
            front_tasks = front_tasks - set(executed_schedule.rcpsp_schedule.keys())
            front_tasks = front_tasks | running_tasks
        initial_task = "source"
        initial_task_mode = {
            1: dict(duration=0, **{res: 0 for res in executed_rcpsp.resources.keys()})
        }
        tasks_list = [initial_task]
        non_executed_tasks = [
            t
            for t in executed_rcpsp.tasks_list
            if t not in executed_schedule.rcpsp_schedule
        ]
        # running_tasks = list(running_tasks)
        tasks_list = tasks_list + non_executed_tasks + list(running_tasks)
        # Create "remaining" RCPSP: keep all the non executed tasks plus the running ones
        mode_details = {
            task: deepcopy(executed_rcpsp.mode_details[task])
            for task in non_executed_tasks
        }
        successors = {
            task: executed_rcpsp.successors[task] for task in non_executed_tasks
        }
        name_task = {
            task: executed_rcpsp.name_task[task] for task in non_executed_tasks
        }
        running_tasks_modes = {
            task: deepcopy(executed_rcpsp.mode_details[task]) for task in running_tasks
        }
        for task in running_tasks_modes:
            for mode in running_tasks_modes[task]:
                running_tasks_modes[task][mode]["duration"] = (
                    executed_schedule.rcpsp_schedule[task]["end_time"] - current_time
                )
        mode_details.update(running_tasks_modes)
        mode_details.update({initial_task: initial_task_mode})
        successors.update({initial_task: list(front_tasks)})
        successors.update(
            {
                task: executed_rcpsp.successors[task]
                for task in executed_rcpsp.mode_details
                if task in running_tasks
            }
        )
        name_task.update({initial_task: "fake initial task"})
        name_task.update(
            {
                task: executed_rcpsp.name_task[task]
                for task in executed_rcpsp.mode_details
                if task in running_tasks
            }
        )
        remaining_rcpsp = RCPSPModel(
            resources=executed_rcpsp.resources,
            tasks_list=tasks_list,
            source_task=initial_task,
            sink_task=executed_rcpsp.sink_task,  # sink remains the same.
            non_renewable_resources=executed_rcpsp.non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=executed_rcpsp.horizon - current_time,
            horizon_multiplier=executed_rcpsp.horizon_multiplier,
            name_task=name_task,
        )
        return executed_rcpsp, remaining_rcpsp, running_tasks

    def next_tasks_hindsight(
        self,
        rcpsp,
        running_tasks,
        executed_schedule: Optional[RCPSPSolution],
        current_time: int,
    ):
        starts_hint = {}
        if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING:
            starts_hint = {"source": 0}
            starts_hint.update({task: 0 for task in running_tasks})
        if self._params_remaining_rcpsp == ParamsRemainingRCPSP.KEEP_FULL_RCPSP:
            starts_hint = {
                t: executed_schedule.get_start_time(t)
                for t in executed_schedule.rcpsp_schedule
            }

        # Sample RCPSPs and choose the best next task to start on average
        scenarios = defaultdict(lambda: [])
        for _ in range(self._samples):
            t = perf_counter()
            sampled_rcpsp = rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )
            status, makespan, feasible_solution = self.compute_schedule(
                sampled_rcpsp, starts_hint=starts_hint, current_time=current_time
            )
            if status == cp_model.INFEASIBLE:
                continue
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError("Invalid CPSAT model.")

            # Extract the next tasks to start
            tt = perf_counter()
            tasks_to_start = set()
            date_to_start = float("inf")
            for task, start in feasible_solution.items():
                if (
                    task != "source"
                    and task not in running_tasks
                    and (
                        self._params_remaining_rcpsp
                        == ParamsRemainingRCPSP.EXACT_REMAINING
                        or self._params_remaining_rcpsp
                        == ParamsRemainingRCPSP.KEEP_FULL_RCPSP
                        and task not in executed_schedule.rcpsp_schedule
                    )
                ):
                    if start < date_to_start:
                        tasks_to_start = set([task])
                        date_to_start = start
                    elif start == date_to_start:
                        tasks_to_start.add(task)

            scenarios[frozenset(tasks_to_start)].append(
                (feasible_solution, makespan, date_to_start)
            )

        if len(scenarios) == 0:
            raise RuntimeError("No feasible scenario")

        shift = (
            current_time
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING
            else 0
        )

        if self._mode == ExecutionMode.HINDSIGHT_LEX:
            tt = perf_counter()
            for ts, lmk in scenarios.items():
                highest_makespan = 0
                worst_solution = None
                date_to_start = None
                for mk in lmk:
                    solution = mk[0]
                    mkspan = mk[1]
                    if mkspan >= highest_makespan:
                        worst_solution = solution
                        date_to_start = mk[2]
                        highest_makespan = mk[1]
                scenarios[ts] = (
                    worst_solution,
                    np.mean([mk[1] for mk in lmk]),
                    0
                    if self._deadline is None
                    else float(
                        sum([1 if mk[1] + shift > self._deadline else 0 for mk in lmk])
                    )
                    / float(len(lmk)),
                    date_to_start,
                )

            # Select the best tasks to start in term of highest choice frequency
            # then average makespan is no deadline is set otherwise highest probability
            # of meeting the deadline
            tasks_frequency = defaultdict(lambda: 0)
            for tasks in scenarios:
                tasks_frequency[tasks] = tasks_frequency[tasks] + 1
            highest_frequency = 0
            best_tasks_list = []
            for tasks, frequency in tasks_frequency.items():
                if frequency == highest_frequency:
                    best_tasks_list.append(tasks)
                elif frequency > highest_frequency:
                    best_tasks_list = [tasks]
                    highest_frequency = frequency
            best_tasks = None
            best_next_start = None
            best_makespan = float("inf")
            best_deadline_prob = float("inf")
            worst_expected_schedule = None
            for tasks in best_tasks_list:
                solution_makespan_date = scenarios[tasks]
                if solution_makespan_date[1] < best_makespan:
                    best_makespan = solution_makespan_date[1]
                    if self._deadline is None:
                        worst_expected_schedule = solution_makespan_date[0]
                        best_next_start = solution_makespan_date[3]
                        best_tasks = tasks
                if self._deadline is not None:
                    if solution_makespan_date[2] < best_deadline_prob:
                        worst_expected_schedule = solution_makespan_date[0]
                        best_makespan = solution_makespan_date[1]
                        best_deadline_prob = solution_makespan_date[2]
                        best_next_start = solution_makespan_date[3]
                        best_tasks = tasks

        elif self._mode == ExecutionMode.HINDSIGHT_DBP:

            best_tasks = None
            best_next_start = None
            best_makespan = float("inf")
            best_deadline_prob = float("inf")
            worst_expected_schedule = None

            for ts, lmk in scenarios.items():
                avg_date_to_start_pre = int(np.mean([mk[2] for mk in lmk]))
                avg_makespan = 0
                valid_samples = 0
                highest_makespan = 0
                avg_deadline_exceed = 0
                worst_schedule = None

                for _ in range(self._samples):
                    sampled_rcpsp = rcpsp.create_rcpsp_model(
                        MethodRobustification(MethodBaseRobustification.SAMPLE)
                    )
                    scn_starts_hint = deepcopy(starts_hint)
                    additional_starts = self.simple_compute_earliest_starting_date(
                        sampled_rcpsp, tasks_to_start=ts, hard_starts=scn_starts_hint
                    )
                    if self._debug_logs:
                        print("add starts", additional_starts)
                    scn_starts_hint.update(additional_starts)
                    # scn_starts_hint.update({t: avg_date_to_start_pre for t in ts})
                    status, makespan, feasible_solution = self.compute_schedule(
                        sampled_rcpsp,
                        starts_hint=scn_starts_hint,
                        current_time=current_time,
                    )
                    if status == cp_model.INFEASIBLE:
                        avg_deadline_exceed += 1
                        continue
                    elif status == cp_model.MODEL_INVALID:
                        raise RuntimeError("Invalid CPSAT model.")

                    valid_samples += 1
                    avg_makespan += makespan
                    avg_deadline_exceed += int(
                        makespan + shift > self._deadline
                        if self._deadline is not None
                        else False
                    )

                    if makespan >= highest_makespan:
                        highest_makespan = makespan
                        worst_schedule = feasible_solution

                avg_date_to_start_post = int(
                    np.mean([feasible_solution[t] for t in ts])
                )

                if valid_samples > 0:
                    avg_makespan /= valid_samples
                    if avg_makespan < best_makespan:
                        best_makespan = avg_makespan
                        if self._deadline is None:
                            best_tasks = ts
                            best_next_start = avg_date_to_start_post
                            worst_expected_schedule = worst_schedule

                if self._deadline is not None:
                    avg_deadline_exceed /= self._samples
                    if avg_deadline_exceed < best_deadline_prob:
                        best_deadline_prob = avg_deadline_exceed
                        best_tasks = ts
                        best_next_start = avg_date_to_start_post
                        worst_expected_schedule = worst_schedule

        return best_tasks, best_next_start, best_makespan, worst_expected_schedule

    def compute_earliest_starting_date(
        self,
        sampled_rcpsp: RCPSPModel,
        tasks_to_start: Set[Hashable],
        hard_starts: Dict[Hashable, int],
    ):
        if sampled_rcpsp.n_jobs == 2:
            sol = RCPSPSolution(
                problem=sampled_rcpsp,
                rcpsp_schedule={
                    t: {
                        "start_time": 0,
                        "end_time": 0 + sampled_rcpsp.mode_details[t][1]["duration"],
                    }
                    for t in sampled_rcpsp.tasks_list
                },
                rcpsp_modes=[],
            )
            return {t: sol.get_start_time(t) for t in tasks_to_start}
        else:
            index_to_use = [
                sampled_rcpsp.index_task_non_dummy[t]
                for t in sampled_rcpsp.tasks_list_non_dummy
            ]
            perm = [
                sampled_rcpsp.index_task_non_dummy[tt]
                for tt in hard_starts
                if tt in sampled_rcpsp.index_task_non_dummy
            ] + [sampled_rcpsp.index_task_non_dummy[tt] for tt in tasks_to_start]
            perm += [
                i
                for i in index_to_use
                if i not in hard_starts and i not in tasks_to_start
            ]
            sol = RCPSPSolution(
                problem=sampled_rcpsp,
                rcpsp_permutation=perm,
                rcpsp_modes=[1 for i in range(sampled_rcpsp.n_jobs_non_dummy)],
            )
            assert sampled_rcpsp.n_jobs > 2
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING:
                sol.generate_schedule_from_permutation_serial_sgs_2(
                    current_t=0,
                    completed_tasks={},
                    scheduled_tasks_start_times=hard_starts,
                )
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.KEEP_FULL_RCPSP:
                sol.generate_schedule_from_permutation_serial_sgs_2(
                    current_t=self._current_time,
                    completed_tasks={},
                    scheduled_tasks_start_times=hard_starts,
                )
            if self._debug_logs:
                print("Task to start")
                print({t: sol.get_start_time(t) for t in tasks_to_start})
            return {t: sol.get_start_time(t) for t in tasks_to_start}

    def simple_compute_earliest_starting_date(
        self,
        sampled_rcpsp: RCPSPModel,
        tasks_to_start: Set[Hashable],
        hard_starts: Dict[Hashable, int],
    ):
        ressources = {
            r: sampled_rcpsp.get_resource_availability_array(r)
            for r in sampled_rcpsp.resources_list
        }
        for t in hard_starts:
            d = sampled_rcpsp.mode_details[t][1]["duration"]
            if d > 0:
                for r in sampled_rcpsp.resources_list:
                    re = sampled_rcpsp.mode_details[t][1][r]
                    if re > 0:
                        ressources[r][hard_starts[t] : hard_starts[t] + d] -= re
        l = list(tasks_to_start)
        cur_sched = hard_starts
        for i in range(len(l)):
            min_dates = [
                hard_starts[t] + sampled_rcpsp.mode_details[t][1]["duration"]
                for t in cur_sched
                if l[i] in sampled_rcpsp.successors[t]
            ]
            if len(min_dates) > 0:
                mindate = max(min_dates)
            else:
                mindate = 0
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.KEEP_FULL_RCPSP:
                mindate = max(mindate, self._current_time)
            dur = sampled_rcpsp.mode_details[l[i]][1]["duration"]
            if dur == 0:
                cur_sched[l[i]] = mindate
                continue
            good_date = next(
                t
                for t in range(mindate, sampled_rcpsp.horizon)
                if all(
                    np.min(ressources[r][t : t + dur])
                    >= sampled_rcpsp.mode_details[l[i]][1][r]
                    for r in ressources
                )
            )
            cur_sched[l[i]] = good_date
            for r in sampled_rcpsp.resources_list:
                re = sampled_rcpsp.mode_details[l[i]][1][r]
                if re > 0:
                    ressources[r][cur_sched[l[i]] : cur_sched[l[i]] + dur] -= re
        return {t: cur_sched[t] for t in tasks_to_start}

    def next_tasks_reactive(
        self,
        rcpsp,
        running_tasks,
        executed_schedule: Optional[RCPSPSolution],
        current_time: int,
    ):
        if self._mode == ExecutionMode.REACTIVE_AVERAGE:
            reactive_rcpsp = rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.AVERAGE)
            )
        elif self._mode == ExecutionMode.REACTIVE_WORST:
            reactive_rcpsp = rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.WORST_CASE)
            )
        elif self._mode == ExecutionMode.REACTIVE_BEST:
            reactive_rcpsp = rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.BEST_CASE)
            )
        starts_hint = {}
        if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING:
            starts_hint = {"source": 0}
            starts_hint.update({task: 0 for task in running_tasks})
        if self._params_remaining_rcpsp == ParamsRemainingRCPSP.KEEP_FULL_RCPSP:
            starts_hint = {
                t: executed_schedule.get_start_time(t)
                for t in executed_schedule.rcpsp_schedule
            }

        status, makespan, feasible_solution = self.compute_schedule(
            reactive_rcpsp, starts_hint=starts_hint, current_time=current_time
        )

        if status == cp_model.INFEASIBLE:
            raise RuntimeError("Infeasible remaining RCPSP.")
        elif status == cp_model.MODEL_INVALID:
            raise RuntimeError("Invalid CPSAT model.")

        # Extract the next tasks to start
        tasks_to_start = set()
        date_to_start = float("inf")
        for task, start in feasible_solution.items():
            if (
                task != "source"
                and task not in running_tasks
                and (
                    self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING
                    or self._params_remaining_rcpsp
                    == ParamsRemainingRCPSP.KEEP_FULL_RCPSP
                    and task not in executed_schedule.rcpsp_schedule
                )
            ):
                if start < date_to_start:
                    tasks_to_start = set([task])
                    date_to_start = start
                elif start == date_to_start:
                    tasks_to_start.add(task)
        return tasks_to_start, date_to_start, makespan, feasible_solution

    def compute_schedule(
        self,
        rcpsp: RCPSPModel,
        starts_hint: Dict[int, int] = None,
        current_time: int = 0,
    ) -> Tuple[int, int, List[int]]:
        data = Graph().create_from_data(rcpsp)
        data.to(self._device)
        out = self._model(data)
        data.out = out
        # TODO: is there a cleaner way to do this?
        # data._slice_dict['out'] = data._slice_dict['x']
        # data._inc_dict['out'] = data._inc_dict['x']
        tt = perf_counter()
        t2t, dur, r2t, rc = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
        )
        xorig = np.around(
            data.out[len(rc) :, 0].cpu().detach().numpy(), decimals=0
        ).astype(int)

        if self._scheduler == Scheduler.CPSAT:
            return self.compute_schedule_cpsat(
                rcpsp,
                t2t,
                dur,
                r2t,
                rc,
                xorig,
                starts_hint,
                current_time,
                params_cp=self._params_cp,
            )
        elif self._scheduler == Scheduler.SGS:
            return self.compute_schedule_sgs(
                rcpsp, t2t, dur, r2t, rc, xorig, starts_hint, current_time
            )

    def compute_schedule_cpsat(
        self,
        rcpsp,
        t2t,
        dur,
        r2t,
        rc,
        xorig,
        starts_hint,
        current_time,
        params_cp: CPSatSpecificParams,
    ):
        curt = perf_counter()

        for phase in ["P1", "P2"]:
            model = cp_model.CpModel()
            horizon = int(2.0 * rcpsp.horizon)
            starts = [
                model.NewIntVar(0, horizon - 1, "start_task[{}]".format(i))
                for i in range(len(dur))
            ]
            ends = [
                model.NewIntVar(0, horizon - 1, "start_task[{}]".format(i))
                for i in range(len(dur))
            ]
            durs = [model.NewConstant(int(dur[i])) for i in range(len(dur))]
            intervals = [
                model.NewIntervalVar(
                    starts[i], durs[i], ends[i], "interval_task[{}]".format(i)
                )
                for i in range(len(dur))
            ]
            makespan = model.NewIntVar(0, horizon - 1, "makespan")
            for t in range(t2t.shape[0]):
                for p in range(t2t.shape[1]):
                    if t2t[t][p] > 0:
                        model.Add(ends[p] <= starts[t])
                        # model.AddNoOverlap([intervals[t], intervals[p]])  # redundant

            for r in range(r2t.shape[0]):
                model.AddCumulative(
                    [intervals[t] for t in range(len(dur)) if r2t[r][t] > 0],
                    [int(r2t[r][t]) for t in range(len(dur)) if r2t[r][t] > 0],
                    int(rc[r]),
                )

            model.AddMaxEquality(makespan, ends)

            if starts_hint is not None:
                constrained_starts = {
                    i: model.NewConstant(starts_hint[t])
                    for i, t in enumerate(rcpsp.successors)
                    if t in starts_hint
                }
                for task_idx, start_constant in constrained_starts.items():
                    model.Add(starts[task_idx] == start_constant)

            # Search for a feasible solution
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = params_cp.num_workers
            solver.parameters.max_time_in_seconds = max(
                0, CPSAT_TIME_LIMIT - int(perf_counter() - curt)
            )
            solver.parameters.max_time_in_seconds = min(
                solver.parameters.max_time_in_seconds, params_cp.time_limit_seconds
            )
            if self._debug_logs:
                print(f"{solver.parameters.max_time_in_seconds} max time in seconds")
            xorig_list = xorig.tolist()
            shift = min(xorig_list)
            xorig_list = [x - shift for x in xorig_list]
            if params_cp.warm_start_with_gnn:
                for i, x in enumerate(xorig_list):
                    if starts_hint is None or i not in starts_hint:
                        model.AddHint(starts[i], x)
            # model._CpModel__model.solution_hint.vars.extend(list(range(len(dur))))
            # model._CpModel__model.solution_hint.values.extend(xorig.tolist())
            if phase == "P1" and self._deadline is not None:
                model.Add(makespan <= self._deadline - current_time)
            if params_cp.do_minimization:
                model.Minimize(makespan)
            status = solver.Solve(model)
            if self._debug_logs:
                print(status, "Infeasible - ", cp_model.INFEASIBLE)
                print(status, "Unknown - ", cp_model.UNKNOWN)
                print(status, "optimal - ", cp_model.OPTIMAL)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                solution = [solver.Value(s) for s in starts]
                # Compress the solution
                starts_np = np.array(solution, dtype=np.int64)
                ends_np = starts_np + np.array(dur, dtype=np.int64)
                current_time = 0
                ub = np.max(ends_np) * 10
                while current_time < np.max(ends_np):
                    masked_starts = (
                        ((starts_np <= current_time) * ub)
                        + (np.multiply(starts_np, starts_np > current_time))
                    ).astype(int)
                    next_start = np.amin(masked_starts)
                    masked_ends = (
                        (((ends_np <= current_time) + (starts_np > current_time)) * ub)
                        + (
                            np.multiply(
                                ends_np,
                                1
                                - (
                                    (ends_np <= current_time)
                                    + (starts_np > current_time)
                                ),
                            )
                        )
                    ).astype(int)
                    next_end = np.amin(masked_ends)
                    if next_end == ub:
                        starts_np[:] = np.array(
                            [
                                s - (next_start - current_time)
                                if s >= current_time
                                else s
                                for s in starts_np
                            ]
                        )
                    else:
                        current_time = min(next_start, next_end)
                solution = starts_np.tolist()
                return (
                    status,
                    # solver.Value(makespan),
                    np.max(starts_np + np.array(dur, dtype=np.int64)),
                    {t: solution[i] for i, t in enumerate(rcpsp.successors)},
                )
            elif phase == "P1" and self._deadline is not None:
                continue
            else:
                if self._debug_logs:
                    print("Horizon rcpsp : ", rcpsp.horizon)
                    print("horizon for cp ", 2 * rcpsp.horizon)
                    print(rcpsp.successors)
                    print(rcpsp.mode_details)
                    print(starts_hint)
                    print("infeasible for some reason...")
                    break

        return (cp_model.INFEASIBLE, 10000000, {})

    def compute_schedule_sgs(
        self, rcpsp, t2t, dur, r2t, rc, xorig, starts_hint, current_time
    ):
        do_model = rcpsp
        sorted_index = np.argsort(xorig)
        tasks = [do_model.tasks_list[j] for j in sorted_index]
        perm = [
            do_model.index_task_non_dummy[t]
            for t in tasks
            if t in do_model.index_task_non_dummy
        ]
        if do_model.n_jobs == 2:
            sol = RCPSPSolution(
                problem=do_model,
                rcpsp_schedule={
                    t: {
                        "start_time": 0,
                        "end_time": 0 + do_model.mode_details[t][1]["duration"],
                    }
                    for t in do_model.tasks_list
                },
                rcpsp_modes=[1 for i in range(len(perm))],
            )
        else:
            sol = RCPSPSolution(
                problem=do_model,
                rcpsp_permutation=perm,
                rcpsp_modes=[1 for i in range(len(perm))],
            )
            assert do_model.n_jobs > 2
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.EXACT_REMAINING:
                sol.generate_schedule_from_permutation_serial_sgs_2(
                    current_t=0,
                    completed_tasks={},
                    scheduled_tasks_start_times=starts_hint,
                )
            if self._params_remaining_rcpsp == ParamsRemainingRCPSP.KEEP_FULL_RCPSP:
                sol.generate_schedule_from_permutation_serial_sgs_2(
                    current_t=self._current_time,
                    completed_tasks={},
                    scheduled_tasks_start_times=starts_hint,
                )
                # print("cur time", self._current_time)
                # print("schedule from permutation : ", sol.problem.evaluate(sol))
                # print(sol.rcpsp_schedule)
            # print("Schedule sgs t=", max([sol.get_start_time(t) for t in rcpsp.tasks_list]))
            # print("sgs ", perf_counter()-t)
        makespan = do_model.evaluate(sol)["makespan"]
        feasible_solution = {
            t: sol.rcpsp_schedule[t]["start_time"] for t in sol.rcpsp_schedule
        }
        return (
            cp_model.FEASIBLE,
            makespan,
            {t: feasible_solution[t] for t in feasible_solution},
        )

    @staticmethod
    def plot_ressource_view_online(
        rcpsp_model: RCPSPModel,
        rcpsp_sol: RCPSPSolution,
        current_time: int,
        list_resource: List[Union[int, str]] = None,
        title_figure="",
        fig=None,
        ax=None,
        ax_xlim=None,
    ):
        modes_dict = rcpsp_model.build_mode_dict(rcpsp_sol.rcpsp_modes)
        if list_resource is None:
            list_resource = rcpsp_model.resources_list
        if ax is None:
            fig, ax = plt.subplots(
                nrows=len(list_resource), figsize=(10, 5), sharex=True
            )
            fig.suptitle(title_figure)
        polygons_ax = {i: [] for i in range(len(list_resource))}
        labels_ax = {i: [] for i in range(len(list_resource))}
        sorted_activities = sorted(
            rcpsp_sol.rcpsp_schedule,
            key=lambda x: rcpsp_sol.get_start_time(x),
        )
        for j in sorted_activities:
            time_start = rcpsp_sol.get_start_time(j)
            time_end = rcpsp_sol.get_end_time(j)
            for i in range(len(list_resource)):
                cons = rcpsp_model.mode_details[j][modes_dict[j]][list_resource[i]]
                if cons == 0:
                    continue
                bound = rcpsp_model.get_max_resource_capacity(list_resource[i])
                for k in range(0, bound):
                    # polygon = Polygon([(time_start, k), (time_end, k), (time_end, k+cons),
                    #                    (time_start, k+cons), (time_start, k)])
                    polygon = Rectangle(
                        xmin=time_start, ymin=k, xmax=time_end, ymax=k + cons
                    )
                    areas = [
                        SchedulingExecutor.area_intersection(polygon, p)
                        for p in polygons_ax[i]
                    ]
                    if len(areas) == 0 or max(areas) == 0:
                        polygons_ax[i].append(polygon)
                        labels_ax[i].append(j)
                        break
        for i in range(len(list_resource)):
            patches = []
            for polygon in polygons_ax[i]:
                x = [
                    polygon.xmin,
                    polygon.xmax,
                    polygon.xmax,
                    polygon.xmin,
                    polygon.xmin,
                ]
                y = [
                    polygon.ymin,
                    polygon.ymin,
                    polygon.ymax,
                    polygon.ymax,
                    polygon.ymin,
                ]
                ax[i].plot(
                    x,
                    y,
                    zorder=-1,
                    color="b"
                    if x[1] <= current_time
                    else "g"
                    if x[0] >= current_time
                    else "cyan",
                )
                patches.append(pp(xy=[(xx, yy) for xx, yy in zip(x, y)]))
            p = PatchCollection(
                patches, cmap=matplotlib.cm.get_cmap("Blues"), alpha=0.4
            )
            ax[i].add_collection(p)
        merged_times, merged_cons = compute_nice_resource_consumption(
            rcpsp_model, rcpsp_sol, list_resources=list_resource
        )
        for i in range(len(list_resource)):
            ax[i].vlines(
                current_time,
                0,
                max(merged_cons),
                colors="darkgrey",
                linestyles="dashed",
            )
        for i in range(len(list_resource)):
            ax[i].plot(
                merged_times[i],
                merged_cons[i],
                color="r",
                linewidth=2,
                label="Consumption " + str(list_resource[i]),
                zorder=1,
            )
            ax[i].axhline(
                y=rcpsp_model.resources[list_resource[i]],
                linestyle="--",
                label="Limit : " + str(list_resource[i]),
                zorder=0,
            )
            ax[i].legend(fontsize=5)
            lims = ax[i].get_xlim()
            if ax_xlim is None:
                ax[i].set_xlim([lims[0], 1.0 * lims[1]])
            else:
                ax[i].set_xlim([ax_xlim[0], ax_xlim[1]])
        return fig, ax

    @staticmethod
    def plot_task_gantt_online(
        rcpsp_model: RCPSPModel,
        rcpsp_sol: RCPSPSolution,
        current_time: int,
        fig=None,
        ax=None,
        ax_xlim=None,
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 5))
            ax.set_title("Gantt Task")
        tasks = rcpsp_model.tasks_list
        nb_task = rcpsp_model.n_jobs
        sorted_task_by_end = sorted(
            rcpsp_sol.rcpsp_schedule,
            key=lambda x: 100000 * rcpsp_sol.get_end_time(x)
            + rcpsp_model.index_task[x],
        )
        max_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[-1]]["end_time"]
        min_time = rcpsp_sol.rcpsp_schedule[sorted_task_by_end[0]]["start_time"]
        patches = []
        for j in range(nb_task):
            nb_colors = nb_task // 2
            colors = plt.cm.get_cmap("hsv", nb_colors)
            box = [
                (j - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
                (j - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
                (j + 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
                (j + 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
                (j - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
            ]
            # polygon = Polygon([(b[1], b[0]) for b in box])
            x = [xy[1] for xy in box]
            y = [xy[0] for xy in box]
            my_color = (
                "b"
                if box[1][1] <= current_time
                else "g"
                if box[0][1] >= current_time
                else "cyan"
            )
            ax.plot(x, y, zorder=-1, color=my_color)
            patches.append(pp(xy=[(xx[1], xx[0]) for xx in box], facecolor=my_color))
        ax.vlines(current_time, -0.5, nb_task, colors="darkgrey", linestyles="dashed")
        p = PatchCollection(patches, match_original=True, alpha=0.4)
        ax.add_collection(p)
        if ax_xlim is None:
            ax.set_xlim((min_time, max_time))
        else:
            ax.set_xlim((ax_xlim[0], ax_xlim[1]))
        ax.set_ylim((-0.5, nb_task))
        ax.set_yticks(range(nb_task))
        ax.set_yticklabels(
            tuple(["Task " + str(tasks[j]) for j in range(nb_task)]),
            fontdict={"size": 7},
        )
        return fig, ax

    @staticmethod
    def area_intersection(a: Rectangle, b: Rectangle):
        # Solution picked here, to avoid shapely.
        # https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kobe_rcpsp_dir = os.path.join(root_dir, "kobe-rcpsp/data/rcpsp")
    rcpsp = parse_file(os.path.join(kobe_rcpsp_dir, "j30.sm/j301_1.sm"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = ResTransformer
    # Net = ResGINE
    model = Net().to(device)
    real_path = None
    path1 = os.path.join(root_dir, "torch_folder/model_ResTransformer_256_50000.tch")
    path2 = os.path.join(root_dir, "torch_data/model_ResTransformer_256_50000.tch")
    if os.path.exists(path1):
        real_path = path1
    if os.path.exists(path2):
        real_path = path2
    model.load_state_dict(
        torch.load(
            real_path,
            map_location=device,
        )
    )
    executor = SchedulingExecutor(
        rcpsp=rcpsp,
        model=model,
        device=device,
        scheduler=Scheduler.SGS,
        mode=ExecutionMode.HINDSIGHT_DBP,
        samples=10,
        deadline=None,
        params_cp=CPSatSpecificParams.default_cp_reactive(),
    )

    executed_schedule, current_time = executor.reset()
    makespan = None
    stop = False

    fig_gantt = None
    ax_gantt = None
    fig_res = None
    ax_res = None
    plt.ion()

    while not stop:
        (
            next_tasks,
            next_start,
            expected_makespan,
            expected_schedule,
        ) = executor.next_tasks(executed_schedule, current_time)
        # print("Best tasks to start at time {}: {}".format(next_start, list(next_tasks)))

        current_time, executed_schedule, stop = executor.progress(
            next_tasks, next_start, expected_schedule
        )
        print(
            "Executed schedule: {}\nExpected schedule: {}\nCurrent time: {}".format(
                executed_schedule, expected_schedule, current_time
            )
        )
        makespan = expected_makespan * 1.1 if makespan is None else makespan
        merged_schedule = deepcopy(executed_schedule.rcpsp_schedule)
        merged_schedule.update(expected_schedule.rcpsp_schedule)
        merged_solution = RCPSPSolution(
            problem=executed_schedule.problem,
            rcpsp_schedule=merged_schedule,
            rcpsp_permutation=[],
            standardised_permutation=[],
        )
        fig_gantt, ax_gantt = SchedulingExecutor.plot_task_gantt_online(
            merged_solution.problem,
            merged_solution,
            current_time,
            fig=fig_gantt,
            ax=ax_gantt,
            ax_xlim=[0, makespan],
        )
        fig_res, ax_res = SchedulingExecutor.plot_ressource_view_online(
            merged_solution.problem,
            merged_solution,
            next_start,
            fig=fig_res,
            ax=ax_res,
            ax_xlim=[0, makespan],
        )
        plt.show()
        plt.pause(0.01)
        ax_gantt.cla()
        for ar in ax_res:
            ar.cla()