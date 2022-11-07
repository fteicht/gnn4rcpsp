import os
from collections import defaultdict, namedtuple
from copy import deepcopy
from enum import Enum
from time import perf_counter
from typing import Dict, List, Set, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws,
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

Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")


class Scheduler(Enum):
    CPSAT = 1
    SGS = 2


class ExecutionMode(Enum):
    REACTIVE = 1
    HINDSIGHT_LEX = 2  # Select first most appearing top tasks, then break tie by lowest average makespan
    HINDSIGHT_DBP = 3  # Extract first potential best tasks, then reevaluate each of them on each scenario and select top tasks by lowest average makespan


class SchedulingExecutor:
    def __init__(
        self,
        rcpsp: RCPSPModel,
        model: torch.nn.Module,
        device: torch.device,
        scheduler: Scheduler,
        mode: ExecutionMode,
        samples: int = 100,
    ) -> None:
        self._rcpsp = rcpsp
        self._model = model
        self._device = device
        self._scheduler = scheduler
        self._mode = mode
        self._poisson_laws = create_poisson_laws(
            base_rcpsp_model=self._rcpsp,
            range_around_mean_resource=1,
            range_around_mean_duration=3,
            do_uncertain_resource=False,
            do_uncertain_duration=True,
        )
        self._uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=self._rcpsp,
            poisson_laws={
                task: laws
                for task, laws in self._poisson_laws.items()
                if task in self._rcpsp.mode_details
            },
            uniform_law=True,
        )
        self._simulated_rcpsp = None
        self._executed_schedule = None
        self._current_time = None
        self._samples = samples

    def reset(self) -> Tuple[int, RCPSPSolution]:
        """Samples a new RCPSP (uncertain task durations and resource capacities)
        and returns an empty schedule and current time set to 0"""
        self._simulated_rcpsp = self._uncertain_rcpsp.create_rcpsp_model(
            MethodRobustification(MethodBaseRobustification.SAMPLE)
        )
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
        """Takes the next task to execute, their starting date and the expected schedule,
        and returns the new current time and updated executed schedule
        and a boolean indicating whether all the tasks have been scheduled or not"""

        # Add next_task to the executed schedule if it is scheduled to start now
        if next_start == self._current_time:
            self._executed_schedule.rcpsp_schedule.update(
                {
                    next_task: {
                        "start_time": next_start,
                        "end_time": next_start
                        + int(  # in case the max() below returns a numpy type
                            max(
                                mode["duration"]
                                for mode in self._simulated_rcpsp.mode_details[
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
                ] = self._simulated_rcpsp.mode_details[task]

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
        """Takes a schedule which has been executed (i.e. list of committed start dates for some tasks),
        the current time, and returns the next tasks to start, their starting date, the expected makespan and the full updated schedule"""

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
        rcpsp = RCPSPModel(
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
            ) = self.next_tasks_hindsight(rcpsp, running_tasks)
        elif self._mode == ExecutionMode.REACTIVE:
            (
                best_tasks,
                best_next_start,
                best_makespan,
                worst_expected_schedule,
            ) = self.next_tasks_reactive(rcpsp, running_tasks)

        # Return the best next tasks and the worst schedule in term of makespan
        # among the scenario schedules that feature those best next tasks to start next
        return (
            best_tasks - running_tasks,
            best_next_start + current_time,
            best_makespan,
            RCPSPSolution(
                problem=executed_rcpsp,
                rcpsp_schedule={
                    task: {
                        "start_time": worst_expected_schedule[task] + current_time,
                        "end_time": worst_expected_schedule[task]
                        + current_time
                        + int(  # in case the max() below returns a numpy type
                            max(
                                mode["duration"]
                                for mode in rcpsp.mode_details[task].values()
                            )
                        ),
                    }
                    for task in rcpsp.mode_details.keys()
                    if task != "source" and task not in running_tasks
                },
                rcpsp_permutation=[],
                standardised_permutation=[],
            ),
        )

    def next_tasks_hindsight(self, rcpsp, running_tasks):
        # Make the "remaining" RCPSP uncertain
        uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=rcpsp,
            poisson_laws={
                task: laws
                for task, laws in self._poisson_laws.items()
                if task in rcpsp.mode_details
            },
            uniform_law=True,
        )

        starts_hint = {"source": 0}
        starts_hint.update({task: 0 for task in running_tasks})

        # Sample RCPSPs and choose the best next task to start on average
        scenarios = defaultdict(lambda: [])
        for _ in range(self._samples):
            sampled_rcpsp = uncertain_rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )

            status, makespan, feasible_solution = self.compute_schedule(
                sampled_rcpsp, starts_hint=starts_hint
            )

            if status == cp_model.INFEASIBLE:
                continue
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError("Invalid CPSAT model.")

            # Extract the next tasks to start
            tasks_to_start = set()
            date_to_start = float("inf")
            for task, start in feasible_solution.items():
                if task != "source" and task not in running_tasks:
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

        if self._mode == ExecutionMode.HINDSIGHT_LEX:

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
                    date_to_start,
                )

            # Select the best tasks to start in term of highest choice frequency then average makespan
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
            worst_expected_schedule = None
            for tasks in best_tasks_list:
                solution_makespan_date = scenarios[tasks]
                if solution_makespan_date[1] < best_makespan:
                    worst_expected_schedule = solution_makespan_date[0]
                    best_makespan = solution_makespan_date[1]
                    best_next_start = solution_makespan_date[2]
                    best_tasks = tasks

        elif self._mode == ExecutionMode.HINDSIGHT_DBP:

            best_tasks = None
            best_next_start = None
            best_makespan = float("inf")
            worst_expected_schedule = None

            for ts, lmk in scenarios.items():
                avg_date_to_start_pre = int(np.mean([mk[2] for mk in lmk]))
                avg_makespan = 0
                valid_samples = 0
                highest_makespan = 0
                worst_schedule = None

                for _ in range(self._samples):
                    sampled_rcpsp = uncertain_rcpsp.create_rcpsp_model(
                        MethodRobustification(MethodBaseRobustification.SAMPLE)
                    )

                    scn_starts_hint = deepcopy(starts_hint)
                    scn_starts_hint.update({t: avg_date_to_start_pre for t in ts})

                    status, makespan, feasible_solution = self.compute_schedule(
                        sampled_rcpsp, starts_hint=scn_starts_hint
                    )

                    if status == cp_model.INFEASIBLE:
                        continue
                    elif status == cp_model.MODEL_INVALID:
                        raise RuntimeError("Invalid CPSAT model.")

                    valid_samples += 1
                    avg_makespan += makespan

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
                        best_tasks = ts
                        best_next_start = avg_date_to_start_post
                        worst_expected_schedule = worst_schedule

        return best_tasks, best_next_start, best_makespan, worst_expected_schedule

    def next_tasks_reactive(self, rcpsp, running_tasks):
        starts_hint = {"source": 0}
        starts_hint.update({task: 0 for task in running_tasks})
        status, makespan, feasible_solution = self.compute_schedule(
            rcpsp, starts_hint=starts_hint
        )
        if status == cp_model.INFEASIBLE:
            raise RuntimeError("Infeasible remaining RCPSP.")
        elif status == cp_model.MODEL_INVALID:
            raise RuntimeError("Invalid CPSAT model.")
        # Extract the next tasks to start
        tasks_to_start = set()
        date_to_start = float("inf")
        for task, start in feasible_solution.items():
            if task != "source" and task not in running_tasks:
                if start < date_to_start:
                    tasks_to_start = set([task])
                    date_to_start = start
                elif start == date_to_start:
                    tasks_to_start.add(task)
        print(
            f"Starting tasks {tasks_to_start} at time {date_to_start} with makespan {makespan}; feasible solution: {feasible_solution}"
        )
        return tasks_to_start, date_to_start, makespan, feasible_solution

    def compute_schedule(
        self, rcpsp: RCPSPModel, starts_hint: Dict[int, int] = None
    ) -> Tuple[int, int, List[int]]:
        data = Graph().create_from_data(rcpsp)
        data.to(self._device)
        out = self._model(data)
        data.out = out
        # TODO: is there a cleaner way to do this?
        # data._slice_dict['out'] = data._slice_dict['x']
        # data._inc_dict['out'] = data._inc_dict['x']

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
                rcpsp, t2t, dur, r2t, rc, xorig, starts_hint
            )
        elif self._scheduler == Scheduler.SGS:
            return self.compute_schedule_sgs(
                rcpsp, t2t, dur, r2t, rc, xorig, starts_hint
            )

    def compute_schedule_cpsat(self, rcpsp, t2t, dur, r2t, rc, xorig, starts_hint):
        horizon_start = int(max(xorig + dur))
        current_horizon = horizon_start
        curt = perf_counter()

        while current_horizon <= 2 * horizon_start:

            model = cp_model.CpModel()
            horizon = current_horizon
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
            solver.parameters.max_time_in_seconds = max(
                0, CPSAT_TIME_LIMIT - int(perf_counter() - curt)
            )
            xorig_list = xorig.tolist()
            shift = min(xorig_list)
            xorig_list = [x - shift for x in xorig_list]
            for i, x in enumerate(xorig_list):
                if starts_hint is None or i not in starts_hint:
                    model.AddHint(starts[i], x)
            # model._CpModel__model.solution_hint.vars.extend(list(range(len(dur))))
            # model._CpModel__model.solution_hint.values.extend(xorig.tolist())
            status = solver.Solve(model)

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
            else:
                current_horizon = int(1.05 * current_horizon)

        return (status, float("inf"), {})

    def compute_schedule_sgs(self, rcpsp, t2t, dur, r2t, rc, xorig, starts_hint):
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
            sol.generate_schedule_from_permutation_serial_sgs_2(
                current_t=0, completed_tasks={}, scheduled_tasks_start_times=starts_hint
            )

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
        print("Best tasks to start at time {}: {}".format(next_start, list(next_tasks)))

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
