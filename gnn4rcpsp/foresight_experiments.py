from typing import Optional

import json
import sys
import os
import warnings
from collections import defaultdict
from datetime import datetime
from multiprocessing import Manager
from time import perf_counter

import torch

from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRCPSPModel,
    create_poisson_laws,
)
from executor import (
    CPSatSpecificParams,
    ExecutionMode,
    ParamsRemainingRCPSP,
    Scheduler,
    SchedulingExecutor,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from infer_schedules import build_rcpsp_model
from models import ResTransformer
from torch_geometric.data import DataLoader
from ortools.sat.python import cp_model


class RCPSPOrtoolsSolver(SolverDO):
    def __init__(self, problem: RCPSPModel, params_objective_function: Optional[ParamsObjectiveFunction]=None):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.model: cp_model.CpModel = None

    def init_model(self, **kwargs):
        model = cp_model.CpModel()
        horizon = int(self.problem.horizon)
        nb_tasks = self.problem.n_jobs
        duration = [self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list]
        starts = [
            model.NewIntVar(0, horizon - 1, "start_task[{}]".format(i))
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(0, horizon - 1, "start_task[{}]".format(i))
            for i in range(nb_tasks)
        ]
        durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        intervals = [
            model.NewIntervalVar(
                starts[i], durs[i], ends[i], "interval_task[{}]".format(i)
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(ends[self.problem.index_task[t]] <= starts[self.problem.index_task[s]])
        for r in self.problem.resources_list:
            model.AddCumulative(
                [intervals[t] for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                [int(self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0)) for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                self.problem.get_max_resource_capacity(r),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get('starts_hint', None)
        if starts_hint is not None:
            for t in starts_hint:
                if t in self.problem.index_task:
                    index = self.problem.index_task[t]
                    model.AddHint(starts[index], int(starts_hint[t]))
        model.Minimize(makespan)
        self.starts = starts
        self.model = model

    def solve(self, parameters_cp: ParametersCP, **kwargs) -> ResultStorage:
        # Search for a feasible solution
        if self.model is None:
            self.init_model(**kwargs)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = parameters_cp.time_limit
        status = solver.Solve(self.model)
        if status == cp_model.OPTIMAL:
            st = "OPTIMAL"
        elif status == cp_model.INFEASIBLE:
            st = "INFEASIBLE"
        elif status == cp_model.FEASIBLE:
            st = "FEASIBLE"
        elif status == cp_model.UNKNOWN:
            st = "UNKNOWN"
        solution = [solver.Value(s) for s in self.starts]
        schedule = {self.problem.tasks_list[t]: {"start_time": solution[t],
                                                 "end_time": solution[t]
                                                 + self.problem.mode_details[self.problem.tasks_list[t]][1]["duration"]}
                    for t in range(len(solution))}
        sol = RCPSPSolution(problem=self.problem, rcpsp_modes=[1]*self.problem.n_jobs_non_dummy, rcpsp_schedule=schedule)
        fit = self.aggreg_sol(sol)
        res = ResultStorage(list_solution_fits=[(sol, fit)],
                            mode_optim=self.params_objective_function.sense_function)
        res.st = st
        return res


def run_foresight_experiments():
    results = json.load(open("../bigstatisitics_hindsight_vs_reactive_stats_30_10_20221118131949.json", "r"))
    data_list = torch.load("../torch_data/data_list.tch")
    test_loader = DataLoader(
        data_list,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    device = "cpu"
    Net = ResTransformer
    # Net = ResGINE
    model = Net().to(device)
    model.load_state_dict(
        torch.load(
            "../torch_data/model_ResTransformer_256_50000.tch",
            map_location=torch.device(device),
        )
    )
    res_json = json.load(open("foresight_res.json", "r"))
    for benchmark in results:
        id_benchmark = int(benchmark.split(" ")[1])
        data_bc = data_list[id_benchmark]
        t2t, dur, r2t, rc = (
            data_bc.t2t.view(len(data_bc.dur), -1).data.cpu().detach().numpy(),
            data_bc.dur.data.cpu().detach().numpy(),
            data_bc.r2t.view(len(data_bc.rc), len(data_bc.dur)).data.cpu().detach().numpy(),
            data_bc.rc.data.cpu().detach().numpy(),
        )
        rcpsp_model = build_rcpsp_model(t2t, dur, r2t, rc)[0]
        if benchmark not in res_json:
            res_json[benchmark] = {}
        for scenario in results[benchmark]:
            if scenario in res_json[benchmark]:
                print("done")
                continue
            one_schedule = results[benchmark][scenario]["SGS-HINDSIGHT_DBP"]["schedule"]
            duration_tasks = {int(x): one_schedule[x]["end_time"]-one_schedule[x]["start_time"] for x in one_schedule}
            for j in duration_tasks:
                rcpsp_model.mode_details[j][1]["duration"] = duration_tasks[j]
            rcpsp_model.update_functions()
            solver = RCPSPOrtoolsSolver(problem=rcpsp_model)
            starts_int = {int(x): one_schedule[x]["start_time"] for x in one_schedule}
            baseline = max(starts_int.values())
            all_baselines = [results[benchmark][scenario][a]["executed"] for a in results[benchmark][scenario]
                             if results[benchmark][scenario][a]["executed"] != "Fail"]
            print("makespan baseline : ", baseline)
            solver.init_model(starts_hint=starts_int)
            t = perf_counter()
            params_cp = ParametersCP(time_limit=100,
                                     all_solutions=False,
                                     nr_solutions=100000,
                                     nb_process=4,
                                     multiprocess=True,
                                     intermediate_solution=False)
            res = solver.solve(params_cp)
            t_end = perf_counter()-t
            sol, fit = res.get_best_solution_fit()
            print(fit)
            makespan = -fit
            relative_improve = [(x-makespan)/makespan*100 for x in all_baselines]
            print("makespan ", makespan)
            print("Relartive improve ", relative_improve)
            res_json[benchmark][scenario] = {"FORESIGHT": {"executed": makespan,
                                                           "schedule": sol.rcpsp_schedule, "timing": t_end-t,
                                                           "status": res.st}}
            json.dump(res_json, open("foresight_res.json", "w"), indent=4)
            json.dump(res_json, open("foresight_res_2.json", "w"), indent=4)


if __name__ == "__main__":
    run_foresight_experiments()