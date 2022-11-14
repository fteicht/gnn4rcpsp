import random

import json
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from time import perf_counter

import torch

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws,
)
from executor import CPSatSpecificParams, ExecutionMode, Scheduler, SchedulingExecutor
from gnn4rcpsp.gnn import ResGINE
from gnn4rcpsp.graph import Graph
from infer_schedules import build_rcpsp_model, make_feasible_sgs
from models import ResTransformer
from torch_geometric.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

NUM_SAMPLES = 15
num_scenario_per_instance = 2

ExecutionModeNames = {
    ExecutionMode.REACTIVE_AVERAGE: "REACTIVE_AVG",
    ExecutionMode.REACTIVE_WORST: "REACTIVE_WORST",
    ExecutionMode.REACTIVE_BEST: "REACTIVE_BEST",
    ExecutionMode.HINDSIGHT_LEX: "HINDSIGHT_LEX",
    ExecutionMode.HINDSIGHT_DBP: "HINDSIGHT_DBP",
}
SchedulerModeNames = {Scheduler.SGS: "SGS", Scheduler.CPSAT: "CPSAT"}


def modif_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = ResTransformer
    model = Net().to(device)
    model.load_state_dict(
        torch.load(
            "../torch_data/model_ResTransformer_256_50000.tch",
            map_location=torch.device(device),
        )
    )
    from discrete_optimization.rcpsp.rcpsp_parser import parse_file
    from discrete_optimization.rcpsp.rcpsp_solvers import CP_RCPSP_MZN
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kobe_rcpsp_dir = os.path.join(root_dir, "kobe-rcpsp/data/rcpsp")
    origin_rcpsp = parse_file(os.path.join(kobe_rcpsp_dir, "j90.sm/j901_1.sm"))
    data = Graph().create_from_data_bis(origin_rcpsp)
    # solution=dummy,
    # solution_makespan=model_rcpsp.evaluate(dummy)["makespan"])
    data.to(device)
    out = model(data)
    xorig = np.around(
        out[len(origin_rcpsp.resources_list):, 0].cpu().detach().numpy(), decimals=0
    ).astype(int)
    print("Original inference : ", xorig)
    sorted_index = np.argsort(xorig)
    tasks = [origin_rcpsp.tasks_list[j] for j in sorted_index]
    perm = [
        origin_rcpsp.index_task_non_dummy[t]
        for t in tasks
        if t in origin_rcpsp.index_task_non_dummy
    ]
    sol = RCPSPSolution(problem=origin_rcpsp, rcpsp_permutation=perm)
    print("evaluation model original ", origin_rcpsp.evaluate(sol))
    res = []

    # Do some experimentation with more and more noised scenarios.
    for delta in range(1, 10):
        uniform_laws = create_poisson_laws(
            base_rcpsp_model=origin_rcpsp,
            range_around_mean_resource=1,
            range_around_mean_duration=delta,
            do_uncertain_resource=False,
            do_uncertain_duration=True,
        )
        uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=origin_rcpsp,
            poisson_laws={
                task: laws
                for task, laws in uniform_laws.items()
                if task in origin_rcpsp.mode_details
            },
            uniform_law=True,
        )
        for j in range(20):
            new_model = uncertain_rcpsp.create_rcpsp_model(method_robustification=
                                                           MethodRobustification(MethodBaseRobustification.SAMPLE))
            solver = CP_RCPSP_MZN(rcpsp_model=new_model,
                                  cp_solver_name=CPSolverName.CHUFFED)
            solver.init_model(output_type=True)
            params_cp = ParametersCP.default()
            params_cp.time_limit = 2.0
            resstore = solver.solve(parameters_cp=params_cp)
            makespan = new_model.evaluate(resstore.get_best_solution_fit()[0])["makespan"]
            data = Graph().create_from_data_bis(new_model)
            # solution=dummy,
            # solution_makespan=model_rcpsp.evaluate(dummy)["makespan"])
            data.to(device)
            out = model(data)
            xorig = np.around(
                out[len(origin_rcpsp.resources_list):, 0].cpu().detach().numpy(), decimals=0
            ).astype(int)
            print("New inference : ", xorig)
            sorted_index = np.argsort(xorig)
            tasks = [new_model.tasks_list[j] for j in sorted_index]
            perm = [
                new_model.index_task_non_dummy[t]
                for t in tasks
                if t in new_model.index_task_non_dummy
            ]
            sol = RCPSPSolution(problem=new_model, rcpsp_permutation=perm)
            print("evaluation ", new_model.evaluate(sol))
            mks = max(sol.rcpsp_schedule[t]["start_time"] for t in sol.rcpsp_schedule)
            delta_ms = max(xorig) - mks
            print("delta ", delta)
            res += [(delta, delta_ms, mks, makespan)]
            print("SGS ", mks, " CP ", makespan)
            print("% ", (makespan-mks)/makespan*100, " % ")

    fig, ax = plt.subplots(1)
    ax.scatter([x[0] for x in res],
               [x[1] for x in res])

    fig, ax = plt.subplots(1)
    ax.scatter([x[0] for x in res],
               [x[2] for x in res])

    fig, ax = plt.subplots(1)
    ax.scatter([x[0] for x in res],
               [x[3] for x in res])
    fig.savefig("gnnsgs_vs_cp_function_of_delta.png")
    plt.show()

    res = []

    # Experiments : pop a given number of task in the problem
    for number_remove in range(1, 10):
        for k in range(20):
            some_random_picks = random.sample([t for t in origin_rcpsp.tasks_list
                                               if t not in {origin_rcpsp.source_task,
                                                            origin_rcpsp.sink_task}],
                                              number_remove)
            new_model = origin_rcpsp.copy()
            for some_random_pick in some_random_picks:
                new_model.mode_details.pop(some_random_pick)
                new_model.tasks_list.remove(some_random_pick)
                new_model.successors.pop(some_random_pick)
                for p in new_model.successors:
                    if some_random_pick in new_model.successors[p]:
                        new_model.successors[p].remove(some_random_pick)
            new_model = new_model.copy()
            solver = CP_RCPSP_MZN(rcpsp_model=new_model,
                                  cp_solver_name=CPSolverName.CHUFFED)
            solver.init_model(output_type=True)
            params_cp = ParametersCP.default()
            params_cp.time_limit = 1.0
            resstore = solver.solve(parameters_cp=params_cp)
            makespan_cp = new_model.evaluate(resstore.get_best_solution_fit()[0])["makespan"]
            data = Graph().create_from_data_bis(new_model)
            # solution=dummy,
            # solution_makespan=model_rcpsp.evaluate(dummy)["makespan"])
            data.to(device)
            out = model(data)
            xorig = np.around(
                out[len(origin_rcpsp.resources_list):, 0].cpu().detach().numpy(), decimals=0
            ).astype(int)
            print("New inference : ", xorig)
            sorted_index = np.argsort(xorig)
            tasks = [new_model.tasks_list[j] for j in sorted_index]
            perm = [
                new_model.index_task_non_dummy[t]
                for t in tasks
                if t in new_model.index_task_non_dummy
            ]
            sol = RCPSPSolution(problem=new_model, rcpsp_permutation=perm)
            print("evaluation ", new_model.evaluate(sol))
            mks = max(sol.rcpsp_schedule[t]["start_time"] for t in sol.rcpsp_schedule)
            delta = max(xorig)-mks
            print("delta ", delta)
            res += [(number_remove, delta, mks, makespan_cp)]
            print("% ", (makespan_cp-mks)/makespan_cp*100, "% diff")
    fig, ax = plt.subplots(1)
    ax.scatter([x[0] for x in res], [x[1] for x in res])

    fig, ax = plt.subplots(1)
    ax.scatter([x[0] for x in res], [x[2] for x in res])

    fig, ax = plt.subplots(1)
    ax.scatter([x[0] for x in res], [(x[3]-x[2])/x[3]*100 for x in res])
    fig.savefig("Overcost of cp, modif instances.png")
    plt.show()


if __name__ == "__main__":
    modif_model()
