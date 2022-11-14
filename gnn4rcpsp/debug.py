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
from infer_schedules import build_rcpsp_model, make_feasible_sgs, make_feasible_simple_sgs
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


def exploring(bench_id, data, result_file):
    data_batch = data

    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    data_list = data_batch.to_data_list()
    cnt = 0
    for data in data_list:
        cnt += 1
        # print(f"Instance {cnt}")
        data.to(device)
        out = model(data)
        data.out = out
        t2t, dur, r2t, rc = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
        )
        xorig_1 = np.around(
            out[len(rc):, 0].cpu().detach().numpy(), decimals=0
        ).astype(int)
        print("original model inference : ", xorig_1)

        model_rcpsp = build_rcpsp_model(t2t, dur, r2t, rc)[0]
        data = Graph().create_from_data_bis(model_rcpsp)
        data.to(device)
        out = model(data)
        xorig = np.around(
            out[len(rc):, 0].cpu().detach().numpy(), decimals=0
        ).astype(int)
        print("check bijection inference ", xorig)
        model_rcpsp = model_rcpsp.copy()
        print(model_rcpsp.tasks_list)
        model_rcpsp.tasks_list = [-1] + model_rcpsp.tasks_list[:30] + [150] + model_rcpsp.tasks_list[30:]
        model_rcpsp.mode_details.pop(0)
        model_rcpsp.tasks_list.remove(0)
        model_rcpsp.mode_details[150] = {1: {k: 1 if k =="duration" else 0
                                             for k in ["duration"]+model_rcpsp.resources_list}}
        model_rcpsp.mode_details[-1] = {1: {k: 0 for k in ["duration"]+model_rcpsp.resources_list}}
        model_rcpsp.successors[150] = [model_rcpsp.sink_task, model_rcpsp.tasks_list[32]]  # [model_rcpsp.source_task] # you can try uncommenting
        model_rcpsp.successors[model_rcpsp.tasks_list[29]] += [150]
        model_rcpsp.successors[-1] = model_rcpsp.successors[0]
        model_rcpsp.successors.pop(0)

        model_rcpsp = RCPSPModel(resources=model_rcpsp.resources,
                                 tasks_list=model_rcpsp.tasks_list,
                                 non_renewable_resources=model_rcpsp.non_renewable_resources,
                                 mode_details=model_rcpsp.mode_details,
                                 successors=model_rcpsp.successors,
                                 horizon=model_rcpsp.horizon,
                                 source_task=-1,
                                 sink_task=model_rcpsp.sink_task)
        print("Nb jobs = ", model_rcpsp.n_jobs)
        # model_rcpsp.source_task = "source"
        print("sec")
        dummy = model_rcpsp.get_dummy_solution()
        data = Graph().create_from_data_bis(model_rcpsp)
                                            #solution=dummy,
                                            #solution_makespan=model_rcpsp.evaluate(dummy)["makespan"])
        data.to(device)
        out = model(data)
        xorig = np.around(
            out[len(rc):, 0].cpu().detach().numpy(), decimals=0
        ).astype(int)
        print("inference with additional task : ", xorig)
        sorted_index = np.argsort(xorig)
        print(sorted_index)
        tasks = [model_rcpsp.tasks_list[j] for j in sorted_index]
        perm = [model_rcpsp.index_task_non_dummy[t]
                for t in tasks if t in model_rcpsp.index_task_non_dummy]
        sol = RCPSPSolution(problem=model_rcpsp, rcpsp_permutation=perm)
        print(model_rcpsp.evaluate(sol))


def test(test_loader, test_list, model, device, result_file):
    model.eval()
    with open(result_file, "w") as jsonfile:
        jsonfile.write("{\n}\n")

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    merged_res = {}
    for batch_idx, data in enumerate(
        tqdm(test_loader, desc="Benchmark Loop", leave=True)
    ):
        data.to(device)
        out = model(data)
        data.out = out

        # TODO: is there a cleaner way to do this?
        data._slice_dict["out"] = data._slice_dict["x"]
        data._inc_dict["out"] = data._inc_dict["x"]
        ms = exploring(
            test_list[batch_idx],  # batch_size==1 thus batch_idx==test_instance_id
            data,
            result_file,
        )
        print(batch_idx)
        merged_res[f"Benchmark {batch_idx}"] = ms
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        json.dump(merged_res,
                  open("res_with_cp/results_"+run_id+".json", "w"),
                  indent=4)


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
    print("evaluation original", origin_rcpsp.evaluate(sol))
    res = []
    if False:
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
        ax.scatter([x[0] for x in res], [x[1] for x in res])

        fig, ax = plt.subplots(1)
        ax.scatter([x[0] for x in res], [x[2] for x in res])

        fig, ax = plt.subplots(1)
        ax.scatter([x[0] for x in res], [x[3] for x in res])
        fig.savefig("gnnsgs_vs_cp_function_of_delta.png")
        plt.show()

    res = []

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
            params_cp.time_limit = 2.0
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
    if False:
        data_list = torch.load("../torch_data/data_list.tch")
        train_list = torch.load("../torch_data/train_list.tch")
        print(train_list)
        # test_list = list(set(range(len(data_list))) - set(train_list))
        indexes_of_interest = [1460, 509, 459, 450, 514, 234]
        indexes_of_interest = [1460]
        test_list = indexes_of_interest
        test_loader = DataLoader(
            [data_list[d] for d in indexes_of_interest], batch_size=1, shuffle=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Net = ResGINE
        Net = ResTransformer
        model = Net().to(device)
        model.load_state_dict(
            torch.load(
                "../torch_data/model_ResTransformer_256_50000.tch",
                map_location=torch.device(device),
            )
        )
        run_id = timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        print(f"Run ID: {run_id}")
        result_file = f"../hindsight_vs_reactive_{run_id}.json"
        result = test(
            test_loader=test_loader,
            test_list=test_list,
            model=model,
            device=device,
            result_file=result_file,
        )
