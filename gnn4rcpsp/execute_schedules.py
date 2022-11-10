import json
import os
from collections import defaultdict
from datetime import datetime
from time import perf_counter

import torch
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws,
)
from executor import CPSatSpecificParams, ExecutionMode, Scheduler, SchedulingExecutor
from infer_schedules import build_rcpsp_model
from models import ResTransformer
from torch_geometric.data import DataLoader
from tqdm import tqdm

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


def execute_schedule(bench_id, data, result_file):
    data_batch = data

    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    data_list = data_batch.to_data_list()
    cnt = 0
    for data in data_list:
        cnt += 1
        # print(f"Instance {cnt}")

        t2t, dur, r2t, rc = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
        )
        rcpsp_model = build_rcpsp_model(t2t, dur, r2t, rc)[0]

        poisson_laws = create_poisson_laws(
            base_rcpsp_model=rcpsp_model,
            range_around_mean_resource=1,
            range_around_mean_duration=3,
            do_uncertain_resource=False,
            do_uncertain_duration=True,
        )
        uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=rcpsp_model,
            poisson_laws={
                task: laws
                for task, laws in poisson_laws.items()
                if task in rcpsp_model.mode_details
            },
            uniform_law=True,
        )

        makespans = defaultdict(lambda: {})

        for scn in tqdm(range(num_scenario_per_instance), desc="Scenario Loop", leave=False):
            sample_rcpsp = uncertain_rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )

            for scheduler, execution_mode in tqdm(
                [
                    (Scheduler.CPSAT, ExecutionMode.REACTIVE_AVERAGE),
                    (Scheduler.CPSAT, ExecutionMode.HINDSIGHT_LEX),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_AVERAGE),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_WORST),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_BEST),
                    (Scheduler.SGS, ExecutionMode.HINDSIGHT_LEX),
                    (Scheduler.SGS, ExecutionMode.HINDSIGHT_DBP),
                ],
                desc="Mode Loop",
                leave=False,
            ):
                executor = SchedulingExecutor(
                    rcpsp_model,
                    model,
                    device,
                    scheduler,
                    execution_mode,
                    NUM_SAMPLES,
                    params_cp=CPSatSpecificParams.default_cp_reactive(),
                )
                stop = False
                key_in_result = (
                    ExecutionModeNames[execution_mode]
                    + "-"
                    + SchedulerModeNames[scheduler]
                )
                executed_schedule, current_time = executor.reset(sim_rcpsp=sample_rcpsp)
                makespans[f"Scenario {scn}"][key_in_result] = {"expectations": []}
                timer = perf_counter()
                try:
                    while not stop:
                        (
                            next_tasks,
                            next_start,
                            expected_makespan,
                            expected_schedule,
                        ) = executor.next_tasks(executed_schedule, current_time)
                        current_time, executed_schedule, stop = executor.progress(
                            next_tasks, next_start, expected_schedule
                        )
                        print(current_time, expected_makespan)

                        makespans[f"Scenario {scn}"][key_in_result][
                            "expectations"
                        ].append(float(expected_makespan))

                    makespans[f"Scenario {scn}"][key_in_result][
                        "executed"
                    ] = int(current_time)
                    makespans[f"Scenario {scn}"][key_in_result]["timing"] = (
                        perf_counter() - timer
                    )
                    makespans[f"Scenario {scn}"][key_in_result][
                        "schedule"
                    ] = {t: {k: int(executed_schedule.rcpsp_schedule[t][k])
                             for k in executed_schedule.rcpsp_schedule[t]}
                         for t in executed_schedule.rcpsp_schedule}
                    print("Algo, ", key_in_result, " makespan ", current_time)
                except Exception as e:
                    # Something bad occured...
                    print("Computation failed", scheduler, execution_mode)
                    makespans[f"Scenario {scn}"][key_in_result]["executed"] = "Fail"
                    makespans[f"Scenario {scn}"][key_in_result]["timing"] = "Fail"
                    makespans[f"Scenario {scn}"][key_in_result]["schedule"] = "Fail"

        # with open(result_file, "r+") as jsonfile:
        #     jsonfile.seek(0, os.SEEK_END)
        #     jsonfile.seek(jsonfile.tell() - 4, os.SEEK_SET)
        #     char = jsonfile.read(1)
        #     jsonfile.seek(0, os.SEEK_END)
        #     jsonfile.seek(jsonfile.tell() - 3, os.SEEK_SET)
        #     if char == "{":  # first bench
        #         jsonfile.write("\n")
        #     elif char == "}":
        #         jsonfile.write(",\n")
        #     jsonfile.write(f'"Benchmark {bench_id}": ')
        #     json.dump(makespans, jsonfile, indent=2)
        #     jsonfile.write("\n}\n")
        return makespans


def test(test_loader, test_list, model, device, result_file):
    model.eval()
    with open(result_file, "w") as jsonfile:
        jsonfile.write("{\n}\n")

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    merged_res = {}
    for batch_idx, data in enumerate(
        tqdm(test_loader, desc="Benchmark Loop", leave=True)
    ):
        if batch_idx < 3:
            continue
        data.to(device)
        out = model(data)
        data.out = out
        # TODO: is there a cleaner way to do this?
        data._slice_dict["out"] = data._slice_dict["x"]
        data._inc_dict["out"] = data._inc_dict["x"]

        ms = execute_schedule(
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


if __name__ == "__main__":
    data_list = torch.load("../torch_data/data_list.tch")
    train_list = torch.load("../torch_data/train_list.tch")
    test_list = list(set(range(len(data_list))) - set(train_list))
    test_loader = DataLoader(
        [data_list[d] for d in test_list], batch_size=1, shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = ResTransformer
    # Net = ResGINE
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
