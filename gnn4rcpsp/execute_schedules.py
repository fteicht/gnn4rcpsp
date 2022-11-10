from collections import defaultdict
from datetime import datetime
import json
import os
import torch
from torch_geometric.data import DataLoader
from time import perf_counter
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocess import Manager
import multiprocess.context as ctx

from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRCPSPModel,
    create_poisson_laws,
)

from executor import ExecutionMode, Scheduler, SchedulingExecutor, CPSatSpecificParams
from infer_schedules import build_rcpsp_model

from models import ResTransformer
from tqdm import tqdm

SchedulerNames = {Scheduler.SGS: "SGS", Scheduler.CPSAT: "CPSAT"}

ExecutionModeNames = {
    ExecutionMode.REACTIVE_AVERAGE: "REACTIVE_AVG",
    ExecutionMode.REACTIVE_WORST: "REACTIVE_WORST",
    ExecutionMode.REACTIVE_BEST: "REACTIVE_BEST",
    ExecutionMode.HINDSIGHT_LEX: "HINDSIGHT_LEX",
    ExecutionMode.HINDSIGHT_DBP: "HINDSIGHT_DBP",
}

NUM_SAMPLES = 30


def execute_schedule(
    device, model, bench_id, tests_done, nb_tests, result_file, data, json_lock
):
    data.to(device)
    out = model(data)
    data.out = out
    # TODO: is there a cleaner way to do this?
    data._slice_dict["out"] = data._slice_dict["x"]
    data._inc_dict["out"] = data._inc_dict["x"]
    data_batch = data

    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    data_list = data_batch.to_data_list()
    for data in data_list:
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

        for scn in tqdm(range(NUM_SAMPLES), desc="Scenario Loop", leave=False):
            sample_rcpsp = uncertain_rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )

            for scheduler, execution_mode in tqdm(
                [
                    (Scheduler.SGS, ExecutionMode.REACTIVE_AVERAGE),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_WORST),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_BEST),
                    # (Scheduler.CPSAT, ExecutionMode.REACTIVE_AVERAGE),
                    (Scheduler.SGS, ExecutionMode.HINDSIGHT_LEX),
                    (Scheduler.SGS, ExecutionMode.HINDSIGHT_DBP),
                ],
                desc="Mode Loop",
                leave=False,
            ):
                try:
                    executor = SchedulingExecutor(
                        rcpsp_model,
                        model,
                        device,
                        scheduler,
                        execution_mode,
                        NUM_SAMPLES,
                        CPSatSpecificParams(
                            do_minimization=True,
                            warm_start_with_gnn=False,
                            time_limit_seconds=5,
                        )
                        if scheduler == Scheduler.CPSAT
                        else None,
                    )
                    setup_name = f"{SchedulerNames[scheduler]}-{ExecutionModeNames[execution_mode]}"
                    stop = False
                    executed_schedule, current_time = executor.reset(
                        sim_rcpsp=sample_rcpsp
                    )
                    makespans[f"Scenario {scn}"][setup_name] = {"expectations": []}
                    timer = perf_counter()

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

                        makespans[f"Scenario {scn}"][setup_name]["expectations"].append(
                            expected_makespan
                        )

                    makespans[f"Scenario {scn}"][setup_name]["executed"] = current_time
                    makespans[f"Scenario {scn}"][setup_name]["timing"] = (
                        perf_counter() - timer
                    )
                    makespans[f"Scenario {scn}"][setup_name][
                        "schedule"
                    ] = executed_schedule.rcpsp_schedule
                except Exception as e:
                    print(e)
                    continue

        json_lock.acquire()
        tests_done.value += 1
        print(f"Done {tests_done.value} over {nb_tests}")
        with open(result_file, "r+") as jsonfile:
            jsonfile.seek(0, os.SEEK_END)
            jsonfile.seek(jsonfile.tell() - 4, os.SEEK_SET)
            char = jsonfile.read(1)
            jsonfile.seek(0, os.SEEK_END)
            jsonfile.seek(jsonfile.tell() - 3, os.SEEK_SET)
            if char == "{":  # first bench
                jsonfile.write("\n")
            elif char == "}":
                jsonfile.write(",\n")
            jsonfile.write(f'"Benchmark {bench_id}": ')
            json.dump(makespans, jsonfile, indent=2)
            jsonfile.write("\n}\n")
        json_lock.release()


def do_test(data, bench_id, tests_done, nb_tests, result_file, json_lock):
    test_loader = DataLoader([data], batch_size=1, shuffle=False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model.eval()
    for d in test_loader:
        execute_schedule(
            device, model, bench_id, tests_done, nb_tests, result_file, d, json_lock
        )


def test(data_list, test_list, result_file):
    m = Manager()
    json_lock = m.Lock()
    tests_done = m.Value("i", 0)

    with open(result_file, "w") as jsonfile:
        jsonfile.write("{\n}\n")

    with Pool() as pool:
        pool.map(
            lambda x: do_test(*x),
            [
                (
                    data,
                    bench_id,
                    tests_done,
                    len(test_list),
                    result_file,
                    json_lock,
                )
                for bench_id, data in [
                    (bench_id, data_list[bench_id]) for bench_id in test_list
                ]
            ],
        )


if __name__ == "__main__":
    ctx._force_start_method("spawn")
    data_list = torch.load("../torch_data/data_list.tch")
    train_list = torch.load("../torch_data/train_list.tch")
    test_list = list(set(range(len(data_list))) - set(train_list))
    run_id = timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print(f"Run ID: {run_id}")
    result_file = f"../hindsight_vs_reactive_{run_id}.json"
    result = test(
        data_list=data_list,
        test_list=test_list,
        result_file=result_file,
    )
