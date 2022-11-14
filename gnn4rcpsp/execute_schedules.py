import json
import os
import warnings
from collections import defaultdict
from datetime import datetime
from multiprocessing import Manager
from time import perf_counter

import torch
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
from infer_schedules import build_rcpsp_model
from models import ResTransformer
from numba.core.errors import NumbaTypeSafetyWarning
from pathos.multiprocessing import ProcessingPool as Pool
from torch_geometric.data import DataLoader
from tqdm import tqdm

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)

SchedulerNames = {Scheduler.SGS: "SGS", Scheduler.CPSAT: "CPSAT"}

ExecutionModeNames = {
    ExecutionMode.REACTIVE_AVERAGE: "REACTIVE_AVG",
    ExecutionMode.REACTIVE_WORST: "REACTIVE_WORST",
    ExecutionMode.REACTIVE_BEST: "REACTIVE_BEST",
    ExecutionMode.HINDSIGHT_LEX: "HINDSIGHT_LEX",
    ExecutionMode.HINDSIGHT_DBP: "HINDSIGHT_DBP",
}

NUM_HINDSIGHT_SAMPLES = 20
NUM_INSTANCE_SCENARIOS = 4
PARALLEL = False


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
            range_around_mean_duration=1,
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

        for scn in tqdm(
            range(NUM_INSTANCE_SCENARIOS), desc="Scenario Loop", leave=False
        ):
            sample_rcpsp = uncertain_rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )

            for scheduler, execution_mode in tqdm(
                [
                    (Scheduler.SGS, ExecutionMode.REACTIVE_AVERAGE),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_WORST),
                    (Scheduler.SGS, ExecutionMode.REACTIVE_BEST),
                    (Scheduler.CPSAT, ExecutionMode.REACTIVE_AVERAGE),
                    (Scheduler.SGS, ExecutionMode.HINDSIGHT_LEX),
                    (Scheduler.SGS, ExecutionMode.HINDSIGHT_DBP),
                ],
                desc="Mode Loop",
                leave=False,
            ):
                setup_name = (
                    f"{SchedulerNames[scheduler]}-{ExecutionModeNames[execution_mode]}"
                )
                if PARALLEL:
                    json_lock.acquire()
                print(
                    f"Processing {setup_name} of scenario {scn} of benchmark {bench_id} - overall {float(tests_done.value if PARALLEL else tests_done) / float(nb_tests) * 100.0}%"
                )
                if PARALLEL:
                    json_lock.release()
                try:
                    executor = SchedulingExecutor(
                        rcpsp_model,
                        model,
                        device,
                        scheduler,
                        execution_mode,
                        NUM_HINDSIGHT_SAMPLES,
                        CPSatSpecificParams(
                            do_minimization=True,
                            warm_start_with_gnn=False,
                            time_limit_seconds=0.5,
                        )
                        if scheduler == Scheduler.CPSAT
                        else None,
                        params_remaining_rcpsp=ParamsRemainingRCPSP.KEEP_FULL_RCPSP
                        if scheduler == Scheduler.SGS
                        else ParamsRemainingRCPSP.EXACT_REMAINING,
                        poisson_laws=poisson_laws,
                    )
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
                        print("cur time , ", current_time)
                        makespans[f"Scenario {scn}"][setup_name]["expectations"].append(
                            float(expected_makespan)
                        )

                    makespans[f"Scenario {scn}"][setup_name]["executed"] = int(
                        current_time
                    )
                    makespans[f"Scenario {scn}"][setup_name]["timing"] = (
                        perf_counter() - timer
                    )
                    makespans[f"Scenario {scn}"][setup_name]["schedule"] = {
                        t: {
                            k: int(executed_schedule.rcpsp_schedule[t][k])
                            for k in executed_schedule.rcpsp_schedule[t]
                        }
                        for t in executed_schedule.rcpsp_schedule
                    }
                    print("method ", setup_name, " : ", current_time)
                except Exception as e:
                    makespans[f"Scenario {scn}"][setup_name]["executed"] = "Fail"
                    makespans[f"Scenario {scn}"][setup_name]["timing"] = "Fail"
                    makespans[f"Scenario {scn}"][setup_name]["schedule"] = "Fail"
                    print(e)
                    continue

        if PARALLEL:
            json_lock.acquire()
            tests_done.value += 1
        else:
            tests_done += 1
        print(f"Done {tests_done.value if PARALLEL else tests_done} over {nb_tests}")
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
        if PARALLEL:
            json_lock.release()


def test(test_loader, test_list, model, device, result_file):
    model.eval()
    if PARALLEL:
        m = Manager()
        json_lock = m.Lock()
        tests_done = m.Value("i", 0)
    else:
        tests_done = 0

    with open(result_file, "w") as jsonfile:
        jsonfile.write("{\n}\n")

    if PARALLEL:
        with Pool(ncpus=10) as pool:
            pool.map(
                lambda x: execute_schedule(*x),
                [
                    (
                        device,
                        model,
                        test_list[
                            batch_idx
                        ],  # batch_size==1 thus batch_idx==test_instance_id
                        tests_done,
                        len(test_list),
                        result_file,
                        data,
                        json_lock,
                    )
                    for batch_idx, data in enumerate(test_loader)
                ],
            )
    else:
        for batch_idx, data in tqdm(
            enumerate(test_loader), desc="Benchmark Loop", leave=False
        ):
            execute_schedule(
                device,
                model,
                test_list[batch_idx],  # batch_size==1 thus batch_idx==test_instance_id
                tests_done,
                len(test_list),
                result_file,
                data,
                None,
            )


if __name__ == "__main__":
    data_list = torch.load("../torch_data/data_list.tch")
    train_list = torch.load("../torch_data/train_list.tch")
    test_list = list(set(range(len(data_list))) - set(train_list))
    filtered_test_list = [
        1460,
        509,
        459,
        450,
        514,
        234,
        237,
        391,
        285,
        1720,
        231,
        69,
        406,
        399,
        1728,
        1948,
        1949,
        502,
        461,
        1469,
        1425,
        471,
        464,
        527,
        2032,
        1244,
        1858,
        19,
        1112,
        303,
        1556,
        242,
        37,
        66,
        522,
        473,
        1555,
    ]
    filtered_test_list = [1460]
    filtered_test_list = [
        1460,
        509,
        459,
        450,
        514,
        234,
        237,
        391,
        285,
        1720,
        231,
        69,
        406,
        399,
        1728,
        1948,
        1949,
        502,
        461,
        1469,
        1425,
        471,
        464,
        527,
        2032,
        1244,
        1858,
        19,
        1112,
        303,
        1556,
        242,
        37,
        66,
        522,
        473,
        1555,
    ]
    assert set(filtered_test_list).issubset(set(test_list))
    test_loader = DataLoader(
        [data_list[d] for d in filtered_test_list], batch_size=1, shuffle=False
    )
    if PARALLEL:
        device = "cpu"
    else:
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
        test_list=filtered_test_list,
        model=model,
        device=device,
        result_file=result_file,
    )
