from collections import defaultdict
from datetime import datetime
import json
import torch
from torch_geometric.data import DataLoader
from time import perf_counter

from tqdm import tqdm
from executor import ExecutionMode, Scheduler
from models import ResTransformer

from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws,
)

from executor import SchedulingExecutor
from infer_schedules import build_rcpsp_model

NUM_SAMPLES = 100

ExecutionModeNames = {
    ExecutionMode.REACTIVE_AVERAGE: "REACTIVE_AVG",
    ExecutionMode.REACTIVE_WORST: "REACTIVE_WORST",
    ExecutionMode.REACTIVE_BEST: "REACTIVE_BEST",
    ExecutionMode.HINDSIGHT_LEX: "HINDSIGHT_LEX",
    ExecutionMode.HINDSIGHT_DBP: "HINDSIGHT_DBP",
}


def execute_schedule(bench_id, data):
    data_batch = data
    batch_results = {}

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

        for scn in tqdm(range(NUM_SAMPLES), desc="Scenario Loop", leave=False):
            sample_rcpsp = uncertain_rcpsp.create_rcpsp_model(
                MethodRobustification(MethodBaseRobustification.SAMPLE)
            )

            for execution_mode in tqdm(
                [
                    ExecutionMode.REACTIVE_AVERAGE,
                    ExecutionMode.REACTIVE_WORST,
                    ExecutionMode.REACTIVE_BEST,
                    ExecutionMode.HINDSIGHT_LEX,
                    ExecutionMode.HINDSIGHT_DBP,
                ],
                desc="Mode Loop",
                leave=False,
            ):
                executor = SchedulingExecutor(
                    rcpsp_model,
                    model,
                    device,
                    Scheduler.SGS,
                    execution_mode,
                    NUM_SAMPLES,
                )
                stop = False
                executed_schedule, current_time = executor.reset(sim_rcpsp=sample_rcpsp)
                makespans[f"Scenario {scn}"][ExecutionModeNames[execution_mode]] = {
                    "expectations": []
                }
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

                    makespans[f"Scenario {scn}"][ExecutionModeNames[execution_mode]][
                        "expectations"
                    ].append(expected_makespan)

                makespans[f"Scenario {scn}"][ExecutionModeNames[execution_mode]][
                    "executed"
                ] = current_time
                makespans[f"Scenario {scn}"][ExecutionModeNames[execution_mode]][
                    "timing"
                ] = (perf_counter() - timer)
                makespans[f"Scenario {scn}"][ExecutionModeNames[execution_mode]][
                    "schedule"
                ] = executed_schedule

        batch_results[f"Benchmark {bench_id}"] = makespans


def test(test_loader, test_list, model, device):
    result_dict = {}
    model.eval()

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    for batch_idx, data in enumerate(
        tqdm(test_loader, desc="Benchmark Loop", leave=True)
    ):
        data.to(device)
        out = model(data)
        data.out = out
        # TODO: is there a cleaner way to do this?
        data._slice_dict["out"] = data._slice_dict["x"]
        data._inc_dict["out"] = data._inc_dict["x"]

        result_dict.update(
            execute_schedule(
                test_list[batch_idx],  # batch_size==1 thus batch_idx==test_instance_id
                data,
            ),
        )


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
    result = test(
        test_loader=test_loader,
        test_list=test_list,
        model=model,
        device=device,
    )
    with open(f"../hindsight_vs_reactive_{run_id}.json", "w") as jsonfile:
        json.dump(result, jsonfile, indent=2)
