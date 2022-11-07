from datetime import datetime
import json
import torch
from torch_geometric.data import DataLoader
from time import perf_counter
from executor import ExecutionMode, Scheduler
from models import ResTransformer

from executor import SchedulingExecutor
from infer_schedules import build_rcpsp_model

NUM_SAMPLES = 100


def execute_schedule(bench_id, data):
    data_batch = data
    batch_results = {}

    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    data_list = data_batch.to_data_list()
    cnt = 0
    for data in data_list:
        cnt += 1
        print(f"Instance {cnt}")

        t2t, dur, r2t, rc, con, ref_makespan = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
            data.con.view(*data.con_shape).data.cpu().detach().numpy(),
            data.reference_makespan,
        )
        rcpsp_model = build_rcpsp_model(t2t, dur, r2t, rc)[0]
        hindsight_makespans = {}
        reactive_makespans = {}

        for scn in range(NUM_SAMPLES):
            # HINDSIGHT
            hindsight_executor = SchedulingExecutor(
                rcpsp_model,
                model,
                device,
                Scheduler.SGS,
                ExecutionMode.HINDSIGHT_DBP,
                NUM_SAMPLES,
            )
            stop = False
            executed_schedule, current_time = hindsight_executor.reset()
            hindsight_makespans[f"Scenario {scn}"] = {"expectations": []}
            timer = perf_counter()

            while not stop:
                (
                    next_tasks,
                    next_start,
                    expected_makespan,
                    expected_schedule,
                ) = hindsight_executor.next_tasks(executed_schedule, current_time)

                current_time, executed_schedule, stop = hindsight_executor.progress(
                    next_tasks, next_start, expected_schedule
                )

                hindsight_makespans[f"Scenario {scn}"]["expectations"].append(
                    expected_makespan
                )

            hindsight_makespans[f"Scenario {scn}"]["executed"] = current_time
            hindsight_makespans[f"Scenario {scn}"]["timing"] = perf_counter() - timer

            # REACTIVE
            reactive_executor = SchedulingExecutor(
                rcpsp_model,
                model,
                device,
                Scheduler.SGS,
                ExecutionMode.REACTIVE,
                NUM_SAMPLES,
            )
            stop = False
            executed_schedule, current_time = reactive_executor.reset()
            reactive_makespans[f"Scenario {scn}"] = {"expectations": []}
            timer = perf_counter()

            while not stop:
                (
                    next_tasks,
                    next_start,
                    expected_makespan,
                    expected_schedule,
                ) = reactive_executor.next_tasks(executed_schedule, current_time)

                current_time, executed_schedule, stop = reactive_executor.progress(
                    next_tasks, next_start, expected_schedule
                )

                reactive_makespans[f"Scenario {scn}"]["expectations"].append(
                    expected_makespan
                )

            reactive_makespans[f"Scenario {scn}"]["executed"] = current_time
            reactive_makespans[f"Scenario {scn}"]["timing"] = perf_counter() - timer

        batch_results[f"Benchmark {bench_id}"] = {
            "hindsight": hindsight_makespans,
            "reactive": reactive_makespans,
        }


def test(test_loader, test_list, model, device, writer):
    result_dict = {}
    model.eval()

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    for batch_idx, data in enumerate(test_loader):
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
    # writer = SummaryWriter(f's3://iooda-gnn4rcpsp-bucket/tensorboard_logs/{run_id}')
    # writer = SummaryWriter(f'../tensorboard_logs/{run_id}')
    writer = None
    result = test(
        test_loader=test_loader,
        test_list=test_list,
        model=model,
        device=device,
        writer=writer,
    )
    with open(f"../hindsight_vs_reactive_{run_id}.json", "w") as jsonfile:
        json.dump(result, jsonfile, indent=2)
