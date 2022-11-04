from copy import deepcopy
import os
from typing import Dict, List, Set, Tuple, Union
import torch
import numpy as np
from time import perf_counter
from collections import defaultdict, namedtuple
from ortools.sat.python import cp_model
from models import ResTransformer
from skdecide.discrete_optimization.rcpsp.parser.rcpsp_parser import parse_file
from skdecide.discrete_optimization.rcpsp.rcpsp_model import (
    UncertainRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    MethodBaseRobustification,
    MethodRobustification,
    create_poisson_laws,
)
from skdecide.discrete_optimization.rcpsp.rcpsp_utils import (
    compute_nice_resource_consumption,
)
from graph import Graph

from executor import SchedulingExecutor
from infer_schedules import build_rcpsp_model

NUM_SAMPLES = 100


def execute_stochastic_schedule(data):
    data_batch = data

    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    data_list = data_batch.to_data_list()
    cnt = 0
    for data in data_list:
        cnt += 1
        print("Instance {}".format(cnt))

        t2t, dur, r2t, rc, con, ref_makespan = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
            data.con.view(*data.con_shape).data.cpu().detach().numpy(),
            data.reference_makespan,
        )
        rcpsp_model = build_rcpsp_model(t2t, dur, r2t, rc, ref_makespan)
        executor = SchedulingExecutor(rcpsp_model, model, device, NUM_SAMPLES)


def test(test_loader, test_list, model, device, writer):
    result_dict = {}
    model.eval()

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    for batch_idx, data in enumerate(test_loader):
        cur_time = perf_counter()
        data.to(device)
        out = model(data)
        data.out = out
        # TODO: is there a cleaner way to do this?
        data._slice_dict["out"] = data._slice_dict["x"]
        data._inc_dict["out"] = data._inc_dict["x"]

        execute_stochastic_schedule(data)
        execute_reactive_schedule(data)


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
        torch.load("../torch_folder/model_ResTransformer_256_50000.tch")
    )
    run_id = timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
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
    with open(f"../cp_solutions/inference_vs_cpsat_{run_id}.json", "w") as jsonfile:
        json.dump(result, jsonfile, indent=2)
