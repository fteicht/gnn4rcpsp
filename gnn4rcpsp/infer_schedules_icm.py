import datetime
import json
from pickletools import optimize
import numpy as np
from sniffio import current_async_library
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm
from time import perf_counter

import nlopt
from ortools.sat.python import cp_model

from learn_schedules import check_solution, compute_loss

from models import ResTransformer, ResGINE

CPSAT_TIME_LIMIT = 600  # 10 mn
CPSAT_FEAS_TIME_LIMIT = 30  # 30 s
CONSTRAINT_MAKESPAN = True
SEARCH_FOR_OPTIMALITY = False


def make_feasible_nlopt(data):
    def objective(xorig, x, dur, grad):
        v = np.array([x[i] - xorig[i] for i in range(len(x))])
        for j in range(len(grad)):
            grad[j] = (1.0 / float(len(x))) * 2.0 * v[j]
        return (1.0 / float(len(x))) * np.dot(v, v.T)

    def all_positive_contraint(i, x, grad):
        for j in range(len(grad)):
            grad[j] = 0.0
        grad[i] = -1.0
        return -x[i]

    def precedence_constraint(t, p, t2t, x, grad):
        for j in range(len(grad)):
            grad[j] = 0.0
        grad[p] = 1.0
        grad[t] = -1.0
        return x[p] + t2t[t][p] - x[t]

    def resource_constraint(r, t, r2t, dur, x, grad):
        # Wrong derivative on purpose!
        # (assume ramp function for constraintderivative but heaviside for constraint value)
        for j in range(len(grad)):
            grad[j] = 0.0
        for tp in range(r2t.shape[1]):
            if tp != t and r2t[r][tp] > 0:
                grad[tp] = (
                    r2t[r][tp] * (2 * x[t] - 2 * x[tp] - dur[tp])
                    if (x[t] - x[tp]) * (x[tp] + dur[tp] - x[t]) > 0
                    else 0.0
                )
        grad[t] = np.dot(
            (
                np.array(
                    [
                        (x[t] - x[tp]) * (x[tp] + dur[tp] - x[t])
                        for tp in range(r2t.shape[1])
                        if tp != t and r2t[r][tp] > 0
                    ]
                )
                > 0
            )
            .astype(int)
            .T,
            np.array(
                [
                    (2 * x[tp] - 2 * x[t] + dur[tp]) * r2t[r][tp]
                    for tp in range(r2t.shape[1])
                    if tp != t and r2t[r][tp] > 0
                ]
            ),
        )
        return (
            r2t[r][t]
            - rc[r]
            + np.dot(
                (
                    np.array(
                        [
                            (x[t] - x[tp]) * (x[tp] + dur[tp] - x[t])
                            for tp in range(r2t.shape[1])
                            if tp != t and r2t[r][tp] > 0
                        ]
                    )
                    > 0
                )
                .astype(int)
                .T,
                np.array(
                    [
                        r2t[r][tp]
                        for tp in range(r2t.shape[1])
                        if tp != t and r2t[r][tp] > 0
                    ]
                ),
            )
        )

    data_batch = data
    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    data_list = data_batch.to_data_list()
    for data in data_list:

        t2t, dur, r2t, rc, con, ref_makespan = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
            data.con.view(*data.con_shape).data.cpu().detach().numpy(),
            data.reference_makespan,
        )

        xorig = data.out[len(rc) :, 0].cpu().detach().numpy()

        nb_tasks = len(dur)
        # opt = nlopt.opt(nlopt.LD_MMA, nb_tasks)
        opt = nlopt.opt(nlopt.LD_MMA, nb_tasks)
        # don't use lower bounds since they assume xorig to be feasible
        # => prefer all_positive_constraint because inequality constraints don't assume this
        # opt.set_lower_bounds(np.zeros(shape=(nb_tasks,), dtype=np.float32))

        opt.set_min_objective(lambda x, grad: objective(xorig, x, dur, grad))

        for i in range(nb_tasks):
            opt.add_inequality_constraint(
                lambda x, grad: all_positive_contraint(i, x, grad), 1e-8
            )

        for t in range(t2t.shape[0]):
            for p in range(t2t.shape[1]):
                if t2t[t][p] > 0:
                    opt.add_inequality_constraint(
                        lambda x, grad: precedence_constraint(t, p, t2t, x, grad), 1e-8
                    )

        for r in range(r2t.shape[0]):
            for t in range(r2t.shape[1]):
                if r2t[r][t] > 0:
                    opt.add_inequality_constraint(
                        lambda x, grad: resource_constraint(r, t, r2t, dur, x, grad),
                        1e-8,
                    )

        opt.set_xtol_rel(1e-4)
        xdest = opt.optimize(xorig)

        feasible = True
        grad = np.zeros(shape=(nb_tasks,))

        for i in range(nb_tasks):
            feasible = feasible and (all_positive_contraint(i, xdest, grad) <= 1e-8)

        for t in range(t2t.shape[0]):
            for p in range(t2t.shape[1]):
                if t2t[t][p] > 0:
                    feasible = feasible and (
                        precedence_constraint(t, p, t2t, xdest, grad) <= 1e-8
                    )

        for r in range(r2t.shape[0]):
            for t in range(r2t.shape[1]):
                if r2t[r][t] > 0:
                    feasible = feasible and (
                        resource_constraint(r, t, r2t, dur, xdest, grad) <= 1e-8
                    )

        print("result code = ", opt.last_optimize_result())
        print("makespan: {}".format(np.max(xdest + dur) / ref_makespan))
        print("feasible: ", " YES" if feasible else " NO")


def make_feasible_cpsat(data):
    data_batch = data

    cpsat_result = {
        "feasibility_timing": [],
        "feasibility_rel_makespan_cor": [],
        "feasibility_rel_makespan_ref": [],
        "feasibility_abs_makespan": [],
    }

    if SEARCH_FOR_OPTIMALITY:
        cpsat_result.update(
            {
                "optimization_timing": [],
                "optimization_rel_makespan_cor": [],
                "optimization_rel_makespan_ref": [],
                "optimization_abs_makespan": [],
            }
        )

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
        xorig = np.around(
            data.out[len(rc) :, 0].cpu().detach().numpy(), decimals=0
        ).astype(int)
        makespan_orig = max(xorig + dur)
        start_time = perf_counter()
        current_makespan_scaling = 1.0
        previous_makespan_scaling = current_makespan_scaling
        feasible_solution = None

        while perf_counter() - start_time < CPSAT_TIME_LIMIT:

            model = cp_model.CpModel()
            horizon = int(makespan_orig * current_makespan_scaling * 1.2)
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

            # for t in range(t2t.shape[0]):
            #     model.Add(starts[t] + durs[t] == ends[t])

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

            if CONSTRAINT_MAKESPAN:
                scaled_makespan = model.NewConstant(
                    int(makespan_orig * current_makespan_scaling)
                )
                model.Add(makespan <= scaled_makespan)
                if current_makespan_scaling > previous_makespan_scaling:
                    previous_makespan = model.NewConstant(
                        int(makespan_orig * previous_makespan_scaling)
                    )
                    model.Add(makespan > previous_makespan)

            # First we search for a feasible solution
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = (
                CPSAT_FEAS_TIME_LIMIT if CONSTRAINT_MAKESPAN else CPSAT_TIME_LIMIT
            )
            xorig_list = xorig.tolist()
            for i, x in enumerate(xorig_list):
                model.AddHint(starts[i], x)
            # model._CpModel__model.solution_hint.vars.extend(list(range(len(dur))))
            # model._CpModel__model.solution_hint.values.extend(xorig.tolist())
            cur_time = perf_counter()
            status = solver.Solve(model)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                feasible_solution = [solver.Value(s) for s in starts]
                print(
                    "FEASIBLE: [{}-{}]".format(
                        int(makespan_orig * previous_makespan_scaling),
                        int(makespan_orig * current_makespan_scaling),
                    )
                )
                break
            elif status == cp_model.INFEASIBLE:
                print(
                    "INFEASIBLE: [{}-{}]".format(
                        int(makespan_orig * previous_makespan_scaling),
                        int(makespan_orig * current_makespan_scaling),
                    )
                )
                previous_makespan_scaling = current_makespan_scaling
                current_makespan_scaling += 0.05
                if CONSTRAINT_MAKESPAN:
                    continue
                else:
                    break
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError("Invalid CPSAT model.")

        if feasible_solution is not None:
            cpsat_result["feasibility_timing"].append(
                perf_counter() - (cur_time if not CONSTRAINT_MAKESPAN else start_time)
            )
            cpsat_result["feasibility_rel_makespan_cor"].append(
                solver.Value(makespan) / max(xorig + dur)
            )
            cpsat_result["feasibility_rel_makespan_ref"].append(
                solver.Value(makespan) / max(ref_makespan)
            )
            cpsat_result["feasibility_abs_makespan"].append(int(solver.Value(makespan)))
        else:
            cpsat_result["feasibility_timing"].append(-1)
            cpsat_result["feasibility_rel_makespan_cor"].append(-1)
            cpsat_result["feasibility_rel_makespan_ref"].append(-1)
            cpsat_result["feasibility_abs_makespan"].append(-1)
            continue

        # Second we search for an optimal solution
        if SEARCH_FOR_OPTIMALITY:
            model.Minimize(makespan)
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = CPSAT_TIME_LIMIT
            model.ClearHints()
            for i, x in enumerate(feasible_solution):
                model.AddHint(starts[i], x)
            cur_time = perf_counter()
            status = solver.Solve(model)

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print("Found optimal solution")
                cpsat_result["optimization_timing"].append(perf_counter() - cur_time)
                cpsat_result["optimization_rel_makespan_cor"].append(
                    solver.Value(makespan) / max(xorig + dur)
                )
                cpsat_result["optimization_rel_makespan_ref"].append(
                    solver.Value(makespan) / max(ref_makespan)
                )
                cpsat_result["optimization_abs_makespan"].append(
                    int(solver.Value(makespan))
                )
            elif status == cp_model.INFEASIBLE:
                cpsat_result["optimization_timing"].append(-1)
                cpsat_result["optimization_rel_makespan_cor"].append(-1)
                cpsat_result["optimization_rel_makespan_ref"].append(-1)
                cpsat_result["optimization_abs_makespan"].append(-1)
                continue
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError("Invalid CPSAT model.")

    for k in cpsat_result:
        cpsat_result[k] = np.mean([r for r in cpsat_result[k] if r > -1])

    if len(data_list) == 1:  # batch sizes of 1 so 1 batch == 1 instance
        if np.isnan(cpsat_result["feasibility_timing"]):
            cpsat_result["schedule"] = "INFEASIBLE"
        else:
            cpsat_result["schedule"] = {
                "task {}".format(task_id + 1): {
                    "start_time": int(solver.Value(start)),
                    "end_time": int(solver.Value(start) + dur[task_id]),
                }
                for task_id, start in enumerate(starts)
            }

    return cpsat_result


def test(test_loader, test_list, model, device, writer):
    result_dict = {}
    model.eval()

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    for batch_idx, data in enumerate(test_loader):
        cur_time = perf_counter()
        data.to(device)
        out = model(data)
        inference_time = (perf_counter() - cur_time) / float(data.num_graphs)
        data.out = out
        # TODO: is there a cleaner way to do this?
        data._slice_dict["out"] = data._slice_dict["x"]
        data._inc_dict["out"] = data._inc_dict["x"]

        loss = compute_loss(data=data, device=device)
        if writer is not None:
            writer.add_scalar("loss", loss.item(), batch_idx)

        batch_violations = check_solution(data)
        rel_makespan = make_feasible_cpsat(data)

        batch_result = {
            "benchmark_id": test_list[
                batch_idx
            ],  # batch_size==1 thus batch_idx==test_instance_id
            "inference_time": inference_time,
            "checking_all_positive_per": batch_violations["all_positive_per"],
            "checking_all_positive_mag": batch_violations["all_positive_mag"],
            "checking_precedence_per": batch_violations["precedence_per"],
            "checking_precedence_mag": batch_violations["precedence_mag"],
            "checking_resource_per": batch_violations["resource_per"],
            "checking_resource_mag": batch_violations["resource_mag"],
            "checking_makespan": batch_violations["makespan"],
            "feasibility_timing": rel_makespan["feasibility_timing"],
            "feasibility_rel_makespan_cor": rel_makespan[
                "feasibility_rel_makespan_cor"
            ],
            "feasibility_rel_makespan_ref": rel_makespan[
                "feasibility_rel_makespan_ref"
            ],
            "feasibility_abs_makespan": rel_makespan["feasibility_abs_makespan"],
        }

        if SEARCH_FOR_OPTIMALITY:
            batch_result.update(
                {
                    "optimization_timing": rel_makespan["optimization_timing"],
                    "optimization_rel_makespan_cor": rel_makespan[
                        "optimization_rel_makespan_cor"
                    ],
                    "optimization_rel_makespan_ref": rel_makespan[
                        "optimization_rel_makespan_ref"
                    ],
                    "optimization_abs_makespan": rel_makespan[
                        "optimization_abs_makespan"
                    ],
                }
            )

        if "schedule" in rel_makespan:
            batch_result.update({"schedule": rel_makespan["schedule"]})

        result_dict.update({"Benchmark {}".format(batch_idx): batch_result})

        if writer is not None:
            for k, v in batch_result.items():
                if k != "schedule":
                    writer.add_scalar(k, v, batch_idx)
            writer.flush()

    return result_dict


if __name__ == "__main__":
    data_list = torch.load("../torch/data_list.tch")
    train_list = torch.load("../torch/train_list.tch")
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
