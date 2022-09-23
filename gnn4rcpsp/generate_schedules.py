import datetime
from csv import writer
from time import perf_counter
from typing import List

import torch
from docplex.cp.modeler import max_of
from ortools.sat.python import cp_model
from torch_geometric.data import DataLoader

try:
    from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
    from discrete_optimization.rcpsp.solver.cp_solvers import (
        CP_RCPSP_MZN,
        CPSolverName,
        ParametersCP,
    )
except Exception as e:
    print(
        "install discreteopt standalone, this branch in particular..."
        "https://github.com/g-poveda/discrete-optimization/tree/cpsolver_time_tracer, pip install --editable ."
    )
    # raise ImportError("Missing discrete opt library")

CPSAT_TIME_LIMIT = 900  # 15 mn


class CPSatSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, bench_id, start_time, starts, ref_makespan, makespan, outfile):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._bench_id = bench_id
        self._start_time = start_time
        self._starts = starts
        self._ref_makespan = ref_makespan
        self._makespan = makespan
        self._outfile = outfile

    def on_solution_callback(self):
        with open(self._outfile, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(
                [
                    self._bench_id,
                    perf_counter() - self._start_time,
                    max(self._ref_makespan),
                    self.Value(self._makespan),
                ]
                + [self.Value(s) for s in self._starts]
            )
            f_object.close()


def solve_with_cpsat(data_list, device, outfile):

    data_loader = DataLoader(data_list, batch_size=32, shuffle=False)
    bench_id = 0

    for batch_idx, data_batch in enumerate(data_loader):

        data_batch.to(device)

        for data in data_batch.to_data_list():

            print("Solving bench {}/{} ...".format(bench_id, len(data_list) - 1))

            t2t, dur, r2t, rc, con, ref_makespan = (
                data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
                data.dur.data.cpu().detach().numpy(),
                data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
                data.rc.data.cpu().detach().numpy(),
                data.con.view(*data.con_shape).data.cpu().detach().numpy(),
                data.reference_makespan,
            )

            model = cp_model.CpModel()
            horizon = int(max(dur) * len(dur))
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
            model.Minimize(makespan)

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = CPSAT_TIME_LIMIT
            start_time = perf_counter()
            status = solver.Solve(
                model,
                CPSatSolutionPrinter(
                    bench_id, start_time, starts, ref_makespan, makespan, outfile
                ),
            )

            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print(
                    "Found optimal solution for bench {}/{} in {} s.".format(
                        bench_id, len(data_list) - 1, perf_counter() - start_time
                    )
                )
            elif status == cp_model.INFEASIBLE:
                print(
                    "Could not find a feasible solution for bench {}/{}".format(
                        bench_id, len(data_list) - 1
                    )
                )
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError(
                    "Invalid CPSAT model for bench {}/{}".format(
                        bench_id, len(data_list) - 1
                    )
                )

            bench_id += 1


def solve_with_discreteoptim(
    data_list,
    device,
    outfile: str,
    cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
    time_out=25,
):
    def build_rcpsp_model(t2t, dur, r2t, rc):

        nb_tasks = len(dur)
        tasks_list = list(range(nb_tasks))
        successors = {i: [] for i in tasks_list}
        mode_details = {
            tasks_list[i]: {1: {"duration": dur[i]}} for i in range(nb_tasks)
        }
        nb_ressources = r2t.shape[0]
        resources_list = [f"R{i}" for i in range(nb_ressources)]
        resources = {resources_list[i]: int(rc[i]) for i in range(nb_ressources)}

        for t in range(nb_tasks):
            for p in range(nb_tasks):
                if t2t[t][p] > 0:
                    successors[tasks_list[p]] += [tasks_list[t]]
                    # model.AddNoOverlap([intervals[t], intervals[p]])  # redundant
        for t in range(nb_tasks):
            for r in range(nb_ressources):
                mode_details[tasks_list[t]][1][resources_list[r]] = int(r2t[r][t])
        model = RCPSPModel(
            resources=resources,
            non_renewable_resources=[],
            mode_details=mode_details,
            successors=successors,
            horizon=500,
            tasks_list=tasks_list,
            source_task=0,
            sink_task=nb_tasks - 1,
        )
        dummy_solution = model.get_dummy_solution()
        print("makespan dummy ", dummy_solution.get_end_time(model.sink_task))
        return model, dummy_solution

    data_loader = DataLoader(data_list, batch_size=32, shuffle=False)
    bench_id = 0

    for batch_idx, data_batch in enumerate(data_loader):

        data_batch.to(device)

        for data in data_batch.to_data_list():

            print("Solving bench {}/{} ...".format(bench_id, len(data_list) - 1))

            t2t, dur, r2t, rc, con, ref_makespan = (
                data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
                data.dur.data.cpu().detach().numpy(),
                data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
                data.rc.data.cpu().detach().numpy(),
                data.con.view(*data.con_shape).data.cpu().detach().numpy(),
                data.reference_makespan,
            )
            rcpsp_model, _ = build_rcpsp_model(t2t, dur, r2t, rc)
            solver = CP_RCPSP_MZN(
                rcpsp_model=rcpsp_model, cp_solver_name=cp_solver_name
            )
            solver.init_model(output_type=True)
            params = ParametersCP.default()
            params.time_limit = time_out
            result = solver.solve(parameters_cp=params)
            with open(outfile, "a") as f_object:
                writer_object = writer(f_object)
                for x, t in zip(
                    result.list_solution_fits, result.list_computation_time
                ):
                    sol: RCPSPSolution = x[0]
                    makespan = sol.get_max_end_time()
                    writer_object.writerow(
                        [bench_id, t, max(ref_makespan), makespan]
                        + [sol.get_start_time(t) for t in rcpsp_model.tasks_list]
                    )
                f_object.close()
            bench_id += 1


def solve_with_docplex(data_list, device, outfile: str, time_out=25):
    # dirty imports...

    from docplex.cp.expression import interval_var
    from docplex.cp.model import CpoIntervalVar, CpoModel, CpoVariable
    from docplex.cp.modeler import end_before_start, end_of, minimize, pulse
    from docplex.cp.solution import CpoSolveResult
    from docplex.cp.solver.cpo_callback import (
        ALL_CALLBACK_EVENTS,
        EVENT_SOLUTION,
        CpoCallback,
    )

    data_loader = DataLoader(data_list, batch_size=32, shuffle=False)
    bench_id = 0

    class CustomCallback(CpoCallback):
        def __init__(self, vars_of_interest: List[CpoIntervalVar]):
            self.list_solutions = []
            self.list_makespan = []
            self.list_times = []
            self._vars = vars_of_interest

        def invoke(self, solver, event, sres: CpoSolveResult):
            """Notify the callback about a solver event.

            This method is called every time an event is notified by the CPO solver.
            Associated to the event, the solver information is provided as a an object of class
            class:`~docplex.cp.solution.CpoSolveResult` that is instantiated with information available at this step.

            Args:
                solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
                event:  Event id, string with value in ALL_CALLBACK_EVENTS
                sres:   Solver data, object of class :class:`~docplex.cp.solution.CpoSolveResult`
            """
            if event == EVENT_SOLUTION:
                t = perf_counter()
                starts = [sres.get_var_solution(v).get_start() for v in self._vars]
                self.list_solutions += [starts]
                self.list_makespan += [max(starts)]
                self.list_times += [t]

    for batch_idx, data_batch in enumerate(data_loader):
        data_batch.to(device)
        for data in data_batch.to_data_list():
            print("Solving bench {}/{} ...".format(bench_id, len(data_list) - 1))

            t2t, dur, r2t, rc, con, ref_makespan = (
                data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
                data.dur.data.cpu().detach().numpy(),
                data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
                data.rc.data.cpu().detach().numpy(),
                data.con.view(*data.con_shape).data.cpu().detach().numpy(),
                data.reference_makespan,
            )
            mdl = CpoModel()
            # Create task interval variables
            CAPACITIES = [int(rc[i]) for i in range(len(rc))]
            tasks = [
                interval_var(name="T{}".format(i + 1), size=int(dur[i]))
                for i in range(len(dur))
            ]
            # Add precedence constraints
            mdl.add(
                end_before_start(tasks[t], tasks[s])
                for t in range(len(dur))
                for s in range(len(dur))
                if t2t[s][t] > 0
            )
            # Constrain capacity of resources
            mdl.add(
                sum(
                    pulse(tasks[t], int(r2t[r][t]))
                    for t in range(len(dur))
                    if r2t[r][t] > 0
                )
                <= CAPACITIES[r]
                for r in range(len(rc))
            )
            # Minimize end of all tasks
            mdl.add(minimize(max_of(end_of(t) for t in tasks)))
            cll = CustomCallback(vars_of_interest=tasks)
            mdl.add_solver_callback(cll)
            start_time = perf_counter()
            result: CpoSolveResult = mdl.solve(
                execfile="/Applications/CPLEX_Studio201/cpoptimizer/bin/x86-64_osx/cpoptimizer",
                TimeLimit=time_out,
                trace_cpo=False,
                trace_log=False,
                verbose=False,
            )
            with open(outfile, "a") as f_object:
                writer_object = writer(f_object)
                for starts, makespan, time in zip(
                    cll.list_solutions, cll.list_makespan, cll.list_times
                ):
                    writer_object.writerow(
                        [bench_id, time - start_time, max(ref_makespan), makespan]
                        + [int(st) for st in starts]
                    )
                f_object.close()
            bench_id += 1


def run_cp_sat_computations():
    data_list = torch.load("data_list.tch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print(f"Run ID: {run_id}")
    solve_with_cpsat(
        data_list=data_list,
        device=device,
        outfile="cpsat_solutions_{}.csv".format(run_id),
    )


def run_cp_from_discrete_optimisation(
    cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
):
    data_list = torch.load("../torch_folder/data_list.tch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print(f"Run ID: {run_id}")
    solve_with_discreteoptim(
        data_list=data_list,
        device=device,
        outfile="{}_solutions_{}.csv".format(cp_solver_name.name, run_id),
        cp_solver_name=cp_solver_name,
        time_out=25,
    )


def run_minizinc_chuffed_computations():
    run_cp_from_discrete_optimisation(CPSolverName.CHUFFED)


def run_minizinc_cpopt_computations():
    run_cp_from_discrete_optimisation(CPSolverName.CPOPT)


def run_pure_cpopt_computations():
    data_list = torch.load("../torch_folder/data_list.tch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print(f"Run ID: {run_id}")
    solve_with_docplex(
        data_list=data_list,
        device=device,
        outfile="{}_solutions_{}.csv".format("pure_cpo", run_id),
        time_out=25,
    )


if __name__ == "__main__":
    run_pure_cpopt_computations()
