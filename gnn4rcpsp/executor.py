from copy import deepcopy
from email.policy import default
import os
from typing import Tuple
from gym import make
import torch
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model
from models import ResTransformer
from skdecide.discrete_optimization.rcpsp.parser.rcpsp_parser import parse_file
from skdecide.discrete_optimization.rcpsp.rcpsp_model import (
    UncertainRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    MethodBaseRobustification,
    MethodRobustification,
    create_poisson_laws
)
from graph import Graph

CPSAT_TIME_LIMIT = 60  # 1 mn

class SchedulingExecutor :
    
    def __init__(self, rcpsp: RCPSPModel, model: torch.nn.Module, device: torch.device, samples: int = 100) -> None:
        self._rcpsp = rcpsp
        self._model = model
        self._device = device
        self._poisson_laws = create_poisson_laws(
            base_rcpsp_model=self._rcpsp,
            range_around_mean_resource=1,
            range_around_mean_duration=3,
            do_uncertain_resource=True,
            do_uncertain_duration=True
        )
        self._uncertain_rcpsp = UncertainRCPSPModel(
            base_rcpsp_model=self._rcpsp,
            poisson_laws={task: laws for task, laws in self._poisson_laws.items()
                          if task in self._rcpsp.mode_details},
            uniform_law=True
        )
        self._simulated_rcpsp = None
        self._executed_schedule = None
        self._current_time = None
        self._samples = samples
        self._rcpsp_predecessors = defaultdict(lambda : set())
        for task, successors in self._rcpsp.successors.items():
            for succ in successors:
                self._rcpsp_predecessors[succ].add(task)
        
    def reset(self) -> Tuple[int, RCPSPSolution]:
        '''Samples a new RCPSP (uncertain task durations and resource capacities)
        and returns an empty schedule and current time set to 0'''
        self._simulated_rcpsp = self._uncertain_rcpsp.create_rcpsp_model(MethodRobustification(MethodBaseRobustification.SAMPLE))
        self._executed_schedule = RCPSPSolution(
            problem=deepcopy(self._rcpsp),
            rcpsp_permutation=[],
            standardised_permutation=[],
            rcpsp_schedule={})
        self._current_time = 0
        return self._executed_schedule, self._current_time
    
    def progress(self, next_task: int) -> Tuple[int, RCPSPSolution, bool]:
        '''Takes the next task to execute and returns the new current time and updated executed schedule
        and a boolean indicating whether all the tasks have been scheduled or not'''
        
        # Add next_task to the executed schedule
        self._executed_schedule.rcpsp_schedule.update({
            next_task: {
                'start_time': self._current_time,
                'end_time': self._current_time + max(
                    mode['duration'] for mode in self._simulated_rcpsp.mode_details[next_task].values())
            }
        })
        
        # Compute the next event, i.e. smallest task starting or ending date
        event_times = set()
        for dates in self._executed_schedule.rcpsp_schedule.values():
            if dates['end_time'] > self._current_time:
                event_times.add(dates['end_time'])
        self._current_time = min(event_times) if len(event_times) > 0 else current_time
                
        # Update the mode details of the ended tasks
        for task in self._executed_schedule.problem.mode_details.keys():
            if task in self._executed_schedule.rcpsp_schedule and \
                self._executed_schedule.rcpsp_schedule[task]['end_time'] <= self._current_time:
                    self._executed_schedule.problem.mode_details[task] = self._simulated_rcpsp.mode_details[task]
        
        # Return the new current time and updated executed schedule
        return (
            self._current_time,
            self._executed_schedule,
            len(self._executed_schedule.rcpsp_schedule) == len(self._executed_schedule.problem.mode_details)
        )
    
    def next_task(self, executed_schedule: RCPSPSolution, current_time: int) -> Tuple[int, RCPSPSolution]:
        '''Takes a schedule which has been executed (i.e. list of committed start dates for some tasks),
        the current time, and returns the next task's starting date and the full updated schedule'''
        
        executed_rcpsp = executed_schedule.problem
        
        if len(executed_schedule.rcpsp_schedule) == 0:
            tasks = set()
            for task in executed_schedule.rcpsp_schedule:
                tasks = tasks | set(executed_rcpsp.successors[task])
            initial_task = set(executed_rcpsp.mode_details.keys()) - tasks
            initial_task = next(iter(initial_task))
            return initial_task, RCPSPSolution(
                problem=executed_rcpsp,
                rcpsp_schedule={
                    initial_task: {
                        'start_time': current_time,
                        'end_time': current_time + max(mode['duration'] for mode in executed_rcpsp.mode_details[initial_task].values())
                    }
                },
                # disable permutation computation
                rcpsp_permutation=[],
                standardised_permutation=[]
            )
        
        running_tasks = set(task for task, sched in executed_schedule.rcpsp_schedule.items()
                         if sched['end_time'] > current_time)
        front_tasks = set()
        for task in executed_schedule.rcpsp_schedule:
            front_tasks = front_tasks | set(executed_rcpsp.successors[task])
        initial_task = set(executed_rcpsp.mode_details.keys()) - front_tasks
        initial_task = next(iter(initial_task))
        front_tasks = front_tasks - set(executed_schedule.rcpsp_schedule.keys())
        front_tasks = front_tasks | running_tasks
        
        # Compute currently consumed resources
        consumed_resources = defaultdict(lambda: 0.0)
        for res in executed_rcpsp.resources.keys():
            consumed_resources[res] = 0
            for task in running_tasks:
                resource_all_modes = []
                for mode in executed_rcpsp.mode_details[task].values():
                    resource_all_modes.append(mode[res])
                consumed_resources[res] += max(resource_all_modes)
        
        # Extract the front tasks that can be started now
        candidate_tasks = []
        for task in front_tasks:
            if (task not in running_tasks and
                self._rcpsp_predecessors[task].issubset(set(executed_schedule.rcpsp_schedule.keys()) - running_tasks)):
                task_can_start = True
                for res, cap in executed_rcpsp.resources.items():
                    resource_all_modes = []
                    for mode in executed_rcpsp.mode_details[task].values():
                        resource_all_modes.append(mode[res])
                    task_can_start = task_can_start and (consumed_resources[res] + max(resource_all_modes) <= cap)
                if task_can_start:
                    candidate_tasks.append(task)
                    
        # Evaluate each candidate task to start
        scenarios = defaultdict(lambda : [])
        for cantask in candidate_tasks:
            print('Testing {}'.format(cantask))
            
            # Add cantask to the executed schedule
            cantask_schedule = deepcopy(executed_schedule.rcpsp_schedule)
            cantask_schedule.update({cantask: {'start_time': current_time,
                                                'end_time': current_time + max(mode['duration'] for mode in executed_rcpsp.mode_details[cantask].values())}})
            
            # Create "remaining" RCPSP: keep all the non executed tasks plus the running ones
            mode_details = {task: modes for task, modes in executed_rcpsp.mode_details.items()
                            if task != cantask and task not in executed_schedule.rcpsp_schedule}
            successors = {task : succ for task, succ in executed_rcpsp.successors.items()
                          if task != cantask and task not in executed_schedule.rcpsp_schedule}
            name_task = {task: name for task, name in executed_rcpsp.name_task.items()
                         if task != cantask and task not in executed_schedule.rcpsp_schedule}
            
            running_tasks_modes = {task: deepcopy(modes) for task, modes in executed_rcpsp.mode_details.items()
                                   if task in running_tasks or task == cantask}
            for task, modes in running_tasks_modes.items():
                for mode in modes.values():
                    mode['duration'] = cantask_schedule[task]['end_time'] - current_time
            mode_details.update(running_tasks_modes)
            mode_details.update({initial_task: executed_rcpsp.mode_details[initial_task]})
            successors.update({initial_task: list(front_tasks)})
            successors.update({task: executed_rcpsp.successors[task] for task in executed_rcpsp.mode_details
                               if task in running_tasks or task == cantask})
            name_task.update({initial_task: executed_rcpsp.name_task[initial_task]})
            name_task.update({task: executed_rcpsp.name_task[task] for task in executed_rcpsp.mode_details
                              if task in running_tasks or task == cantask})
            rcpsp = RCPSPModel(
                resources=executed_rcpsp.resources,
                non_renewable_resources=executed_rcpsp.non_renewable_resources,
                mode_details=mode_details,
                successors=successors,
                horizon=executed_rcpsp.horizon - max(dates['end_time'] for dates in cantask_schedule.values()),
                horizon_multiplier=executed_rcpsp.horizon_multiplier,
                name_task=name_task
            )
            
            # Check if cantask leads to a feasible schedule with the current world assumptions
            # Compute the median RCPSP (the average or worst case ones appear to be barely feasible in practice)
            # worst_case_rcpsp = best_uncertain_rcpsp.create_rcpsp_model(MethodRobustification(MethodBaseRobustification.WORST_CASE))
            data = Graph().create_from_data(rcpsp)
            data.to(self._device)
            out = self._model(data)
            data.out = out
            # TODO: is there a cleaner way to do this?
            # data._slice_dict['out'] = data._slice_dict['x']
            # data._inc_dict['out'] = data._inc_dict['x']
            
            t2t, dur, r2t, rc = (
                data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
                data.dur.data.cpu().detach().numpy(),
                data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
                data.rc.data.cpu().detach().numpy()
            )
            xorig = np.around(data.out[len(rc):,0].cpu().detach().numpy(), decimals=0).astype(int)
            cantask_idx = min(i for i, t in enumerate(rcpsp.mode_details.keys()) if t == cantask)
                
            model = cp_model.CpModel()
            horizon = int(max(xorig + dur) * 1.2)
            time_zero = model.NewConstant(0)
            starts = [model.NewIntVar(0, horizon - 1, 'start_task[{}]'.format(i)) for i in range(len(dur))]
            ends = [model.NewIntVar(0, horizon - 1, 'start_task[{}]'.format(i)) for i in range(len(dur))]
            durs = [model.NewConstant(int(dur[i])) for i in range(len(dur))]
            intervals = [model.NewIntervalVar(starts[i], durs[i], ends[i], 'interval_task[{}]'.format(i)) for i in range(len(dur))]
            makespan = model.NewIntVar(0, horizon - 1, 'makespan')
            
            for t in range(t2t.shape[0]):
                for p in range(t2t.shape[1]):
                    if t2t[t][p] > 0:
                        model.Add(ends[p] <= starts[t])
                        # model.AddNoOverlap([intervals[t], intervals[p]])  # redundant
            
            for r in range(r2t.shape[0]):
                model.AddCumulative([intervals[t] for t in range(len(dur)) if r2t[r][t] > 0],
                                    [int(r2t[r][t]) for t in range(len(dur)) if r2t[r][t] > 0],
                                    int(rc[r]))
                
            model.AddMaxEquality(makespan, ends)
            model.Add(starts[cantask_idx] == time_zero)
            
            # Search for a feasible solution
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = CPSAT_TIME_LIMIT
            xorig_list = xorig.tolist()
            for i, x in enumerate(xorig_list):
                model.AddHint(starts[i], x)
            # model._CpModel__model.solution_hint.vars.extend(list(range(len(dur))))
            # model._CpModel__model.solution_hint.values.extend(xorig.tolist())
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print('feasible')
                feasible_solution = [solver.Value(s) for s in starts]
            elif status == cp_model.INFEASIBLE:
                continue
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError('Invalid CPSAT model.')
        
            # Make the "remaining" RCPSP uncertain
            uncertain_rcpsp = UncertainRCPSPModel(
                base_rcpsp_model=rcpsp,
                poisson_laws={task: laws for task, laws in self._poisson_laws.items()
                            if task in rcpsp.mode_details},
                uniform_law=True
            )
            
            # Sample RCPSPs and choose the best next task to start on average
            for _ in range(self._samples):
                sampled_rcpsp = uncertain_rcpsp.create_rcpsp_model(MethodRobustification(MethodBaseRobustification.SAMPLE))
                data = Graph().create_from_data(sampled_rcpsp)
                data.to(self._device)
                out = self._model(data)
                data.out = out
                # TODO: is there a cleaner way to do this?
                # data._slice_dict['out'] = data._slice_dict['x']
                # data._inc_dict['out'] = data._inc_dict['x']
                starts = np.around(data.out[len(rcpsp.resources):,0].cpu().detach().numpy(), decimals=0).astype(int)
                dur = np.array([max(mode["duration"] for mode in task_modes.values())
                                for task_modes in sampled_rcpsp.mode_details.values()], dtype=np.int32)
                makespan = max(starts + dur)
                scenarios[cantask].append(makespan)
            scenarios[cantask] = (np.mean(scenarios[cantask]),
                                  RCPSPSolution(
                                    problem=executed_rcpsp,
                                    rcpsp_schedule={
                                        task : feasible_solution[i] + current_time
                                        for i, task in enumerate(rcpsp.mode_details.keys())
                                    },
                                    # disable permutation computation
                                    rcpsp_permutation=[],
                                    standardised_permutation=[])
                                  )
        
        # Select the best task in term of average makespan
        best_task = None
        best_makespan = float("inf")
        best_median_schedule = None
        for task, makespan_and_schedule in scenarios.items():
            makespan = makespan_and_schedule[0]
            if makespan < best_makespan:
                best_makespan = makespan
                best_task = task
                best_median_schedule = makespan_and_schedule[1]
        
        # Return the best next task and the median RCPSP's schedule
        return best_task, best_median_schedule
        

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(__file__))
    kobe_rcpsp_dir = os.path.join(root_dir, 'kobe-rcpsp/data/rcpsp')
    rcpsp = parse_file(os.path.join(kobe_rcpsp_dir, 'j30.sm/j301_1.sm'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = ResTransformer
    # Net = ResGINE
    model = Net().to(device)
    model.load_state_dict(torch.load(os.path.join(root_dir, 'torch/model.tch'), map_location=device))
    executor = SchedulingExecutor(rcpsp=rcpsp,
                                  model=model,
                                  device=device,
                                  samples=10)
    
    executed_schedule, current_time = executor.reset()
    stop = False
    
    while not stop:
        next_task, most_probable_schedule = executor.next_task(executed_schedule, current_time)
        print('Best task at time {}: {}'.format(current_time, next_task))
        current_time, executed_schedule, stop = executor.progress(next_task)