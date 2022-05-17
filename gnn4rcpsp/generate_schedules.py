import datetime
from time import perf_counter
from ortools.sat.python import cp_model
from torch_geometric.data import DataLoader
import torch
from csv import writer

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
        with open(self._outfile, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([
                self._bench_id,
                perf_counter() - self._start_time,
                max(self._ref_makespan),
                self.Value(self._makespan)
            ] +
            [self.Value(s) for s in self._starts])
            f_object.close()

def solve_with_cpsat(data_list, device, outfile):
    
    data_loader = DataLoader(data_list, batch_size=32, shuffle=False)
    bench_id = 0
    
    for batch_idx, data_batch in enumerate(data_loader):
        
        data_batch.to(device)
        
        for data in data_batch.to_data_list():
            
            print('Solving bench {}/{} ...'.format(
                bench_id,
                len(data_list) - 1
            ))
            
            t2t, dur, r2t, rc, con, ref_makespan = (
                data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
                data.dur.data.cpu().detach().numpy(),
                data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
                data.rc.data.cpu().detach().numpy(),
                data.con.view(*data.con_shape).data.cpu().detach().numpy(),
                data.reference_makespan
            )
            
            model = cp_model.CpModel()
            horizon = int(max(dur) * len(dur))
            starts = [model.NewIntVar(0, horizon - 1, 'start_task[{}]'.format(i)) for i in range(len(dur))]
            ends = [model.NewIntVar(0, horizon - 1, 'start_task[{}]'.format(i)) for i in range(len(dur))]
            durs = [model.NewConstant(int(dur[i])) for i in range(len(dur))]
            intervals = [model.NewIntervalVar(starts[i], durs[i], ends[i], 'interval_task[{}]'.format(i)) for i in range(len(dur))]
            makespan = model.NewIntVar(0, horizon - 1, 'makespan')
            
            # for t in range(t2t.shape[0]):
            #     model.Add(starts[t] + durs[t] == ends[t])
            
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
            model.Minimize(makespan)
            
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = CPSAT_TIME_LIMIT
            start_time = perf_counter()
            status = solver.Solve(model, CPSatSolutionPrinter(bench_id, start_time, starts, ref_makespan, makespan, outfile))
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print('Found optimal solution for bench {}/{} in {} s.'.format(
                    bench_id,
                    len(data_list) - 1,
                    perf_counter() - start_time))
            elif status == cp_model.INFEASIBLE:
                print('Could not find a feasible solution for bench {}/{}'.format(
                    bench_id,
                    len(data_list) - 1
                ))
            elif status == cp_model.MODEL_INVALID:
                raise RuntimeError('Invalid CPSAT model for bench {}/{}'.format(
                    bench_id,
                    len(data_list) - 1
                ))
            
            bench_id += 1

if __name__ == "__main__":
    data_list = torch.load('data_list.tch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_id = timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    print(f"Run ID: {run_id}")
    solve_with_cpsat(data_list=data_list, device=device, outfile='cpsat_solutions_{}.csv'.format(run_id))