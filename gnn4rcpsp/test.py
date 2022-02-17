from skdecide.discrete_optimization.generic_tools.cp_tools import ParametersCP
from skdecide.hub.solver.do_solver.do_solver_scheduling import (
    BasePolicyMethod,
    DOSolver,
    PolicyMethodParams,
    SolvingMethod,
)

from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain

import time
import os

test_domain: RCPSP = load_domain(os.path.abspath('../kobe-rcpsp/data/rcpsp/j120.sm/j12052_1.sm'))
test_domain.set_inplace_environment(False)

state = test_domain.get_initial_state()

base_policy_method = BasePolicyMethod.FOLLOW_GANTT

policy_method_params = PolicyMethodParams(
    base_policy_method=base_policy_method, delta_index_freedom=0, delta_time_freedom=0
)

# This time, we want to focus on LNS_CP
method = SolvingMethod.LNS_CP

dict_params = {}

# We will use the default TimeLimit value for the internal CP solver
# This value is 30 seconds
params_cp = ParametersCP.default()
params_cp.TimeLimit = 30

# To be consistent with the previous solve we will cap the overall
# time to the same 300 seconds limit
dict_params["parameters_cp"] = params_cp
dict_params["max_time_seconds"] = 300
dict_params[
    "nb_iteration_lns"
] = 10000  # dummy value, we want that the max_time_seconds to be the limiting param
dict_params["verbose"] = False

# Let's create our solver with the given feature
# This step is mostly similar to the previous one
solver = DOSolver(
    policy_method_params=policy_method_params, method=method, dict_params=dict_params
)

tic = time.perf_counter()
solver.solve(domain_factory=lambda: test_domain)
toc = time.perf_counter()
lns_cp_solution = solver.best_solution.copy()
makespan = solver.do_domain.evaluate(solver.best_solution)["makespan"]
print(f"LNS-CP achieved a {makespan} makespan in: {toc - tic:0.4f} seconds")
