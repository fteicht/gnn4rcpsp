import json
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from discrete_optimization.rcpsp.rcpsp_parser import parse_file
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from tqdm import tqdm


class Graph:
    NODE_INPUT_DIM = (
        2 + 1 + 1
    )  # first 2 are onehot(resource|task), last 2 are (#resources available, task duration)
    EDGE_FEATURE_DIM = (
        4 + 1
    )  # first 4 are onehot(to_successor|to_consumer|to_predecessor|to_provider),, last is for #resources

    def __init__(self) -> None:
        self._data = None
        self._node_labels = None
        self._edge_labels = None

    def data(self):
        return self._data

    def create_from_data(
        self,
        rcpsp_model: RCPSPModel,
        reference_makespan=1,
        solution: RCPSPSolution = None,
        solution_makespan=None,
    ):
        #         print(
        #             "Resources: {},\nSuccessors: {},\n'Mode details: {}".format(
        #                 rcpsp_model.resources, rcpsp_model.successors, rcpsp_model.mode_details
        #             )
        #         )

        # Parameters useful to compute the loss
        nb_tasks, nb_resources = len(rcpsp_model.successors), len(rcpsp_model.resources)

        # We assume resource nodes to appear before task nodes in the nodes ordering
        resource_ids = {r: i for i, r in enumerate(rcpsp_model.resources)}
        task_ids = {
            t: i + len(resource_ids) for i, t in enumerate(rcpsp_model.successors)
        }

        # List resource -> task, consumption edges
        r2t = []
        for t, m in rcpsp_model.mode_details.items():
            assert len(m) == 1 and 1 in m
            for k, v in m[1].items():
                if k != "duration" and v > 0:
                    r2t.append((resource_ids[k], task_ids[t], v))

        # List task -> task, duration edges
        t2t = []
        sources = []
        sinks = []
        successors = set()
        for t, s in rcpsp_model.successors.items():
            for tt in s:
                t2t.append(
                    (
                        task_ids[t],
                        task_ids[tt],
                        rcpsp_model.mode_details[t][1]["duration"]
                        + 1e-8,  # duration must be strictly positive. TODO: alternatives?
                    )
                )
                successors.add(task_ids[tt])
            if len(s) == 0:
                sinks.append(task_ids[t])
        sinks = np.array(sinks) - nb_resources
        sources = set(task_ids.values()) - successors
        sources = np.array(list(sources)) - nb_resources

        # Creates pytorch geometric data
        edge_index = torch.tensor(
            [
                [r[0] for r in r2t] + [t[0] for t in t2t],
                [t[1] for t in r2t] + [t[1] for t in t2t],
            ],
            dtype=torch.long,
        )
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # first 2 are onehot(resource|task), last 2 are (#resources available, task duration)
        node_features = torch.tensor(
            [[1, 0, r, 0] for r in rcpsp_model.resources.values()]
            + [
                [0, 1, 0, rcpsp_model.mode_details[t][1]["duration"]]
                for t in rcpsp_model.successors
            ]
        )

        # first 4 are onehot(to_successor|to_consumer|to_predecessor|to_provider), last is for #resources
        edge_features = torch.tensor(
            [[1, 0, 0, 0, r[2]] for r in r2t]
            + [[0, 1, 0, 0, 0] for t in t2t]
            + [[0, 0, 1, 0, r[2]] for r in r2t]
            + [[0, 0, 0, 1, 0] for t in t2t]
        )

        _t2t_ = torch.zeros((nb_tasks, nb_tasks), dtype=torch.float32)
        _dur_ = torch.zeros(nb_tasks, dtype=torch.float32)
        for (
            p,
            t,
            d,
        ) in t2t:
            _t2t_[t - nb_resources][p - nb_resources] = d
            _dur_[p - nb_resources] = d

        _r2t_ = torch.zeros((nb_resources, nb_tasks), dtype=torch.float32)
        for r, t, v in r2t:
            _r2t_[r][t - nb_resources] = v

        # con = torch.stack(
        #     [
        #         torch.stack(
        #             [
        #                 torch.stack(
        #                     [
        #                         _r2t_[r][tp] if tp != t else torch.tensor(0.0)
        #                         for tp in range(_r2t_.shape[1])
        #                     ]
        #                 )
        #                 for t in range(_r2t_.shape[1])
        #             ]
        #         )
        #         for r in range(_r2t_.shape[0])
        #     ]
        # )

        # Pytorch Geometric Data
        self._data = Data(
            x=node_features.type(torch.float32),
            edge_index=edge_index,
            edge_attr=edge_features.type(torch.float32),
            t2t=_t2t_.view(-1),  # flatten to enable concatenation in minibatches
            dur=_dur_,
            r2t=_r2t_.view(-1),  # transpose to enable concatenation in minibatches
            rc=torch.tensor(list(rcpsp_model.resources.values()), dtype=torch.float32),
            sources=sources,
            sinks=sinks,
            # con=con.view(-1),
            # con_shape=con.shape,
            reference_makespan=reference_makespan,
            solution_starts=torch.tensor(
                [v["start_time"] for v in solution.rcpsp_schedule.values()],
                dtype=torch.float32,
            )
            if solution is not None
            else torch.zeros(nb_tasks, dtype=torch.float32),
            solution_makespan=solution_makespan
            if solution_makespan is not None
            else -1,
        )
        # self._data.diameter = nx.algorithms.distance_measures.diameter(to_networkx(self._data))

        self._node_labels = [
            r + "[C={}]".format(c) for r, c in rcpsp_model.resources.items()
        ] + [
            "T{}[D={}]".format(t, rcpsp_model.mode_details[t][1]["duration"])
            for t in rcpsp_model.successors
        ]
        self._edge_labels = ["C={}".format(r[2]) for r in r2t] + [
            "P={}".format(t[2]) for t in t2t
        ]
        self._edge_labels = self._edge_labels

        self._vars = (t2t, r2t, rcpsp_model.resources.values(), nb_tasks, nb_resources)

        self.rcpsp_model = rcpsp_model

        return self._data

    def plot_graph(self):
        g = to_networkx(self._data)
        pos = nx.spring_layout(g, seed=0)
        plt.figure(
            1,
            figsize=(
                max(30, len(self._node_labels) // 3),
                max(30, len(self._node_labels) // 3),
            ),
        )
        nx.draw(
            g,
            pos,
            edgelist=self._data.edge_index.numpy().T,  # needed to ensure it fits edge_color order
            labels={i: l for i, l in enumerate(self._node_labels)},
            connectionstyle="arc3,rad=0.05",
            with_labels=True,
            font_size=8,
            node_size=[2000 for _ in range(len(self._node_labels))],
            node_color=[
                "lightcoral" if l[0] == "R" else "cornflowerblue"
                for l in self._node_labels
            ],
            edge_color=[
                "lightcoral" if l[0] == "C" else "cornflowerblue"
                for l in self._edge_labels
            ]
            + ["gray"] * len(self._edge_labels),
        )
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels={
                (
                    int(self._data.edge_index[0][i]),
                    int(self._data.edge_index[1][i]),
                ): l
                for i, l in enumerate(self._edge_labels)
            },
            font_size=8,
        )
        plt.show()


def load_data(kobe_rcpsp_directory, solution_file):
    data_list = []
    rcpsp_model_list = []
    optima_list = []
    with open(solution_file, "r") as sol_file:
        solutions = json.load(sol_file)
    # kobe_rcpsp_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kobe_rcpsp/data/rcpsp'))
    for d in tqdm(os.listdir(kobe_rcpsp_directory)):
        if os.path.splitext(d)[1] == ".sm":
            for f in tqdm(os.listdir(os.path.join(kobe_rcpsp_directory, d))):
                optima = pd.read_csv(
                    os.path.join(kobe_rcpsp_directory, d + "/optimum/optimum.csv"),
                    index_col="problem",
                ).T
                bench_file = os.path.basename(f)
                if os.path.splitext(bench_file)[1] == ".sm" and bench_file in solutions:
                    optimum = optima[f]["optimum"]
                    reference_makespan = (
                        [max(map(int, filter(lambda x: x, optimum.split(".."))))]
                        if type(optimum) is str
                        else [optimum]
                    )
                    rcpsp_model = parse_file(
                        os.path.join(kobe_rcpsp_directory, d + "/" + f)
                    )
                    g = Graph()
                    g.create_from_data(
                        rcpsp_model=rcpsp_model,
                        reference_makespan=reference_makespan,
                        solution=RCPSPSolution(
                            rcpsp_model,
                            rcpsp_schedule={
                                int(t): s
                                for t, s in solutions[bench_file]["sol"][
                                    "rcpsp_schedule"
                                ].items()
                            },
                        ),
                        solution_makespan=solutions[bench_file]["fit"]["makespan"],
                    )
                    data_list.append(g._data)
                    rcpsp_model_list.append(g.rcpsp_model)
                    optima_list.append(optimum)

    print("Processed {} benchmarks".format(len(data_list)))
    torch.save(data_list, "./data_list.tch")
    torch.save(rcpsp_model_list, "./rcpsp_model_list.tch")
    torch.save(optima_list, "./optima_list.tch")

    return data_list


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(__file__))
    kobe_rcpsp_dir = os.path.join(root_dir, "kobe-rcpsp/data/rcpsp")
    solutions_dir = os.path.join(root_dir, "cp_solutions")
    load_data(
        kobe_rcpsp_directory=kobe_rcpsp_dir,
        solution_file=os.path.join(
            solutions_dir, "postpro_benchmark_merged_single_modes.json"
        ),
    )
