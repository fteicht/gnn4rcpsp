import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def basic_stats(array_):
    print("Mean ", np.mean(array_))
    print("Median ", np.median(array_))
    pp = [1, 5, 10, 25, 50, 75, 90, 95, 100]
    p = np.percentile(array_, pp)
    print("Percentiles ", pp, p)


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


def analyse(tag_images=""):
    # sres = json.load(open("results/res_2022-11-09 08:58:58.100821.json", "r"))
    # res = json.load(open("res_with_cp/results_20221110065845.json", "r"))
    # res = json.load(open("../hindsight_vs_reactive_20221114171053.json", 'r'))
    # res = json.load(open("../hindsight_vs_reactive_20221117001231.json", 'r'))
    # res_2 = json.load(open("../hindsight_vs_reactive_20221117103208.json", 'r'))
    # res_3 = json.load(open("../hindsight_vs_reactive_20221117194608.json", 'r'))

    # Small statistics
    # folder_with_results = "/Users/poveda_g/Downloads/execute_schedules_20221118111557/"
    # json_files = [
    #     os.path.join(folder_with_results, f)
    #     for f in os.listdir(folder_with_results)
    #     if "json" in f
    # ]
    # res = {}
    # from functools import reduce
    #
    # reduce(lambda x, y: res.update(json.load(open(y, "r"))), json_files)
    # names_scenarios = set([int(x[10:]) for x in res])
    # for j in filtered_test_list:
    #     if j not in names_scenarios:
    #         print(j, " not here")

    res = json.load(
        open(
            "../bigstatisitics_hindsight_vs_reactive_stats_30_10_20221118131949.json",
            "r",
        )
    )

    flat_scenarios = [
        (fk, sc)
        for fk in res
        for sc in res[fk]
        if all((res[fk][sc][algo]["executed"] != "Fail" for algo in res[fk][sc]))
    ]
    print(len(flat_scenarios), " scenarios ")
    algorithms_name = list(res[flat_scenarios[0][0]][flat_scenarios[0][1]].keys())
    best_result_per_flat_scenario = {
        f: min([res[f[0]][f[1]][a]["executed"] for a in algorithms_name])
        for f in flat_scenarios
    }
    for f in flat_scenarios:
        for a in algorithms_name:
            res[f[0]][f[1]][a]["relative-overcost-to-best"] = (
                (res[f[0]][f[1]][a]["executed"] - best_result_per_flat_scenario[f])
                / best_result_per_flat_scenario[f]
                * 100
            )

    res_per_algo = {
        a: [
            {
                kk: res[f[0]][f[1]][a][kk]
                for kk in ["executed", "timing", "relative-overcost-to-best"]
            }
            for f in flat_scenarios
        ]
        for a in algorithms_name
    }

    # fig, ax = plt.subplots(1)
    # for key in res_per_algo:
    #     sns.histplot(res_per_algo[key], label=f"algo {key}", kde=False, alpha=0.5, ax=ax)
    # ax.legend()
    import pandas as pd

    list_records = [
        {
            "algo": a,
            "makespan": v["executed"],
            "rel-overcost": v["relative-overcost-to-best"],
            "timing": v["timing"],
        }
        for a in res_per_algo
        for v in res_per_algo[a]
    ]

    df = pd.DataFrame.from_records(list_records)
    makespan_algos = {
        a: np.array([v["executed"] for v in res_per_algo[a]]) for a in res_per_algo
    }
    for a in makespan_algos:
        print(f"algo={a}")
        basic_stats(makespan_algos[a])
    keys_algo = sorted(makespan_algos.keys())
    merged_makespan = np.array([makespan_algos[a] for a in keys_algo])
    best_algos = np.argsort(merged_makespan, axis=0)

    from collections import Counter

    print(keys_algo)
    rankings = {i: Counter(list(best_algos[i, :])) for i in range(best_algos.shape[0])}

    ranking_algo = sorted(
        [j for j in range(len(keys_algo))],
        key=lambda x: rankings[0].get(x, 0),
        reverse=True,
    )
    for rank_algo in range(len(ranking_algo)):
        j = ranking_algo[rank_algo]
        print(
            f"{rank_algo+1}) {keys_algo[j]} ",
            " ".join([f"n°{i+1}={rankings[i].get(j, 0)}" for i in sorted(rankings)]),
        )

    def plot_pareto_front():
        pass

    fig, ax = plt.subplots(1, figsize=(10, 10))  # figsize=(15, 10))
    i = 0
    sns.boxplot(df, x="makespan", y="algo", ax=ax)
    ax.legend()
    ax.set_title("Makespan distribution per algorithm")
    ax.set_ylabel("Algorithm")
    ax.set_xlabel("Makespan")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_makespan_per_algo.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, figsize=(10, 10))  # figsize=(15, 10))
    sns.boxplot(df, x="rel-overcost", y="algo", ax=ax)
    ax.legend()
    ax.set_title("Relative overcost to best algorithm")
    ax.set_ylabel("Algorithm")
    ax.set_xlabel("relative overcost compared to best algorithm, (%)")
    ax.set_xscale("log")
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_rel_makespan_per_algo.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.boxplot(df, x="timing", y="algo", ax=ax)
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.legend()
    ax.set_title("Computation time distribution per algorithm (s)")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_computation_time_per_algo.png", bbox_inches="tight")
    plt.show()

    # fig, ax = plt.subplots(1)
    # ax.set_title("Makespan (objective function)")
    # sns.histplot(df, x="algo", y="makespan", ax=ax, kde=True)

    # fig, ax = plt.subplots(1)
    # ax.set_title("Computation time")
    # sns.histplot(df, x="algo", y="timing", ax=ax, kde=True)
    # plt.show()


if __name__ == "__main__":
    analyse(tag_images="bigstatistics")
