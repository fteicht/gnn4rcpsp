import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def basic_stats(array_):
    print("Mean ", np.mean(array_))
    print("Median ", np.median(array_))
    pp = [1, 5, 10, 25, 50, 75, 90, 95, 100]
    p = np.percentile(array_, pp)
    print("Percentiles ", pp, p)

def analyse():
    # sres = json.load(open("results/res_2022-11-09 08:58:58.100821.json", "r"))
    res = json.load(open("res_with_cp/results_20221110065845.json", "r"))
    res = json.load(open("../hindsight_vs_reactive_20221114171053.json", 'r'))
    res = json.load(open("../hindsight_vs_reactive_20221117001231.json", 'r'))
    res_2 = json.load(open("../hindsight_vs_reactive_20221117103208.json", 'r'))
    res_3 = json.load(open("../hindsight_vs_reactive_20221117194608.json", 'r'))
    res.update({x+"-2": res_2[x] for x in res_2})
    res.update({x+"-3": res_3[x] for x in res_3})


    flat_scenarios = [(fk, sc) for fk in res for sc in res[fk]]
    algorithms_name = list(res[flat_scenarios[0][0]][flat_scenarios[0][1]].keys())
    res_per_algo = {a: [{kk: res[f[0]][f[1]][a][kk] for kk in ["executed", "timing"]}
                        for f in flat_scenarios] for a in algorithms_name}
    # fig, ax = plt.subplots(1)
    # for key in res_per_algo:
    #     sns.histplot(res_per_algo[key], label=f"algo {key}", kde=False, alpha=0.5, ax=ax)
    # ax.legend()
    import pandas as pd
    list_records = [{"algo": a, "makespan": v["executed"],
                     "timing": v["timing"]}
                    for a in res_per_algo
                    for v in res_per_algo[a]]
    df = pd.DataFrame.from_records(list_records)
    makespan_algos = {a: np.array([v["executed"] for v in res_per_algo[a]]) for a in res_per_algo}
    for a in makespan_algos:
        print(f"algo={a}")
        basic_stats(makespan_algos[a])
    keys_algo = sorted(makespan_algos.keys())
    merged_makespan = np.array([makespan_algos[a] for a in keys_algo])
    best_algos = np.argsort(merged_makespan, axis=0)

    from collections import Counter
    print(keys_algo)
    rankings = {i: Counter(list(best_algos[i, :])) for i in range(best_algos.shape[0])}

    ranking_algo = sorted([j for j in range(len(keys_algo))
                           ], key=lambda x: rankings[0].get(x, 0), reverse=True)
    for rank_algo in range(len(ranking_algo)):
        j = ranking_algo[rank_algo]
        print(f"{rank_algo+1}) {keys_algo[j]} ", ' '.join([f"n°{i+1}={rankings[i].get(j, 0)}"
                                                           for i in sorted(rankings)]))

    def plot_pareto_front():
        pass

    fig, ax = plt.subplots(1, figsize=(10, 10))  # figsize=(15, 10))
    i = 0
    sns.boxplot(df, x="makespan", y="algo", ax=ax)
    ax.legend()
    plt.tight_layout()
    ax.set_title("Makespan distribution per algorithm")
    fig.savefig("makespan_per_algo_2.png", bbox_inches='tight')

    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.boxplot(df, x="timing", y="algo", ax=ax)
    ax.legend()
    ax.set_title("Computation time distribution per algorithm (s)")
    plt.tight_layout()
    fig.savefig("computation_time_per_algo_2.png", bbox_inches='tight')
    plt.show()

    # fig, ax = plt.subplots(1)
    # ax.set_title("Makespan (objective function)")
    # sns.histplot(df, x="algo", y="makespan", ax=ax, kde=True)

    # fig, ax = plt.subplots(1)
    # ax.set_title("Computation time")
    # sns.histplot(df, x="algo", y="timing", ax=ax, kde=True)
    # plt.show()


if __name__ == "__main__":
    analyse()