import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def boxplot_2d(x, y, ax, whis=1.5,
               color="r",
               alpha=1,
               label=""):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]
    ## the box
    box = Rectangle(
        (xlimits[0], ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec='k',
        color=color,
        zorder=0, alpha=alpha,
        label=label
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        alpha=alpha,
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        alpha=alpha,
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]], [ylimits[1]], color="k", marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        alpha=alpha,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0], ylimits[2]],
        color='k',
        alpha=alpha,
        zorder=1
    )
    ax.add_line(whisker_bar)

    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1], ylimits[1]],
        color='k',
        alpha=alpha,
        zorder=1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        alpha=alpha,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]],
        color = 'k',
        alpha=alpha,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom],
        color = 'k',
        alpha=alpha,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]],
        color = 'k',
        alpha=alpha,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top],
        color = 'k',
        alpha=alpha,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        alpha=alpha,
        facecolors='none', edgecolors='k'
    )


def basic_stats(array_):
    print("Mean ", np.mean(array_))
    print("Median ", np.median(array_))
    print("std", np.std(array_))
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


def analyse(tag_images="", put_title_and_ax=False):
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
    algorithms_to_keep = algorithms_name
    algorithms_to_keep = ["SGS-HINDSIGHT_DBP",
                          "SGS-REACTIVE_AVG",
                          "SGS-REACTIVE_WORST",
                          "SGS-REACTIVE_BEST",
                          "CPSAT-HINDSIGHT_DBP",
                          "CPSAT-REACTIVE_AVG"]
    renamed_algo = ["SERENE (SGS-HINDSIGHT_DBP)",
                    "SIREN-REACTIVE_AVG",
                    "SIREN-REACTIVE_WORST",
                    "SIREN-REACTIVE_BEST",
                    "CPSAT-HINDSIGHT_DBP",
                    "CPSAT-REACTIVE_AVG"]
    renamed_algo_dict = {algorithms_to_keep[i]: renamed_algo[i] for i in range(len(renamed_algo))}
    res_per_algo = {
        renamed_algo_dict[a]: [
            {
                kk: res[f[0]][f[1]][a][kk]
                for kk in ["executed", "timing", "relative-overcost-to-best"]
            }
            for f in flat_scenarios
        ]
        for a in algorithms_to_keep
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


    fig, ax = plt.subplots(1)  # figsize=(15, 10))
    i = 0
    sns.boxplot(df, x="makespan", y="algo", ax=ax)
    ax.legend()
    if put_title_and_ax:
        ax.set_title("Makespan distribution per algorithm")
        ax.set_ylabel("Algorithm")
        ax.set_xlabel("Makespan")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_makespan_per_algo.png", bbox_inches="tight",dpi=1000)

    fig, ax = plt.subplots(1)  # figsize=(15, 10))
    sns.boxplot(df, x="rel-overcost", y="algo", ax=ax)
    ax.legend()
    if put_title_and_ax:
        ax.set_title("Relative overcost to best algorithm")
        ax.set_ylabel("Algorithm")
        ax.set_xlabel("relative overcost compared to best algorithm, (%)")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_xscale("log")
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_rel_makespan_per_algo.png", bbox_inches="tight",dpi=1000)

    fig, ax = plt.subplots(1)
    sns.boxplot(df, x="timing", y="algo", ax=ax)
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.legend()
    if put_title_and_ax:
        ax.set_title("Computation time distribution per algorithm (s)")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_computation_time_per_algo.png", bbox_inches="tight", dpi=1000)
    plt.show()

    # fig, ax = plt.subplots(1)
    # ax.set_title("Makespan (objective function)")
    # sns.histplot(df, x="algo", y="makespan", ax=ax, kde=True)

    # fig, ax = plt.subplots(1)
    # ax.set_title("Computation time")
    # sns.histplot(df, x="algo", y="timing", ax=ax, kde=True)
    # plt.show()


def analyse_with_foresight(tag_images=""):

    res = json.load(
        open(
            "../bigstatisitics_hindsight_vs_reactive_stats_30_10_20221118131949.json",
            "r",
        )
    )
    res_foresight = json.load(open('foresight_res.json', "r"))
    for bench in res_foresight:
        for scenar in res_foresight[bench]:
            res[bench][scenar].update(res_foresight[bench][scenar])
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
            if "expectations" in res[f[0]][f[1]][a]:
                res[f[0]][f[1]][a]["mean_reaction_timing"] = res[f[0]][f[1]][a]["timing"]/\
                                                             len(res[f[0]][f[1]][a]["expectations"])
            else:
                res[f[0]][f[1]][a]["mean_reaction_timing"] = 1 # we don't care.

    algorithms_to_keep = algorithms_name
    algorithms_to_keep = ["SGS-HINDSIGHT_DBP",
                          "SGS-REACTIVE_AVG",
                          "SGS-REACTIVE_WORST",
                          "SGS-REACTIVE_BEST",
                          "CPSAT-HINDSIGHT_DBP",
                          "CPSAT-REACTIVE_AVG"]#,
                          #"FORESIGHT"]
    renamed_algo = ["SERENE",
                    "SIREN-REACTIVE_AVG",
                    "SIREN-REACTIVE_WORST",
                    "SIREN-REACTIVE_BEST",
                    "CPSAT-HINDSIGHT_DBP",
                    "CPSAT-REACTIVE_AVG"]
                    #"FORESIGHT"]
    renamed_algo_dict = {algorithms_to_keep[i]: renamed_algo[i] for i in range(len(renamed_algo))}
    res_per_algo = {
        renamed_algo_dict[a]: [
            {
                kk: res[f[0]][f[1]][a][kk]
                for kk in ["executed",
                           "timing",
                           "relative-overcost-to-best",
                           "mean_reaction_timing"]
            }
            for f in flat_scenarios
        ]
        for a in algorithms_to_keep
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
            "mean_reaction_timing": v["mean_reaction_timing"],
        }
        for a in res_per_algo
        for v in res_per_algo[a]
    ]

    df = pd.DataFrame.from_records(list_records)
    makespan_algos = {
        a: np.array([v["executed"] for v in res_per_algo[a]]) for a in res_per_algo
    }
    rel_overcost_algos = {
        a: np.array([v["relative-overcost-to-best"] for v in res_per_algo[a]]) for a in res_per_algo
    }

    mean_computing_time_algos = {
        a: np.array([v["mean_reaction_timing"] for v in res_per_algo[a]]) for a in res_per_algo
    }
    for a in rel_overcost_algos:
        print(f"algo={a}")
        basic_stats(rel_overcost_algos[a])
        print("mean reaction timing")
        basic_stats(mean_computing_time_algos[a])
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

    fig, ax = plt.subplots(1)  # figsize=(15, 10))
    i = 0
    sns.boxplot(df, x="makespan", y="algo", ax=ax)
    #ax.legend()
    ax.set_title("Makespan distribution per algorithm")
    ax.set_ylabel("Algorithm")
    ax.set_xlabel("Makespan")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_makespan_per_algo.png", bbox_inches="tight",dpi=1000)

    fig, ax = plt.subplots(1)  # figsize=(15, 10))
    sns.boxplot(df, x="rel-overcost", y="algo", ax=ax)
    #ax.legend()
    #ax.set_title("Relative overcost to Foresight")
    #ax.set_ylabel("Algorithm")
    #ax.set_xlabel("relative overcost compared to Foresight, (%)")
    #ax.set_xscale("log")
    #ax.grid(axis="x")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_rel_makespan_per_algo.png", bbox_inches="tight",dpi=1000)

    fig, ax = plt.subplots(1)
    sns.boxplot(df, x="timing", y="algo", ax=ax)
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.set_xlabel("")
    ax.set_ylabel("")

    #ax.legend()
    #ax.set_title("Computation time distribution per algorithm (s)")
    plt.tight_layout()
    fig.savefig(f"{tag_images}_computation_time_per_algo.png", bbox_inches="tight", dpi=500)


    # kinds = ["scatter", "hist", "hex", "kde", "reg", "resid"]
    # for k in kinds:
    #     #fig, ax = plt.subplots(1)
    #     try:
    #         g = sns.jointplot(
    #             data=df,
    #             x="timing", y="rel-overcost", hue="algo",
    #             kind=k, #ax=ax
    #         )
    #         g.ax_joint.set_xscale("log")
    #         g.ax_joint.grid(axis="x")
    #         plt.savefig(f"{k}_scatterplot.png", bbox_inches="tight", dpi=1000)
    #         plt.close("all")
    #     except Exception as e:
    #         print(e)
    for k in ["timing", "mean_reaction_timing"]:
        fig, ax = plt.subplots(1)
        colors = plt.get_cmap('Paired')(np.arange(0, 1, 1/len(renamed_algo)))
        x_median = []
        y_median = []
        for algo, color in zip(renamed_algo, colors):
            x = df[df["algo"] == algo][k]
            y = df[df["algo"] == algo]["rel-overcost"]
            x_median += [np.median(x)]
            y_median += [np.median(y)]
            boxplot_2d(x, y, ax, label=algo, color=color, alpha=0.7)
        x_median = np.array(x_median)
        y_median = np.array(y_median)
        sorted_x = np.argsort(x_median)
        x_median = x_median[sorted_x]
        y_median = y_median[sorted_x]
        ax.fill_between(x_median, y_median, ax.get_ylim()[1], color="grey", alpha=0.2)
        ax.legend(loc='upper right', fontsize='x-small')
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.grid(axis="x",
                which='major',
                linestyle="-")
        ax.grid(axis="x",
                which='minor',
                linestyle="--")
        ax.set_ylabel("Relative overcost %")
        ax.set_xlabel("Computation time (s)")
        fig.savefig(f"{k}_2d_boxplot_bigstatistics.png", dpi=1000)
    plt.show()
    # fig, ax = plt.subplots(1)
    # ax.set_title("Makespan (objective function)")
    # sns.histplot(df, x="algo", y="makespan", ax=ax, kde=True)

    # fig, ax = plt.subplots(1)
    # ax.set_title("Computation time")
    # sns.histplot(df, x="algo", y="timing", ax=ax, kde=True)
    # plt.show()


if __name__ == "__main__":
    # analyse(tag_images="bigstat_notitle", put_title_and_ax=False)
    analyse_with_foresight(tag_images="bigstatistics_with_foresight")
