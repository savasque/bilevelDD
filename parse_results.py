import argparse
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern']})
plt.rcParams.update({'font.size': 20, "text.usetex": True})

from src.utils.utils import mkdir

def parse(paths):
    # Consolidate results
    files = [join(path, f) for path in paths for f in listdir(path) if isfile(join(path, f)) and f.split(".")[1] == "xlsx" and "summarized" not in f and "consolidated" not in f]
    df = pd.DataFrame()
    for file in files:
        if "cplex22" in file or "mibs" in file or "cplex|w25" in file:
            with open(file, "rb") as data:
                data = pd.read_excel(data)
                if "fischetti" in file:
                    data["approach"] = "FLMS"
                    data["solver"] = "cplex"
                if "mibs" in file:
                    data["approach"] = "MibS"
                    data["solver"] = "cplex"
                if "instance_id" not in data: data["instance_id"] = data["input_file"].apply(lambda x: x.split("/")[-1].split(".")[0])  # Add instance_id to B&C solver
                if "max_width" in data: data["width"] = data["max_width"]
                if "SEP" in file: data["approach"] = data["approach"] + "-{}".format(file.split("/")[-2].split("-")[-1])
                if "model_runtime" not in data: data["model_runtime"] = None
                if "num_cuts" not in data: data["num_cuts"] = None
                if "opt" not in data: data["opt"] = data["bilevel_gap"] < 1e-6
                if "n" in data.columns:
                    data["nL"] = data["n"]
                    data["nF"] = data["n"]
                    del data["n"]
                data.rename(
                    columns={
                        "final_bound": "lower_bound",
                        "zbest": "upper_bound",
                        "%final_gap": "bilevel_gap",
                        "time (s.)": "total_runtime"
                    },
                    inplace=True
                )
                df = pd.concat((df, data))
    df = df[["instance_id", "approach", "nL", "nF", "width", "max_width", "solver", "lower_bound", "upper_bound", "bilevel_gap", "opt", "total_runtime", "model_runtime", "compilation_runtime", "iters", "num_cuts", "HPR", "HPR_runtime"]]
    df.loc[(df.bilevel_gap == 100) & (df.upper_bound > 1e10), "bilevel_gap"] = np.nan
    # df.to_excel(join("results", "consolidated_results.xlsx"), index=False)

    # # Build summary
    # summary = df.copy()
    # summary.loc[summary.opt == 1, "bilevel_gap"] = np.nan
    # summary = summary.groupby(by=["nL", "nF", "p", "rhs_ratio", "approach", "solver", "width"]).agg(
    #     avg_gap = ("bilevel_gap", "mean"),
    #     solved_instances = ("opt", "sum"),
    #     avg_total_runtime = ("total_runtime", "mean"),
    #     avg_model_runtime = ("model_runtime", "mean"),
    #     avg_num_cuts = ("num_cuts", "mean")
    # )
    # summary.to_excel(join("results", "summarized_results.xlsx"))

    # Create plots
    data = {key: group[["instance_id", "opt", "bilevel_gap", "total_runtime"]].to_dict("records") for key, group in df.groupby(["approach", "solver"])}
    total_instances = 60
    filtered_data = {"{}_{}".format(key[1], key[0]): value for key, value in data.items()}
    for key in filtered_data:
        if len(filtered_data[key]) != total_instances:
            raise ValueError("Missing results")
    filtered_data = {key: [value for value in values] for key, values in filtered_data.items()}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 10), sharey=True, facecolor='w')
    plt.subplots_adjust(wspace=0, hspace=0)

    # Runtime axis
    ax.set_xlim(0, 3600)
    ax.set_ylim(0, 100)
    # ax.set_xscale("symlog")
    # ax.set_xticks([0, 900, 1800, 2700, 3600], labels=["0", "900", "1800", "2700", r"3600 / $0\%$"])
    ax.set_yticks([25, 50, 75, 100], labels=[r"$25\%$", r"$50\%$", r"$75\%$", r"$100\%$"])
    ax.grid(color="gray", linestyle="dotted", linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instances")

    # Gap axis
    max_gap = max(0 if entry["bilevel_gap"] == float("inf") else entry["bilevel_gap"] for key in filtered_data for entry in filtered_data[key])
    for key in filtered_data:
        for entry in filtered_data[key]:
            if entry["bilevel_gap"] == float("inf") or np.isnan(entry["bilevel_gap"]):
                entry["bilevel_gap"] = max_gap + 1e6
    ax2.set_xscale("symlog")
    # ax2.set_xlim(0, 10 * round((max_gap / 10) + 0.5))
    ax2.set_xlim(0, 10**5 - 100)
    ax2.set_ylim(0, 100)
    ax2_x_limit = 10 * round((max_gap / 10) + 0.5)
    ax2_x_limit = 100 if ax2_x_limit == 0 else ax2_x_limit
    ax2_x_ticks = list(range(0, ax2_x_limit, ax2_x_limit // 5))
    # ax2.set_xticks(ax2_x_ticks, labels=[""] + [r"$\leq {}\%$".format(i) for i in ax2_x_ticks[1:]])
    ax2.set_xticks([10**i for i in range(5)], labels=[r"$\leq 10^{}\%$".format(i) for i in range(5)])
    ax2.grid(color="gray", linestyle="dotted", linewidth=1)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(1.5)
    ax2.tick_params(left=False)
    ax2.set_xlabel("Gap")

    # fig.suptitle("Instances solved / Bilevel gap\n" + r"$n_L = {}, n_F = {}, p = {}\%$".format(nL, nF, p))
    
    colors = {
        "cplex_iterative": "red",
        "gurobi_iterative": "blue",
        "cplex_FLMS": "black",
        "cplex_MibS": "green"
    }
    line_style = {
        # "gurobi_iterative": "dashed",
        "cplex_iterative": "dashed",
        "cplex_FLMS": "solid",
        "cplex_MibS": "dashdot"
    }
    # markers = {
        # "DD0": "o", 
        # "DD25": "^", 
    #     "DD1000": "s", 
    #     "DD2000": "D"
    # }
    labels = {
        # "gurobi_iterative_DD0": r"\texttt{IDDR} (Gurobi, $W=0$)",
        "gurobi_iterative": r"\texttt{IDDR} (Gurobi)",
        "cplex_iterative": r"\texttt{IDDR}",
        # "gurobi_iterative_DD500": r"\texttt{IDDR} (Gurobi, $W=500$)",
        # "gurobi_iterative_DD1000": r"\texttt{IDDR} (Gurobi, $W=1000$)",
        # "gurobi_iterative_DD2000": r"\texttt{IDDR} (Gurobi, $W=2000$)",
        # "cplex_iterative_DD0": r"\texttt{IDDR} (CPLEX, $W=0$)",
        # "cplex_iterative_DD500": r"\texttt{IDDR} (CPLEX, $W=500$)",
        # "cplex_iterative_DD1000": r"\texttt{IDDR} (CPLEX, $W=1000$)",
        # "cplex_iterative_DD2000": r"\texttt{IDDR} (CPLEX, $W=2000$)",
        "cplex_FLMS": r"\texttt{FLMS}",
        "cplex_MibS": r"\texttt{MibS}"
    }
    for idx, key in enumerate(filtered_data.keys()):
        # Solved instances (first axis)
        x1 = sorted([value["total_runtime"] for value in filtered_data[key] if value["opt"] == 1] + [value["total_runtime"] for value in filtered_data[key] if value["opt"] == 1])
        y1 = sorted([(i + 1) * 100 / total_instances for i in range(len(x1) // 2)] + [i * 100 / total_instances for i in range(len(x1) // 2)])
        if y1:
            y1[0] = -1
        if y1:
            x1.append(1e6)
            y1.append(y1[-1])
        if y1:
            ax.plot(
                x1, 
                y1, 
                label=labels[key], 
                linewidth=2, 
                # marker=markers[key.split("_")[-1]], 
                color=colors[key], 
                linestyle=line_style[key]
            )

        # Unsolved instances (second axis)
        x2 = sorted([value["bilevel_gap"] for value in filtered_data[key] if value["opt"] == 0] + [value["bilevel_gap"] for value in filtered_data[key] if value["opt"] == 0])
        y2 = sorted([(i + 1) * 100 / total_instances for i in range(len(x2) // 2)] + [i * 100 / total_instances for i in range(len(x2) // 2)])
        # If there were solved instances, connect lines
        if y1:
            y2 = [y1[-1] + i for i in y2]
            if y2:
                x2 = [-1000] + x2
                y2 = [y1[-1]] + y2
        elif y2:
            # Remove initial marker
            y2 = [-1] + y2
            x2 = [0] + x2
        # If there are instances without gap, finish line at the end of the axis
        if 0 < len(filtered_data[key]) < 30:
            if y2:
                x2.append(ax2_x_limit + 1000)
                y2.append(y2[-1])
            elif y1:
                x2 = [0, ax2_x_limit + 1000]
                y2 = [y1[-1], y1[-1]]
        if y1 or y2:
            ax2.plot(
                x2, 
                y2, 
                label=labels[key], 
                linewidth=2, 
                # marker=markers[key.split("_")[-1]], 
                color=colors[key],
                linestyle=line_style[key]
            )
            ax2.legend()
    
    handles, labels = plt.gca().get_legend_handles_labels()
    lable_order = [2, 0, 1]

    # pass handle & labels lists along with order as below
    plt.legend([handles[i] for i in lable_order], [labels[i] for i in lable_order])

    # plt.show()
    mkdir("results/img", override=False)
    plt.savefig("results/img/bobilib_perf_profiles.png".format(1), dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    args = parser.parse_args()

    if not args.path: args.path = [
            # "results/server/mibs-results",
            # "results/server/fischetti-results",
            "results/server/miplib/results-final2"
            # "/Users/savasquez/research/bilevel-singlefollower/bisp-side-constr/results/server/iterative-results"
            # "results-server/results"
        ]

    parse(args.path)