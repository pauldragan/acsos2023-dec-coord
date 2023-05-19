import random
from enum import Enum, auto
import logging
import pandas as pd
import sklearn
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

pgfconfig = {
    "pgf.texsystem":   "pdflatex", # or any other engine you want to use
    "text.usetex":     True,       # use TeX for all texts
    "font.family":     "serif",
    "font.serif":      [],         # empty entries should cause the usage of the document fonts
    "font.sans-serif": [],
    "font.monospace":  [],
    "font.size":       6,         # control font sizes of different elements
    "axes.labelsize":  6,
    "legend.fontsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
}

mpl.use("pgf")
mpl.rcParams.update(pgfconfig)

DFPATH = "dpop_12man_12strat.pkl"
DFPATH = './scalability_results_dict.pkl'
#DFPATH = "./experiment_results_correct_best/cost/2/dpop_12man_12strat.pkl"

def main(path):
    with open(DFPATH, 'rb') as handle:
        dict_results = pickle.load(handle)

    df = pd.DataFrame(dict_results)
    # df = pd.read_pickle(path)
    print(df)

    df = df[df["number_managers"] > 3]
    print(df)
    df = df[(df["number_strategies"] == 5) | (df["number_strategies"] == 35) | (df["number_strategies"] == 20)]
    print(df)
    df = df[df["algorithm"] == "dpop"]

    df_grouped = df.groupby(["number_strategies", "algorithm", "timeout"])

    fmts = {5: "o-k", 20: "-k", 35: "-k"}
    linestyle = {5: "--", 20: ":", 35: ":" }
    color = {5: "black", 20: "black", 35: "black"}
    marker = {5: "o", 20: "x", 35: "D"}
    every = {5: 0.4, 20: 0.1, 35: 0.15}
    label = {5: "$|D_A|=|D_I|=5$", 20: "$|D_A|=|D_I|=20$", 35: "$|D_A|=|D_I|=35$"}
    # marker = {5: "o", 12: ""}
    # fig, ax = plt.subplots(1, 3, sharex=True, figsize=(3.5, 3.5), tight_layout=True)
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(7.12, 1.25), tight_layout=True)
    for k, g in df_grouped:
        print(k)
        # g.plot(x="number_managers", y=["msg_count"], ax=ax[0], grid=True, xlabel="Number of application managers (AMs)", ylabel="Number of messages", linestyle=linestyle[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]] + str(k)])
        # g.plot(x="number_managers", y=["msg_size"], ax=ax[1], grid=True,  xlabel="Number of application managers (AMs)", ylabel="Total message size (bytes)", linestyle=linestyle[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]] + str(k)])
        # g.plot(x="number_managers", y=["time_avg"], ax=ax[2], grid=True,  xlabel="Number of application managers (AMs)", ylabel="Time (seconds)", linestyle=linestyle[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]] + str(k)])
        series = g.set_index("number_managers").sort_index()
        series.plot(use_index=True, y=["msg_count"], ax=ax[0], grid=True, xlabel="Number of application managers (AMs)", ylabel="Number of messages", linestyle=linestyle[k[0]], color=color[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]]])
        series.plot(use_index=True, y=["msg_size"], ax=ax[1], grid=True,  xlabel="Number of application managers (AMs)", ylabel="Total size of msgs. (no. elements)", linestyle=linestyle[k[0]], color=color[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]]])
        series.plot(use_index=True, y=["time_avg"], ax=ax[2], grid=True,  xlabel="Number of application managers (AMs)", ylabel="Time (seconds)", linestyle=linestyle[k[0]], color=color[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]]])


    # df_grouped = df.groupby(["number_managers", "algorithm"])
    # # fig, ax = plt.subplots()
    # for k, g in df_grouped:
    #     # print(k)
    #     g.plot(x="number_strategies", y=["msg_count","msg_size"], ax=ax[1])
    #     # g.plot(x="number_strategies", y=["time"], ax=ax)
    # plt.show()

    plt.savefig("latex/DPOP_Eval_v2.pgf")
    plt.savefig("latex/DPOP_Eval_v2.pdf")


def secondplot(path):
    with open(DFPATH, 'rb') as handle:
        dict_results = pickle.load(handle)

    df = pd.DataFrame(dict_results)
    # df = pd.read_pickle(path)
    print(df)

    # df = df[df["number_managers"] == 10]
    df = df[df["number_managers"].isin([10, 20, 30])]
    # df = df[df["timeout"].isin([0, 0.5, 5, 10])]
    # df = df[df["timeout"].isin([0, 1.0, 10])]
    df = df[df["timeout"].isin([0])]
    # df = df[df["number_strategies"] < 70]
    # df = df[df["algorithm"] != "mgm2"]
    df = df[df["algorithm"] == "dpop"]
    print(df)
    df_grouped = df.groupby(["algorithm", "number_managers", "timeout"])
    df_dpop = df[df["algorithm"] == "dpop"]

    fmts = {10: "o-k", 20: "-k", 30: "-k"}
    linestyle = {10: "--", 20: ":", 30: ":" }
    color = {10: "black", 20: "black", 30: "black"}
    marker = {10: "o", 20: "x", 30: "D"}
    # every = {10: 0.4, 20: 0.1, 30: 0.110}
    every = {10: 0.1, 20: 0.1, 30: 0.1}
    label = {10: "Number of AMs = 10", 20: "Number of AMs = 20", 30: "Number of AMs = 30"}
    # marker = {5: "o", 12: ""}
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(7.12, 1.25), tight_layout=True)
    for k, g in df_grouped:
        # print(k, g.set_index("number_strategies"))
        # # g.set_index("number_strategies").plot(x="number_strategies", y=["time_avg"], ax=ax[0], grid=True, xlabel="Number of policies", ylabel="Total message size (bytes)")
        # g.set_index("number_strategies").sort_index().plot(use_index=True, y=["time_avg"], ax=ax[0], grid=True, xlabel="Number of policies", ylabel="Total message size (bytes)")

        # print(g.set_index("number_strategies").sort_index())
        # print(df_dpop.set_index("number_strategies").sort_index())
        # # df_diff = pd.DataFrame({"number_strategies": g["number_strategies"], "cost_avg": g.set_index("number_strategies").sort_index()["cost_avg"] - df_dpop.set_index("number_strategies").sort_index()["cost_avg"]})
        # df_diff = pd.DataFrame((g.set_index("number_strategies").sort_index()["cost_avg"] - df_dpop.set_index("number_strategies").sort_index()["cost_avg"]) ** 2)
        # print(df_diff)
        # # df_diff.plot(x="number_strategies", y=["cost_avg"], ax=ax[1], grid=True, xlabel="Number of policies", ylabel="Total message size (bytes)", label=[k])
        # df_diff.plot(use_index=True, y=["cost_avg"], ax=ax[1], grid=True, xlabel="Number of policies", ylabel="Total message size (bytes)", label=[k])
        # print(g["cost"].mean())

        # g.plot(x="number_strategies", y=["cost_std"], ax=ax[2], grid=True, xlabel="Number of policies", ylabel="Total message size (bytes)", label=[k])

               # linestyle=linestyle[k[0]], color=color[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=3, label=[label[k[0]]])

        print(k, g)
        series = g.set_index("number_strategies").sort_index()
        series.plot(use_index=True, y=["msg_count"], ax=ax[0], grid=True,  xlabel="Domain size ($|D_A| = |D_I|$)", ylabel="Number of messages",  ylim=[100, 750], legend=True, linestyle=linestyle[k[1]], color=color[k[1]], marker=marker[k[1]], markevery=every[k[1]], markersize=3, label=[label[k[1]]])
        series.plot(use_index=True, y=["msg_size"], ax=ax[1], grid=True,  xlabel="Domain size ($|D_A| = |D_I|$)", ylabel="Total size of msgs. (no. elements)",  legend=True, linestyle=linestyle[k[1]], color=color[k[1]], marker=marker[k[1]], markevery=every[k[1]], markersize=3, label=[label[k[1]]])
        series.plot(use_index=True, y=["time"], ax=ax[2], grid=True, xlabel="Domain size ($|D_A| = |D_I|$)", ylabel="Time (seconds)",  legend=True, linestyle=linestyle[k[1]], color=color[k[1]], marker=marker[k[1]], markevery=every[k[1]], markersize=3, label=[label[k[1]]])
        ax[0].legend(loc="upper left")

    plt.savefig("latex/DPOP_Eval_B_v2.pgf")
    plt.savefig("latex/DPOP_Eval_B_v2.pdf")


if __name__ == "__main__":
    main(DFPATH)
    secondplot(DFPATH)
