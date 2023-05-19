import random
from enum import Enum, auto
import logging
import pandas as pd
import sklearn
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

pgfconfig = {
    "pgf.texsystem":   "pdflatex", # or any other engine you want to use
    "text.usetex":     True,       # use TeX for all texts
    "font.family":     "serif",
    "font.serif":      [],         # empty entries should cause the usage of the document fonts
    "font.sans-serif": [],
    "font.monospace":  [],
    "font.size":       9,         # control font sizes of different elements
    "axes.labelsize":  9,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
}

mpl.use("pgf")
mpl.rcParams.update(pgfconfig)

DFPATH = "dpop_12man_12strat.pkl"
#DFPATH = "./experiment_results_correct_best/cost/2/dpop_12man_12strat.pkl"

def main(path):
    df = pd.read_pickle(path)
    print(df)

    df = df[df["number_managers"] > 3]
    print(df)
    df = df[(df["number_strategies"] == 3) | (df["number_strategies"] == 5) | (df["number_strategies"] == 10)]
    print(df)
    df_grouped = df.groupby(["number_strategies", "algorithm"])

    fmts = {3: "o-k", 10: "-k", 5: "-k"}
    linestyle = {3: "--", 10: ":", 5: ":" }
    color = {3: "black", 10: "black", 5: "black"}
    marker = {3: "o", 10: "x", 5: "D"}
    every = {3: 0.4, 10: 0.1, 5: 0.15}
    label = {3: "$|D_A|=|D_I|=3$", 10: "$|D_A|=|D_I|=10$", 5: "$|D_A|=|D_I|=5$"}
    # marker = {3: "o", 12: ""}
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3.5), tight_layout=True)
    for k, g in df_grouped:
        print(k)
        g.plot(x="number_managers", y=["msg_count"], ax=ax[0], grid=True, xlabel="Number of application managers (AMs)", ylabel="Number of messages", linestyle=linestyle[k[0]], color=color[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=4, label=[label[k[0]]])
        g.plot(x="number_managers", y=["msg_size"], ax=ax[1], grid=True,  xlabel="Number of application managers (AMs)", ylabel="Total message size (bytes)", linestyle=linestyle[k[0]], color=color[k[0]], marker=marker[k[0]], markevery=every[k[0]], markersize=4, label=[label[k[0]]])
        # g.plot(x="number_managers", y=["time"], ax=ax)

    # df_grouped = df.groupby(["number_managers", "algorithm"])
    # # fig, ax = plt.subplots()
    # for k, g in df_grouped:
    #     # print(k)
    #     g.plot(x="number_strategies", y=["msg_count","msg_size"], ax=ax[1])
    #     # g.plot(x="number_strategies", y=["time"], ax=ax)
    # plt.show()

    plt.savefig("latex/DPOP_Eval.pgf")
    plt.savefig("latex/DPOP_Eval.pdf")

if __name__ == "__main__":
    main(DFPATH)
