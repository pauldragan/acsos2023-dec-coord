import random
from enum import Enum, auto
import logging
import pandas as pd
import sklearn
import numpy as np
import pickle

from experiments.decentralized.managers.application_manager import ApplicationManager

import sys
from os import path
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.transforms import Bbox

viridis_big = mpl.cm.get_cmap('Greys')
print(viridis_big(0.3), viridis_big(0.0), viridis_big(1.0))
newcmp = mpl.colors.ListedColormap(viridis_big(np.linspace(0.0, 0.5, 128)))
# plot_examples([viridis, newcmp])

# sys.exit()


pgfconfig = {
    "pgf.texsystem":   "pdflatex", # or any other engine you want to use
    "text.usetex":     True,       # use TeX for all texts
    "font.family":     "serif",
    "font.serif":      [],         # empty entries should cause the usage of the document fonts
    "font.sans-serif": [],
    "font.monospace":  [],
    "font.size":       6,         # control font sizes of different elements
    "axes.labelsize":  6,
    "legend.fontsize": 6,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
}

mpl.use("pgf")
mpl.rcParams.update(pgfconfig)

# VARIANTS = ["cost", "pref"]
VARIANTS = ["pref"]
# EXPERIMENTS = [1, 2, 3]
# EXPERIMENTS = [1]
EXPERIMENTS = ["2_ENER", "2_PERF", "3_PERF"]

# EXPERIMENTS_LABELS = {1: "Fixed strategies", 2: "Un-coordinated selection", 3: "Coordinated selection"}
# EXPERIMENTS_LABELS = {1: "Baseline 1", 2: "Baseline 2", 3: "CoAdapt"}
EXPERIMENTS_LABELS = {"2_ENER": "Baseline 1", "2_PERF": "Baseline 2", "3_PERF": "Coordination (CoADAPT)"}
# EXPERIMENTS_LINESTYLES = {1: ":", 2: "--", 3: "-"}
# EXPERIMENTS_LINESTYLES = {1: "-", 2: "-", 3: "-"}
EXPERIMENTS_LINESTYLES = {"2_ENER": "-", "2_PERF": "-", "3_PERF": "-"}
STRATEGIES_MARKERS = {0.0: "o", 0.5: "v", 1.0: "s", 1.5: "p", 2.0: "*"}
# STRATEGIES_NAMES = {0.0: "SI", 0.5: "ST", 1.0: "NN", 1.5: "EN", 2.0: "PE"}
STRATEGIES_NAMES = {0.0: "S", 0.5: "A", 1.0: "N", 1.5: "E", 2.0: "P"}
MANAGERS_NAMES = {"infra": "IM", "app_0": "AM1", "app_1": "AM2", "app_2": "AM3", "app_3": "AM4", "app_4": "AM5"}

# ROOT_DIR = "./experiment_results_december"
# ROOT_DIR = "./experiment_results_december_2"
# ROOT_DIR = "./experiment_results_110123"
# ROOT_DIR = "./experiment_results_03052023"
ROOT_DIR = "./experiment_results_17052023"
# ROOT_DIR = "./experiment_results_december_perfstrat"
# ROOT_DIR = "./experiment_results_correct_best"
# ROOT_DIR = "/tmp/simdex-experiments-plots/experiment_results"

WORKERSPATH = "./experiment_results/cost/worker_log.pkl"
JOBSPATH = "./experiment_results/cost/job_logs.pkl"
COORDINATIONTSPATH = "./experiment_results/cost/coordination_tses.pkl"

SEEDS = ["123456789", "987654321", "12121212", "34343434", "565656", "787878", "909090", "123123", "456456", "789789"]
# SEEDS = ["456456"]
# SEEDS = ["123456789"]
SEEDS = ["909090"]

# 12121212  565656  787878  123123 456456

# WORKERSPATH = "/tmp/simdex-experiments-plots/worker_log.pkl"
# JOBSPATH = "/tmp/simdex-experiments-plots/job_logs.pkl"

TIMEFRAME = "15D"

def full_extent(ax, title, ylabel, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    if title:
        # items += [ax, ax.title,  ax.yaxis.label]
        items += [ax, ax.yaxis.label]
    else:
        if ylabel:
            items += [ax, ax.xaxis.label, ax.yaxis.label]
        else:
            items += [ax, ax.xaxis.label]

    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def main():
    for variant in VARIANTS:
        seeds = SEEDS
        # if variant == "cost":
        #     seeds = ["123456789"]
        # else:
        #     seeds = SEEDS
        totals = {"variant": [], "seed": [], "experiment": [],
                  "ontime": [], "delayed": [], "late": [], "workers": [],
                  "utility": [], "expected_cost": []}
        for seed in seeds:
            # fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 4.5), tight_layout=True)
            # fig, ax = plt.subplots(4, 3, sharex=False, sharey=True, figsize=(7.1, 5), tight_layout=True, gridspec_kw={'height_ratios': [2, 1, 1, 1], 'wspace': 0.05, 'left': 0.1, 'right': 0.2, 'bottom': 0.1, 'top': 0.2})
            # fig, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(7.12, 2.75), gridspec_kw={'height_ratios': [2, 1]})
            # fig, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(7.12, 2.75), gridspec_kw={'height_ratios': [1.5, 1]})
            fig, ax = plt.subplots(5, 3, sharex=False, sharey=False, figsize=(7.12, 4.0), gridspec_kw={'height_ratios': [1.25, 1.25, 1.25, 1.25, 1]} )

            for idx_experiment, experiment in enumerate(EXPERIMENTS):
                strategies_arrays = []
                strategies_managers = []

                base_path = path.join(ROOT_DIR, str(variant), str(experiment))
                df_workers = pd.read_pickle(path.join(base_path, seed, "worker_log.pkl"))
                df_jobs = pd.read_pickle(path.join(base_path, seed, "job_logs.pkl"))
                df_strategy_log = pd.read_pickle(path.join(base_path, seed, "strategy_logs.pkl"))

                # print(list(df_jobs.columns))
                # sys.exit()

                with open(path.join(base_path, seed, "coordination_tses.pkl"), "rb") as fp:
                    coordination_tses_raw = pickle.load(fp)
                    coordination_tses = pd.Series(coordination_tses_raw)
                    coordination_tses = pd.to_datetime(coordination_tses, unit="s")

                if variant == "cost":
                    df_energy_cost = pd.DataFrame()
                    df_infra_cost = pd.DataFrame()

                    df_energy_cost["ts"] = df_workers["ts"]
                    df_energy_cost["energy_cost"] = (np.sin(df_energy_cost["ts"] / 9_000_000) + 1) / 2 / 5
                    df_infra_cost["ts"] = df_workers["ts"]
                    df_infra_cost["infra_cost"] = df_energy_cost["energy_cost"] * df_workers["active_workers"]
                    df_energy_cost["ts"] = pd.to_datetime(df_energy_cost["ts"], unit="s")
                    df_infra_cost["ts"] = pd.to_datetime(df_infra_cost["ts"], unit="s")
                else:
                    df_energy_cost = pd.DataFrame()
                    df_infra_cost = pd.DataFrame()

                    df_energy_cost["ts"] = df_workers["ts"]
                    df_energy_cost["energy_cost"] = np.ones(len(df_workers["ts"])) * 0.5
                    df_infra_cost["ts"] = df_workers["ts"]
                    df_infra_cost["infra_cost"] = df_energy_cost["energy_cost"] * df_workers["active_workers"]
                    df_energy_cost["ts"] = pd.to_datetime(df_energy_cost["ts"], unit="s")
                    df_infra_cost["ts"] = pd.to_datetime(df_infra_cost["ts"], unit="s")

                df_workers["ts"] = pd.to_datetime(df_workers["ts"], unit="s")
                df_jobs["start_ts"] = pd.to_datetime(df_jobs["start_ts"], unit="s")
                df_jobs["spawn_ts"] = pd.to_datetime(df_jobs["spawn_ts"], unit="s")
                # df_jobs["start_ts"] = pd.to_datetime(df_jobs["finish_ts"], unit="s") #?

                df_strategy_log["ts"] = pd.to_datetime(df_strategy_log["ts"], unit="s")
                df_strategy_log["strategy"].replace(to_replace=["NO_ADAPT",
                                                                "USER_EXP_MAX",
                                                                "USER_EXP_MAX_NN",
                                                                "ENERGY_MIN",
                                                                "PERFORMANCE"],
                                                                     value=[0.0,
                                                                            0.5, 1.0, 1.5, 2.0],
                                                    inplace=True)

                df_jobs = df_jobs[df_jobs["compilation_ok"] == 1]
                df_jobs["delayed"].replace(to_replace=["ontime", "delayed",
                                                       # "late"], value=[0.0, 5.0, 10.0], inplace=True)
                                                       "late"], value=[0.0, 0.5, 1.0], inplace=True)
                df_jobs["manager_strategy_cost"] = df_jobs["manager_real_strategy"]
                df_jobs["manager_strategy_cost"].replace(to_replace=["NO_ADAPT",
                                                                     "USER_EXP_MAX",
                                                                     "USER_EXP_MAX_NN"],
                                                                     value=[0.0,
                                                                     0.6, 1.0],
                                                                     inplace=True)
                df_jobs["manager_strategy_cost"] = pd.to_numeric(df_jobs["manager_strategy_cost"])

                df_jobs["infrastructure_strategy_cost"] = df_jobs["infrastructure_strategy"]
                df_jobs["infrastructure_strategy_cost"].replace(to_replace=["PERFORMANCE",
                                                                            "ENERGY_MIN"],
                                                                value=[1.0,
                                                                       0.0],
                                                                inplace=True)
                df_jobs["infrastructure_strategy_cost"] = pd.to_numeric(df_jobs["infrastructure_strategy_cost"])
                df_jobs_resampled = df_jobs[["start_ts",
                                             "delayed"]].resample(TIMEFRAME,
                                                                  axis=0,
                                                                  on="start_ts",
                                                                  origin=df_workers["ts"][0]).mean()
                df_satisfpref_resampled = df_jobs[["start_ts",
                                                   "satisf_pref"]].resample(TIMEFRAME,
                                                                            axis=0, on="start_ts",
                                                                            origin=df_workers["ts"][0]).mean()
                df_costpref_resampled = df_jobs[["start_ts",
                                                 "cost_pref"]].resample(TIMEFRAME,
                                                                        axis=0, on="start_ts",
                                                                        origin=df_workers["ts"][0]).mean()
                df_workers_resampled = df_workers.resample(TIMEFRAME, axis=0,
                                                           on="ts", origin=df_workers["ts"][0]).mean()
                df_energy_cost_resampled = df_energy_cost.resample(TIMEFRAME, axis=0,
                                                           on="ts", origin=df_workers["ts"][0]).mean()
                df_infra_cost_resampled = df_infra_cost.resample(TIMEFRAME, axis=0,
                                                                 on="ts", origin=df_workers["ts"][0]).mean()

                # df_averaged_managers = df_jobs[["start_ts", "delayed",
                #                                 "cost_pref", "satisf_pref",
                #                                 "manager_strategy_cost"]].resample(TIMEFRAME, axis=0,
                #                                                                    on="start_ts", origin=df_workers["ts"][0]).mean()
                # df_averaged_managers["utility"] = ( df_averaged_managers["cost_pref"] * (df_averaged_managers["manager_strategy_cost"] + df_infra_cost_resampled["infra_cost"]) + df_averaged_managers["satisf_pref"] * df_averaged_managers["delayed"])
                # df_averaged_managers.plot(y="utility", ax=ax[0,idx_experiment], grid=True, label=EXPERIMENTS_LABELS[experiment], linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black")

                df_expected_costs = df_jobs[["start_ts",
                                             "expected_cost"]].resample(TIMEFRAME, axis=0,
                                                                        on="start_ts", origin=df_workers["ts"][0]).mean()
                print(df_expected_costs)

                summed_df = None
                for k, g in df_jobs.groupby("manager_name"):
                    g2 = g[g["delayed"] > 0.0]
                    df_averaged_manager = g[["start_ts", "delayed",
                                             "cost_pref", "satisf_pref",
                                             "manager_strategy_cost"]].resample(TIMEFRAME, axis=0,
                                                                                on="start_ts", origin=df_workers["ts"][0]).mean()
                    df_summed_manager = g[["start_ts", "delayed",
                                           "cost_pref", "satisf_pref",
                                           "manager_strategy_cost"]].resample(TIMEFRAME, axis=0,
                                                                              on="start_ts", origin=df_workers["ts"][0]).sum()

                    # utility = ( df_averaged_manager["cost_pref"] * (df_averaged_manager["manager_strategy_cost"] + df_energy_cost_resampled["energy_cost"] * df_workers_resampled["active_workers"]) + df_averaged_manager["satisf_pref"] * df_averaged_manager["delayed"])

                    utility_cost = df_averaged_manager["cost_pref"] * (df_averaged_manager["manager_strategy_cost"] + df_energy_cost_resampled["energy_cost"] * df_workers_resampled["active_workers"])
                    utility_satisfaction = np.maximum(df_averaged_manager["satisf_pref"], df_averaged_manager["delayed"])
                    utility = utility_cost + utility_satisfaction

                    if summed_df is None:
                        summed_df = df_averaged_manager
                        summed_df["utility"] = utility
                    else:
                        summed_df["utility"] = summed_df["utility"] + utility

                    jobs_total = len(g["delayed"])
                    jobs_ontime = len(g[g["delayed"] == 0.0])
                    jobs_delayed = len(g[g["delayed"] == 0.5])
                    jobs_late = len(g[g["delayed"] == 1.0])
                    average_workers = df_workers["active_workers"].mean()
                    print("& AM{} & {:.2%} & {:.2%} & {:.2%} & {:.2f} & {:.2f} \\\\ \n \\cline{{2-5}} \\cline{{7-7}}".format(int(k[-1]) + 1,
                          jobs_ontime / jobs_total, jobs_delayed / jobs_total, jobs_late / jobs_total, average_workers,
                          utility.sum()))



                jobs_total = len(df_jobs["delayed"])
                jobs_ontime = len(df_jobs[df_jobs["delayed"] == 0.0])
                jobs_delayed = len(df_jobs[df_jobs["delayed"] == 0.5])
                jobs_late = len(df_jobs[df_jobs["delayed"] == 1.0])
                average_workers = df_workers["active_workers"].mean()
                print("& \\textbf{{ Total }} & \\textbf{{ {:.2%} }} & \\textbf{{ {:.2%} }} & \\textbf{{ {:.2%} }}& \\textbf{{ {:.2f} }}&  \\textbf{{ {:.2f} }}  \\\\ \n \\cline{{2-5}} \\cline{{7-7}}".format(jobs_ontime / jobs_total, jobs_delayed / jobs_total, jobs_late / jobs_total, average_workers,
                          summed_df["utility"].sum()))

                # summed_df.plot(y="utility", ax=ax[0,idx_experiment], grid=True,  linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black", legend=False, ylim=(2, 11))



                df_jobs_all_resampled = df_jobs[["start_ts",
                                                 "delayed"]].resample(TIMEFRAME,
                                                                      axis=0, on="start_ts", origin=df_workers["ts"][0]).count()

                df_jobs_delayed = df_jobs[df_jobs["delayed"] == 0.5]
                df_jobs_delayed_resampled = df_jobs_delayed[["start_ts",
                                                             "delayed"]].resample(TIMEFRAME,
                                                                                  axis=0,
                                                                                  on="start_ts",
                                                                                  origin=df_workers["ts"][0]).count()
                df_jobs_late = df_jobs[df_jobs["delayed"] == 1.0]
                df_jobs_late_resampled = df_jobs_late[["start_ts",
                                                       "delayed"]].resample(TIMEFRAME,
                                                                            axis=0,
                                                                            on="start_ts",
                                                                            origin=df_workers["ts"][0]).count()


                df_jobs_all_resampled.plot(y="delayed", ax=ax[0,idx_experiment], grid=True,  linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black", legend=False)
                df_jobs_late_resampled.plot(y="delayed", ax=ax[1,idx_experiment], grid=True,  linestyle=EXPERIMENTS_LINESTYLES[experiment], color="gray", ylim=(0, 1800), legend=True)
                df_jobs_delayed_resampled.plot(y="delayed", ax=ax[1,idx_experiment], grid=True,  linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black", ylim=(0, 1800), legend=True)
                df_workers_resampled.plot(y="active_workers", ax=ax[2,idx_experiment], grid=True,  linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black", ylim=(0, 5), legend=False)

                # del_line.set_label("delayed")
                # late_line.set_label("late")
                ax[1,idx_experiment].legend(["late", "delayed"])

                df_expected_costs.plot(y="expected_cost", ax=ax[3,idx_experiment], grid=True,  linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black", legend=False, ylim=(2, 7))
                ax[0, idx_experiment].set_title(label=EXPERIMENTS_LABELS[experiment])

                totals["variant"].append(int(k[-1]) + 1)
                totals["seed"].append(seed)
                totals["experiment"].append(experiment)
                totals["ontime"].append(jobs_ontime / jobs_total * 100.0)
                totals["delayed"].append(jobs_delayed / jobs_total * 100.0)
                totals["late"].append(jobs_late / jobs_total * 100.0)
                totals["workers"].append(average_workers)
                totals["utility"].append(summed_df["utility"].sum())
                totals["expected_cost"].append(df_expected_costs["expected_cost"].sum())

                # df_jobs_resampled.plot(y="delayed", ax=ax[1,idx_experiment], grid=True, label=EXPERIMENTS_LABELS[experiment], linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black")
                # df_infra_cost_resampled.plot(y="infra_cost", ax=ax[2,idx_experiment], grid=True, label=EXPERIMENTS_LABELS[experiment], linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black")
                # df_averaged_managers.plot(y="cost_pref", ax=ax[3,idx_experiment], grid=True, label=EXPERIMENTS_LABELS[experiment], linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black")

                # print("Variant: {}. Experiment: {}. Average utility: {}".format(variant, experiment, df_averaged_managers["utility"].mean()))
                print("Variant: {}. Experiment: {}. Total cost: {}. Total expected cost: {}. Delayed: {}".format(variant, experiment, summed_df["utility"].sum(), df_expected_costs["expected_cost"].sum(), summed_df["delayed"].sum()))

                for k, g in df_strategy_log.groupby("manager_name"):
                    g_mod = g.resample("3M", axis=0, on="ts", origin=df_workers["ts"][0]).first()
                    strategies_managers.append(str(k))
                    strategies_arrays.append(g_mod["strategy"].to_numpy())
                    g_mod["strategy"] = idx_experiment * 4 + g_mod["strategy"]
                    # g_mod.plot(x="ts", y="strategy", ax=ax[1,0], grid=True, label=None, linestyle=EXPERIMENTS_LINESTYLES[experiment], color="black")

                strategies_managers.reverse()
                strategies_arrays.reverse()
                strategies_image = np.stack(strategies_arrays, axis=0)
                print(strategies_image.shape)
                # fig, newax = plt.subplots(1, 1)
                # newax = ax[idx_experiment + 1, idx_experiment]
                newax = ax[4, idx_experiment]
                im = newax.imshow(strategies_image / 2.0, cmap=newcmp, vmax=1.0, vmin=0.0, aspect="auto", interpolation=None)
                for i in range(strategies_image.shape[0]):
                    for j in range(strategies_image.shape[1]):
                        text = newax.text(j, i, STRATEGIES_NAMES[strategies_image[i, j]],
                                          ha="center", va="center", color="black", fontsize="small")
                ylabels=[MANAGERS_NAMES[x] for x in strategies_managers]
                newax.set_yticks(ticks=range(strategies_image.shape[0]), labels=ylabels)
                xlabels = [x.strftime("%Y") for x in coordination_tses.tolist()[::4]]
                xticks = range(strategies_image.shape[1])[::4]
                newax.set_xticks(ticks=xticks, labels=xlabels, minor=False)
                # ax[0, idx_experiment].set_xticks(ticks=xticks, labels=xlabels, minor=False)
                # newax.set_xticks(ticks=coordination_tses.tolist(), minor=True)
                # newax.set_ylabel(EXPERIMENTS_LABELS[experiment])
                # newax.set_ylabel("Active policy")
                # newax.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3, handles=[mpl.text.Text(text="SI", label="Simple")])

                # ax[0,idx_experiment].set_xticks(ticks=coordination_tses.tolist(), minor=True)
                # ax[1,idx_experiment].set_xticks(ticks=coordination_tses.tolist(), minor=True)
                xlabels = [x.strftime("%Y") for x in coordination_tses.tolist()[::4]]
                ax[0,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], labels=xlabels, rotation=0, ha="center", minor=False)
                ax[1,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], labels=xlabels, rotation=0, ha="center", minor=False)
                ax[2,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], labels=xlabels, rotation=0, ha="center", minor=False)
                ax[3,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], labels=xlabels, rotation=0, ha="center", minor=False)
                # ax[1,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], labels=xlabels, rotation=0, ha="center", minor=False)


            # for _, val in coordination_tses.iteritems():
            #     ax[0,idx_experiment].axvline(x=val, ymax=0.05, linestyle="--", color="black", label="Coordination events")
            #     ax[1,idx_experiment].axvline(x=val, ymax=0.05, linestyle="--", color="black", label="Coordination events")


                ax[0,idx_experiment].set_xlabel("")
                ax[1,idx_experiment].set_xlabel("")
                ax[2,idx_experiment].set_xlabel("")
                ax[3,idx_experiment].set_xlabel("")
                ax[4,idx_experiment].set_xlabel("Date (years)")
            # ax[0,idx_experiment].set_ylabel("Average cost")
            # ax[0,idx_experiment].set_xlabel("")

            # ax[0,0].set_ylabel("Real cost")
            ax[0,0].set_ylabel("Jobs")
            ax[1,0].set_ylabel("Jobs")
            ax[2,0].set_ylabel("Workers")
            ax[3,0].set_ylabel("Coord. objective")
            ax[4,0].set_ylabel("Active policy")


            # ax[1,idx_experiment].set_ylabel("Strategy")

            # ax[0,idx_experiment].legend(loc="upper right")
            # ax[0,idx_experiment].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)
            # ax[0,idx_experiment].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)
            # ax[1,idx_experiment].legend(loc="upper right")


            # label = ["SIMPLE", "STATS", "NN", "ENER", "PERF"]
            # labels = []
            # ticks = []
            # for i in range(3):
            #     labels.extend(label)
            #     for t in [0, 0.5, 1.0, 1.5, 2.0]:
            #         ticks.append(t + i * 4)
            # ax[1,idx_experiment].set_yticks(ticks=ticks, labels=labels)
            # ax[1,idx_experiment].get_legend().remove()

            arr_coordination_tses_raw = np.array(coordination_tses_raw)
            arr_min = np.amin(arr_coordination_tses_raw)
            arr_max = np.amax(arr_coordination_tses_raw)
            coordinations = (arr_coordination_tses_raw - arr_min) / (arr_max - arr_min)

            # loc = plticker.FixedLocator(coordination_tses.tolist())
            # print(coordinations)
            # loc = plticker.FixedLocator(coordinations)
            # loc = plticker.MultipleLocator(base=(coordination_tses[1] - coordination_tses[0]))
            # ax[0,idx_experiment].xaxis.set_major_locator(loc)
            # ax[1,idx_experiment].xaxis.set_major_locator(loc)

            # ax[0,idx_experiment].grid(which='major', axis='x', linestyle='-')
            # ax[1,idx_experiment].grid(which='major', axis='x', linestyle='-')

            # ax[0,idx_experiment].set_xticks(ticks=coordination_tses.tolist(), minor=True)
            # # ax[1,idx_experiment].set_xticks(ticks=coordination_tses.tolist(), minor=True)
            # xlabels = [x.strftime("%Y-%m-%d") for x in coordination_tses.tolist()[::4]]
            # ax[0,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], labels=xlabels, rotation=0, ha="center", minor=False)
            # # ax[1,idx_experiment].set_xticks(ticks=coordination_tses.tolist()[::4], minor=False)

            # ax[0,idx_experiment].minorticks_off()
            # ax[1,idx_experiment].minorticks_off()




            # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('\n%d-%m-%Y'))

            # plt.savefig("latex/Exp_Test_{}.pdf".format(variant))
            plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
            plt.savefig("latex/Exp_Plot_{}_{}.pgf".format(variant, seed))
            plt.savefig("latex/Exp_Plot_{}_{}.pdf".format(variant, seed))
            # plt.show()

            # plt.show()

            # for cidx in range(3):
            #     for ridx in range(2):
            #         subax = ax[ridx, cidx]
            #         # extent = ax[0,0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            #         if ridx == 0:
            #             withTitle = True
            #         else:
            #             withTitle = False

            #         if cidx == 0:
            #             withylabel = True
            #         else:
            #             withylabel = False

            #         extent = full_extent(subax, withTitle, withylabel).transformed(fig.dpi_scale_trans.inverted())
            #         plt.savefig("latex/Exp_Plot_{}_{}_ax_{}_{}.pgf".format(variant, seed, ridx, cidx), bbox_inches=extent)
            #         plt.savefig("latex/Exp_Plot_{}_{}_ax_{}_{}.pdf".format(variant, seed, ridx, cidx), bbox_inches=extent)

        totals_df = pd.DataFrame(totals)
        print(totals_df)
        totals_df.to_csv("totals.csv")

        totals_per_baseline = totals_df.groupby("experiment").mean(numeric_only=True)
        print(totals_per_baseline)
        totals_per_baseline.to_csv("totals_per_baseline_{}.csv".format(variant))
        # totals_per_baseline.to_latex("totals_per_baseline_{}.tex".format(variant),
        #                              columns=["ontime", "delayed",
        #                                       "late", "workers", "utility",
        #                                       "expected_cost"],
        #                              header=["ontime", "delayed",
        #                                      "late", "Active workers", "Real cost", "Expected cost"],
        #                              float_format="%.2f")
        totals_per_baseline.to_latex("totals_per_baseline_{}.tex".format(variant),
                                     columns=["delayed", "late",
                                              "workers", ], header=True,
                                     index=True, index_names=True,


                                     float_format="%.2f")

        with open("totals_per_baseline.tex", "w") as tpb_fp:
            for idx, row in totals_per_baseline.iterrows():
                row_txt = "{} & {:.2f}\% & {:.2f}\% & {:.2f} \\\\ \n \\hline".format(EXPERIMENTS_LABELS[idx], row["delayed"], row["late"], row["workers"])
                print(row_txt)
                tpb_fp.write(row_txt + "\n")



if __name__ == "__main__":
    main()
