import random
from enum import Enum, auto
import logging
import pandas as pd
import sklearn
import numpy as np

# from experiments.decentralized.managers.application_manager import ApplicationManager

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "serif"
# })

app_strats = {
    "NO_ADAPT": {"cost": 0.0, "satisf": 0.7},
    "USER_EXP": {"cost": 0.6, "satisf": 0.2},
    "USER_EXP_NN": {"cost": 1.0, "satisf": 0.1}
}

infra_strats = {
    "ENERGY": {"cost": 0.2, "satisf": 0.8},
    "PERFORMANCE": {"cost": 0.8, "satisf": 0.2}
}


def utility_simple(app_vals, infra_vals, cost_prefs, satisf_prefs):
    utility = (cost_prefs * (app_vals["cost"] + infra_vals["cost"]) +
               satisf_prefs * (app_vals["satisf"] + infra_vals["satisf"]))
    return utility

def utility_prod(app_vals, infra_vals, cost_prefs, satisf_prefs):
    utility = (cost_prefs * app_vals["cost"] * infra_vals["satisf"] +
               satisf_prefs * app_vals["satisf"] * infra_vals["satisf"])
    return utility

def utility_squared(app_vals, infra_vals, cost_prefs, satisf_prefs):
    utility = (cost_prefs * (app_vals["cost"] + infra_vals["cost"]) +
               satisf_prefs * (-(app_vals["satisf"] - infra_vals["satisf"]) * (app_vals["satisf"] - infra_vals["satisf"])))
    return utility

def utility_capped(app_vals, infra_vals, cost_prefs, satisf_prefs):
    utility = (cost_prefs * 1.0 * (app_vals["cost"] + infra_vals["cost"]) +
               np.maximum(satisf_prefs, (app_vals["satisf"] + infra_vals["satisf"])))

    return utility

def change_over_prefs(satisf_multi):
    cost_prefs = np.linspace(0, 1, num=100)
    # satisf_prefs = 1.0 - cost_prefs
    # satisf_prefs = np.ones_like(cost_prefs) * np.random.rand()
    satisf_prefs = np.ones_like(cost_prefs) * satisf_multi
    print(satisf_prefs)

    fig, ax = plt.subplots()
    for app_name, app_vals in app_strats.items():
        for infra_name, infra_vals in infra_strats.items():
            utility = utility_simple(app_vals, infra_vals, cost_prefs, satisf_prefs)
            # utility = utility_prod(app_vals, infra_vals, cost_prefs, satisf_prefs)
            # utility = utility_squared(app_vals, infra_vals, cost_prefs, satisf_prefs)
            # utility = utility_capped(app_vals, infra_vals, cost_prefs, satisf_prefs)
            ax.plot(cost_prefs, utility, label="{}+{}".format(app_name, infra_name))

    ax.set_xlabel("cost preference ($c_p$)")
    ax.set_ylabel("$utility(c_p) = c_p (cost(S_A) + cost(S_I)) + (1 - c_p) (satisf(S_A) + satisf(S_I))$")
    ax.set_title("Utility versus change in preference")
    ax.legend()
    plt.show()

def change_over_energy():
    time = np.arange(0, 2 * np.pi, 0.1)
    energy_cost = {}

    energy_cost["ENERGY"] = 0.2 + (1 + np.sin(time)) / 2 / 5
    energy_cost["PERFORMANCE"] = 0.2 + 4 * (1 + np.sin(time)) / 2 / 5

    print(np.amin(energy_cost["ENERGY"]), np.amin(energy_cost["PERFORMANCE"]))


    cost_prefs = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, ax = plt.subplots(len(cost_prefs), 1)
    for i, cost_pref in enumerate(cost_prefs):
        satisf_pref = 1 - cost_pref
        for app_name, app_vals in app_strats.items():
            for infra_name, infra_vals in infra_strats.items():
                time_vals = {}
                time_vals["cost"] = infra_vals["cost"] + energy_cost[infra_name]
                time_vals["satisf"] = infra_vals["satisf"]
                # utility = utility_simple(app_vals, time_vals, cost_pref, satisf_pref)
                # utility = utility_squared(app_vals, time_vals, cost_pref, satisf_pref)
                utility = utility_capped(app_vals, time_vals, cost_pref, satisf_pref)
                ax[i].plot(time, utility, label="{}+{}".format(app_name, infra_name))

                ax[i].set_xlabel("time")
                # ax[i].set_ylabel("$utility(c_p) = c_p (cost(S_A) + cost(S_I)) + (1 - c_p) (satisf(S_A) + satisf(S_I))$")
                ax[i].set_title("Utility versus change in energy costs ($c_p = {}$)".format(cost_pref))
                ax[i].legend()
    plt.show()


if __name__ == "__main__":
    for sm in range(1, 10):
        change_over_prefs(sm / 10.0 * 1.25)
    # change_over_energy()
