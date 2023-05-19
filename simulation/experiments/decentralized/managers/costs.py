import numpy as np

def cost_function(ts):
    return (np.sin(ts / 9_000_000) + 1) / 2

def energy_cost_performance(ts):
    cost = 0.2 + 4 * (1 + np.sin(ts / 9_000_000)) / 5.0
    print("perf cost", cost)
    return cost

def energy_cost_energymin(ts):
    cost = 0.2 + (1 + np.sin(ts / 9_000_000)) / 5.0
    print("min cost", cost)
    return cost
