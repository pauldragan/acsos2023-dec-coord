from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, Variable, create_variables, create_agents
from pydcop.dcop.relations import NAryFunctionRelation, UnaryFunctionRelation, NAryMatrixRelation
from pydcop.infrastructure.run import solve, run_local_thread_dcop
from pydcop.infrastructure.agents import Agent
from pydcop.algorithms.dpop import DpopAlgo
from pydcop.algorithms import AlgorithmDef
from pydcop.algorithms import ComputationDef
from pydcop.computations_graph.objects import ComputationNode
from pydcop.computations_graph.pseudotree import build_computation_graph
from pydcop.infrastructure.communication import InProcessCommunicationLayer
from pydcop.utils.graphs import display_graph, display_bipartite_graph
from pydcop.distribution import oneagent

import random
from enum import Enum, auto
import logging
import pandas as pd
import sklearn
import numpy as np
import pickle
import os.path


import matplotlib.pyplot as plt

NUMBER_APP_STRATEGIES = 10
NUMBER_APP_MANAGERS = 100
REPEATS = 10

# SEED = 909090
SEED = 123

random.seed(SEED)

# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for logger in loggers:
#     logger.setLevel(logging.INFO)

class ApplicationManager:
    # class Strategy(Enum):
    #     NO_ADAPT = 0
    #     USER_EXP_MAX = 1
    #     USER_EXP_MAX_NN = 2

    #     def __str__(self):
    #         return self.name

    # Strategy = Enum("Strategy", ["Strategy_{}".format(x) for x in range(NUMBER_APP_STRATEGIES)])
    Strategy = None

    def __init__(self, name="app"):
        self.name = name

    def get_name(self):
        return self.name

    def compute_infra_constraint(self):
        constraints_dict = {}
        for infra, service in [(x, y) for x in InfrastructureManager.Strategy for y in ApplicationManager.Strategy]:
            constraints_dict[(infra, service)] = random.random()
            # constraints_dict[(infra, service)] = 0.0

        return constraints_dict

    def compute_app_constraint(self):
        constraints_dict = {}
        for service in [y for y in ApplicationManager.Strategy]:
            constraints_dict[service] = random.random()
            # constraints_dict[service] = 0.0

        return constraints_dict

class InfrastructureManager:
    class Strategy(Enum):
        PERFORMANCE = 0
        ENERGY_MIN = 1

        def __str__(self):
            return self.name

    def __init__(self, name="infra"):
        # 0 - Performance
        # 1 - Energy saving
        self.policy = InfrastructureManager.Strategy.PERFORMANCE
        self.name = name


def main(number_app_managers, number_app_strategies, algorithm, timeout):
    ApplicationManager.Strategy = Enum("Strategy",
                                       ["Strategy_{}".format(x) for x
                                        in
                                        range(number_app_strategies)])
    InfrastructureManager.Strategy = Enum("Strategy",
                                          ["Strategy_{}".format(x) for x
                                           in
                                           range(number_app_strategies)])

    random.seed(a=SEED)

    app_managers = []
    for i in range(number_app_managers):
        name = "app_{}".format(i)
        app_managers.append(ApplicationManager(name=name))

    dcop = DCOP("coord")
    # d_app = Domain("app_domain", "", list(ApplicationManager.Strategy))
    d_infra = Domain("infra_domain", "", list(InfrastructureManager.Strategy))

    variables = []
    relations = []

    # v_infra = Variable("infra_var", d_infra, InfrastructureManager.Strategy.PERFORMANCE)
    v_infra = Variable("infra_var", d_infra, InfrastructureManager.Strategy(1))
    variables.append(v_infra)

    # algodef = AlgorithmDef(algorithm, {"stop_cycle": 5}, "min")
    algodef = AlgorithmDef(algorithm, {}, "min")


    def compute_constraint_matrix_coord(constraintdict):
        matrix = np.zeros([len(InfrastructureManager.Strategy),
                           len(ApplicationManager.Strategy)])
        for idx_infra, infrastrat in enumerate(InfrastructureManager.Strategy):
            for idx_app, appstrat in enumerate(ApplicationManager.Strategy):
                matrix[idx_infra, idx_app] = constraintdict[(infrastrat, appstrat)]

        # print("Constraint matrix: ", matrix)
        return matrix

    def compute_constraint_matrix_app(constraintdict):
        matrix = np.zeros([len(ApplicationManager.Strategy)])
        for idx_app, appstrat in enumerate(ApplicationManager.Strategy):
            matrix[idx_app] = constraintdict[appstrat]

        # print("Constraint matrix APP: ", matrix)
        return matrix

    # compinfra = DpopAlgo(v_infra, mode="min")
    # comm = InProcessCommunicationLayer()

    constraint_dicts = {}
    constraint_dicts_app = {}

    for idx, app_man in enumerate(app_managers):
        constraint_dict = app_man.compute_infra_constraint()
        constraint_dict_app = app_man.compute_app_constraint()

        constraint_dicts[app_man.name] = constraint_dict
        constraint_dicts_app[app_man.name] = constraint_dict_app

    agents = []
    app_vars = {}
    for idx, app_man in enumerate(app_managers):
        d_app = Domain("app_domain", "", list(ApplicationManager.Strategy))
        v_name = "app_{}_var".format(idx)
        v_app = Variable(v_name, d_app,
                         ApplicationManager.Strategy(1))
        app_vars[app_man.get_name()] = v_app

        # comp = DpopAlgo(v_app, mode="min")
        # compinfra.add_child(v_app)
        # comp.set_parent(v_infra)

        constraint_dict = constraint_dicts[app_man.name]
        constraint_dict_app = constraint_dicts_app[app_man.name]

        fconstraint = lambda vinfra, vapp : constraint_dict[(vinfra, vapp)] + constraint_dict_app[vapp]
        # relation = NAryFunctionRelation(fconstraint, [v_infra,
        #                                               v_app],
        #                                 "coordconstraint_{}".format(idx))
        relation = NAryMatrixRelation([v_infra, v_app],
                                      compute_constraint_matrix_coord(constraint_dict),
                                      "coordconstraint_{}".format(idx))
        # comp.add_relation(relation)
        dcop.add_constraint(relation)

        relation_app = NAryMatrixRelation([v_app],
                                          compute_constraint_matrix_app(constraint_dict_app),
                                          "coordconstraint_app_{}".format(idx))

        # fconstraint_app = lambda vapp : constraint_dict_app[vapp]
        # fconstraint2_app = lambda vapp, vapp2 : constraint_dict_app[vapp]
        # relation_app = UnaryFunctionRelation("coordconstraint_app_{}".format(idx), v_app,
        #                                       fconstraint_app)
        # relation_app = NAryFunctionRelation(fconstraint2_app, [v_app,
        #                                                        v_app],
        #                                     "coordconstraint_app_{}".format(idx))
        # comp.add_relation(relation_app)
        dcop.add_constraint(relation_app)

        variables.append(v_app)
        relations.append(relation)
        # relations.append(relation_app)

        # agent = Agent("agent_app_{}".format(idx), comm)
        # agent.add_computation(comp)
        # agents.append(agent)
    # agent_infra = Agent("agent_infra", comm)
    # agent_infra.add_computation(comp_infra)
    # agents.append(agent_infra)

    # display_graph(variables, relations)
    # display_bipartite_graph(variables, relations)

    # dcop.add_agents(create_agents('a', list(range(len(variables))), capacity=50))
    # agnts = create_agents('a', list(range(len(relations) + len(variables))), capacity=500)
    agnts = create_agents('a', list(range(3 * len(variables))), capacity=10000)
    dcop.add_agents(agnts)
    cg = build_computation_graph(dcop)
    # print(cg)
    dist = oneagent.distribute(cg, dcop.agents.values())
    # print("Distribution:   ", str(dist))

    # sys.exit()

    # for agnt in agnts:
        # print(agnt._computations)
    # print("created agents!")
    # print(dcop)
    # metrics = solve(dcop, algorithm,'oneagent', timeout=3)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.CRITICAL)
    # metrics = solve(dcop, algorithm,'oneagent', timeout=3)
    if timeout == 0.0:
        # metrics = solve(dcop, algodef,'oneagent', timeout=None)
        metrics = solve(dcop, algorithm,'oneagent', timeout=None)
    else:
        metrics = solve(dcop, algorithm,'oneagent', timeout=timeout)
        # metrics = solve(dcop, algodef,'oneagent', timeout=timeout)
    # metrics = solve(dcop, algodef, dist, cg, timeout=None)
    # print(metrics)
    assignment = metrics["assignment"]


    # print("Assignment:   ", assignment)

    validation_costs = {}
    validation_app_strats = {}
    for infrastrat in InfrastructureManager.Strategy:
        validation_costs[infrastrat] = 0
        validation_app_strats[infrastrat] = {}
        for idx, app_man in enumerate(app_managers):
            constraint_dict = constraint_dicts[app_man.name]

            constraint_dict_app = constraint_dicts_app[app_man.name]
            mincost = 100000
            beststrat = None

            # print(constraint_dict, constraint_dict_app)

            # print(infrastrat, app_man.name)
            for appstrat in ApplicationManager.Strategy:
                cost = (constraint_dict[(infrastrat, appstrat)] + constraint_dict_app[appstrat])
                # cost = (constraint_dict[(infrastrat, appstrat)])
                # cost = (constraint_dict_app[appstrat])
                # print(appstrat, cost)
                if cost < mincost:
                    mincost = cost
                    beststrat = appstrat
            validation_costs[infrastrat] += mincost
            validation_app_strats[infrastrat][app_man.name] = beststrat
    # print("Validation cost: ", validation_costs, " vs ", metrics["cost"])
    # print(validation_app_strats)

    # print(constraint_dicts)
    # print(constraint_dicts_app)

    return metrics

def experiment(in_managers, in_strategies):
    # x_algorithms = ["dpop", "syncbb", "maxsum", "mgm", "mgm2", "amaxsum"]
    # x_algorithms = ["maxsum"]
    # x_algorithms = ["dpop", "syncbb"]
    # x_algorithms = ["dpop", "maxsum", "amaxsum"]
    # x_algorithms = ["dsa", "adsa"] #, "dpop", "dsa", "adsa"]
    x_algorithms = ["dpop"]
    # x_algorithms = ["dsa"]
    # x_algorithms = ["mgm", "mgm2"]
    # x_managers = range(1, NUMBER_APP_MANAGERS + 2, 5)
    x_managers = in_managers
    # x_strategies = range(1, NUMBER_APP_STRATEGIES + 1)
    # x_strategies = [5, 20, 35]
    # x_strategies = range(1, 102, 10)
    x_strategies = in_strategies
    # x_strategies = range(1, 22, 2)
    # x_timeouts = [0, 0.5, 1, 3, 5]
    # x_timeouts = [1, 3, 5]
    # x_timeouts = [0.5, 1, 3, 5, 10, 25]
    # x_timeouts = [0.5, 1, 3, 5, 10]
    x_timeouts = [0]
    # x_timeouts = [10]

    filename = './scalability_results_dict.pkl'
    if os.path.exists(filename):
        print("{} exists. Bootstrapping.".format(filename))
        with open(filename, 'rb') as handle:
            dict_results = pickle.load(handle)
    else:
        print("{} does not exist. Starting from scratch.".format(filename))
        dict_results = {"algorithm": [], "number_managers": [],
                        "number_strategies": [], "msg_count": [], "msg_size": [],
                        "cost": [], "time": [],
                        "time_avg": [], "time_std": [], "time_var": [],
                        "cost_avg": [], "cost_std": [], "cost_var": [],
                        "msg_size_avg": [], "msg_size_std": [], "msg_size_var": [],
                        "msg_count_avg": [], "msg_count_std": [], "msg_count_var": [],
                        "timeout": []}
    # main(3, 3, "dpop")
    for x_algorithm in x_algorithms:
        for x_timeout in x_timeouts:
            for x_manager in x_managers:
                for x_strategy in x_strategies:
                    print("Running config: ({}, {}, {}, {})".format(x_algorithm, x_manager, x_strategy, x_timeout))
                    if (x_algorithm, x_timeout, x_manager, x_strategy) in zip(dict_results["algorithm"], dict_results["timeout"], dict_results["number_managers"], dict_results["number_strategies"]):
                        print("Data point exists. Skipping.")
                        continue

                    metrics_arr = []
                    time_arr = []
                    cost_arr = []
                    msg_count_arr = []
                    msg_size_arr = []
                    for reps in range(REPEATS):
                        metrics = main(x_manager, x_strategy, x_algorithm, x_timeout)
                        metrics_arr.append(metrics)
                        time_arr.append(metrics["time"])
                        cost_arr.append(metrics["cost"])
                        msg_count_arr.append(metrics["msg_count"])
                        msg_size_arr.append(metrics["msg_size"])

                    time_np = np.array(time_arr)
                    avg_time = np.mean(time_np)
                    std_time = np.std(time_np)
                    var_time = np.var(time_np)

                    cost_np = np.array(cost_arr)
                    avg_cost = np.mean(cost_np)
                    std_cost = np.std(cost_np)
                    var_cost = np.var(cost_np)

                    msg_count_np = np.array(msg_count_arr)
                    avg_msg_count = np.mean(msg_count_np)
                    std_msg_count = np.std(msg_count_np)
                    var_msg_count = np.var(msg_count_np)

                    msg_size_np = np.array(msg_size_arr)
                    avg_msg_size = np.mean(msg_size_np)
                    std_msg_size = np.std(msg_size_np)
                    var_msg_size = np.var(msg_size_np)

                    dict_results["number_managers"].append(x_manager)
                    dict_results["number_strategies"].append(x_strategy)
                    dict_results["msg_count"].append(metrics["msg_count"])
                    dict_results["msg_size"].append(metrics["msg_size"])
                    dict_results["algorithm"].append(x_algorithm)
                    dict_results["cost"].append(metrics["cost"])
                    dict_results["time"].append(metrics["time"])
                    dict_results["time_avg"].append(avg_time)
                    dict_results["time_std"].append(std_time)
                    dict_results["time_var"].append(var_time)

                    dict_results["cost_avg"].append(avg_cost)
                    dict_results["cost_std"].append(std_cost)
                    dict_results["cost_var"].append(var_cost)

                    dict_results["msg_size_avg"].append(avg_msg_size)
                    dict_results["msg_size_std"].append(std_msg_size)
                    dict_results["msg_size_var"].append(var_msg_size)

                    dict_results["msg_count_avg"].append(avg_msg_count)
                    dict_results["msg_count_std"].append(std_msg_count)
                    dict_results["msg_count_var"].append(var_msg_count)

                    dict_results["timeout"].append(x_timeout)

                    print(dict_results)

                # Save incrementally
                with open(filename, 'wb') as handle:
                    pickle.dump(dict_results, handle)

    with open(filename, 'wb') as handle:
        pickle.dump(dict_results, handle)

    # df = pd.DataFrame(dict_results)
    # print(df)
    # df.to_pickle("./dpop_12man_12strat.pkl")
    # df_grouped = df.groupby(["number_strategies", "algorithm"])
    # fig, ax = plt.subplots()
    # for k, g in df_grouped:
    #     # print(k)
    #     # g.plot(x="number_managers", y=["msg_count","msg_size", "cost"], ax=ax)
    #     g.plot(x="number_managers", y=["time"], ax=ax)

    # df_grouped = df.groupby(["number_managers", "algorithm"])
    # fig, ax = plt.subplots()
    # for k, g in df_grouped:
    #     # print(k)
    #     # g.plot(x="number_strategies", y=["msg_count","msg_size"], ax=ax)
    #     g.plot(x="number_strategies", y=["time"], ax=ax)
    # plt.show()

if __name__=="__main__":

    print("Running scalability test for increasing applications")
    experiment(range(1, NUMBER_APP_MANAGERS + 2, 5), [5, 20, 35])

    print("Running scalability test for increasing strategies")
    experiment([10, 20, 30], range(1, 102, 5))
