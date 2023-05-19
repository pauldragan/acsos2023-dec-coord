from experiments.decentralized.managers.application_manager import ApplicationManager
from experiments.decentralized.managers.infrastructure_manager import InfrastructureManager
from experiments.decentralized.managers.costs import energy_cost_performance, energy_cost_energymin
from interfaces import AbstractSelfAdaptingStrategy

from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Domain, Variable, create_variables, create_agents
from pydcop.dcop.relations import NAryFunctionRelation, UnaryFunctionRelation, NAryMatrixRelation
from pydcop.infrastructure.run import solve
from pydcop.utils.graphs import display_graph, display_bipartite_graph
from pydcop.computations_graph import pseudotree
from pydcop.distribution import oneagent

import random
from enum import Enum, auto
import logging
import pandas as pd
import sklearn
import numpy as np
import pickle

# random.seed(123456789)

class Experiment(Enum):
    NO_ADAPTATION = 0
    SELF_ADAPTATION = 1
    COORDINATION = 2

    def __str__(self):
        return self.name


class ChangeFactor(Enum):
    COST_VARIES = 0
    PREFERENCE_CHANGES = 1

    def __str__(self):
        return self.name

def compute_constraint_matrix_coord(constraintdict):
    matrix = np.zeros([len(InfrastructureManager.Strategy),
                       len(ApplicationManager.Strategy)])
    for idx_infra, infrastrat in enumerate(InfrastructureManager.Strategy):
        for idx_app, appstrat in enumerate(ApplicationManager.Strategy):
            matrix[idx_infra, idx_app] = constraintdict[(infrastrat, appstrat)]

    print("Constraint matrix: ", matrix)
    return matrix

def compute_constraint_matrix_app(constraintdict):
    matrix = np.zeros([len(ApplicationManager.Strategy)])
    for idx_app, appstrat in enumerate(ApplicationManager.Strategy):
        matrix[idx_app] = constraintdict[appstrat]

    print("Constraint matrix APP: ", matrix)
    return matrix


class DecentralizedSelfAdaptingStrategy(AbstractSelfAdaptingStrategy):
    """Represents a simple SA controller for activation/deactivation of queues based on current workload.

    Activates suspended worker queues when the system gets staturated, deactivates queues that are idle.
    """

    def __init__(self, max_long_queues, ref_jobs):
        self.experiment = Experiment.COORDINATION
        self.variant = ChangeFactor.COST_VARIES

        app_managers_preferences = [0.35, 0.4, 0.45, 0.5, 0.55]
        # app_managers_preferences = [0.5, 0.4, 0.45, 0.35, 0.55]
        # app_managers_preferences = [0.1, 0.1, 0.1, 0.1, 0.1]
        # app_managers_preferences = [0.5, 0.5, 0.6, 0.5, 0.4]
        self.app_managers_number = 5
        self.app_managers = []
        self.dict_manager_satisfaction = {"configuration": [], "satisfaction": [], "name": [], "ts": []}
        for i in range(self.app_managers_number):
            name = "app_{}".format(i)
            self.app_managers.append(ApplicationManager(max_long_queues,
                                                        ref_jobs,
                                                        i,
                                                        name=name,
                                                        cost_pref=app_managers_preferences[i],
                                                        iscollocated=False))
            # self.dict_manager_satisfaction["{}_satisfied".format(name)] = []
            # self.dict_manager_satisfaction["{}_configuration".format(name)] = []

        self.infra_manager = InfrastructureManager()
        self.gpu_resources = 1
        self.gpu_base_prob = 0.1

        self.gpu_coordination_strategy = None
        self.processed_jobs = 0

        # event_list are the ts of 10 evenly spaced jobs
        self.event_ts_list = [1524911717, 1543424958, 1557411389, 1574691097, 1586621518,
                              1604175052, 1607738218, 1618038446, 1633972719, 1636848040]

        # Preference counters
        self.last_preference_change_ts = 0
        # change preference every 6 months
        self.preference_change_period = 60.0 * 60.0 * 24.0 * 30.0 * 6.0

        # Baseline 2 - infra strategy switch
        self.last_infra_switch = 0
        # switch every 6 months
        self.infra_switch_period = 60.0 * 60.0 * 24.0 * 30.0 * 9.0
        self.infra_switch_idx = 0

        # Coordination counters
        self.last_coordination_ts = 0
        # coordinate every 3 months
        self.coordination_period = 60.0 * 60.0 * 24.0 * 30.0 * 3.0
        self.coordination_tses = []

        self.strategy_logs = {"ts": [], "manager_name": [], "strategy": []}

        self.expected_cost = 0.0


    def init(self, ts, dispatcher, workers):
        self._init_app(ts, dispatcher, workers)
        self._init_infra(ts, dispatcher, workers)

    def setup_experiment(self, exp_nr, exp_variant, exp_same_jobs, seed, exp_infra_perf):
        # TODO look for problems
        if exp_nr == 1:
            self.experiment = Experiment.NO_ADAPTATION
        elif exp_nr == 2:
            self.experiment = Experiment.SELF_ADAPTATION
        else:
            self.experiment = Experiment.COORDINATION

        if exp_variant:
            self.variant = ChangeFactor.PREFERENCE_CHANGES
        else:
            self.variant = ChangeFactor.COST_VARIES

        for app_man in self.app_managers:
            app_man.set_cost_variance(not exp_variant)
            app_man.set_usage_of_job_lists(exp_same_jobs)

        if exp_infra_perf:
            self.infra_manager.set_policy(InfrastructureManager.Strategy.PERFORMANCE)
        else:
            self.infra_manager.set_policy(InfrastructureManager.Strategy.ENERGY_MIN)

        random.seed(seed)

    def _init_app(self, ts, dispatcher, workers):
        for app_man in self.app_managers:
            app_man.init(ts, dispatcher, workers)

    def _init_infra(self, ts, dispatcher, workers):
        self.infra_manager.init(ts, dispatcher, workers)

    def _print_strategies(self):
        for appman in self.app_managers:
            print(appman.name, str(appman.policy))

        print(self.infra_manager.name, str(self.infra_manager.policy))

    def _compute_expected_cost(self, ts):
        exp_costs, exp_satisfaction = self._get_expectation(ts)
        total_expected_cost = 0.0
        infra_strat = self.infra_manager.policy
        for appman in self.app_managers:
            appman_strat = appman.policy
            appman_cost = appman.get_expected_cost(appman_strat, infra_strat, exp_costs, exp_satisfaction, ts)
            total_expected_cost += appman_cost

        return total_expected_cost

    def _record_strategies(self, ts):
        for appman in self.app_managers:
            self.strategy_logs["ts"].append(ts)
            self.strategy_logs["manager_name"].append(appman.name)
            self.strategy_logs["strategy"].append(str(appman.policy))

        self.strategy_logs["ts"].append(ts)
        self.strategy_logs["manager_name"].append(self.infra_manager.name)
        self.strategy_logs["strategy"].append(str(self.infra_manager.policy))

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self._check_for_event(ts)
        # print(ts, self.last_coordination_ts, ts - self.last_coordination_ts, self.coordination_period)
        if ((ts - self.last_coordination_ts) >= self.coordination_period):
            if self.experiment is Experiment.SELF_ADAPTATION:
                self._do_self_adaptation(ts, dispatcher, workers, job)
            elif self.experiment is Experiment.NO_ADAPTATION:
                pass
            else:
                # Else, it is coordination
                self._do_adapt_coordination(ts, dispatcher, workers, job)
            self.last_coordination_ts = ts
            self.coordination_tses.append(ts)
            self._print_strategies()
        # Application adaptation
        self._do_adapt_app(ts, dispatcher, workers, job)
        # Infrastructure adaptation
        self._do_adapt_infra(ts, dispatcher, workers, job)

        self._app_check_satisfaction(ts, dispatcher, workers, job)

        self._record_strategies(ts)

        if job:
            self.processed_jobs += 1
            job.expected_cost = self._compute_expected_cost(ts)
            # print("Expected cost: ", job.expected_cost)

    def _check_for_event(self, ts):
        if self.variant is ChangeFactor.PREFERENCE_CHANGES:
            # TODO this could be wrong
            # if any(ts >= event for event in self.event_ts_list):
                # self.event_ts_list = [event for event in self.event_ts_list if event > ts]
                # self._change_preference_of_manager(ts)
            if (ts - self.last_preference_change_ts) >= self.preference_change_period:
                print("Changing preferences!")
                self._change_preference_of_manager(ts)
                self.last_preference_change_ts = ts

    def _change_preference_of_manager(self, ts):
        # currently changes the preference of app_manager[0]
        # self.app_managers[0].change_preferences()
        for app_man in self.app_managers:
            app_man.change_preferences()

    def _get_expectation(self, ts):
        expected_cost = {}
        if self.variant is ChangeFactor.COST_VARIES:
            expected_cost = {InfrastructureManager.Strategy.PERFORMANCE:  energy_cost_performance(ts),
                             InfrastructureManager.Strategy.ENERGY_MIN: energy_cost_energymin(ts),
                             ApplicationManager.Strategy.NO_ADAPT: 0.0,
                             ApplicationManager.Strategy.USER_EXP_MAX: 0.6,
                             ApplicationManager.Strategy.USER_EXP_MAX_NN: 1.0}
        else:
            expected_cost = {InfrastructureManager.Strategy.PERFORMANCE: 0.8,
                             InfrastructureManager.Strategy.ENERGY_MIN: 0.2,
                             ApplicationManager.Strategy.NO_ADAPT: 0.0,
                             ApplicationManager.Strategy.USER_EXP_MAX: 0.6,
                             ApplicationManager.Strategy.USER_EXP_MAX_NN: 1.0}

        expected_satisfaction = {InfrastructureManager.Strategy.PERFORMANCE: 0.2,
                                 InfrastructureManager.Strategy.ENERGY_MIN: 0.8,
                                 ApplicationManager.Strategy.NO_ADAPT: 0.7,
                                 ApplicationManager.Strategy.USER_EXP_MAX: 0.2,
                                 ApplicationManager.Strategy.USER_EXP_MAX_NN: 0.1}
        return expected_cost, expected_satisfaction

    def _do_self_adaptation(self, ts, dispatcher, workers, job=None):
        # dcop = DCOP("coord")
        # d_app = Domain("app_domain", "appstrat", list(ApplicationManager.Strategy))
        # d_infra = Domain("infra_domain", "infrastrat", list(InfrastructureManager.Strategy))

        # variables = []
        # relations = []

        expected_cost, expected_satisfaction = self._get_expectation(ts)

        # v_infra = Variable("infra_var", d_infra, InfrastructureManager.Strategy.PERFORMANCE)
        # variables.append(v_infra)

        # app_vars = {}
        # idx = 0
        # app_man = self.app_managers[idx]
        # v_name = "app_{}_var".format(idx)
        # v_app = Variable(v_name, d_app,
        #                  ApplicationManager.Strategy.NO_ADAPT)
        # app_vars[app_man.get_name()] = v_app
        # constraint_dict = app_man.compute_infra_constraint(expected_cost,
        #                                                    expected_satisfaction, ts)

        # # fconstraint = lambda vinfra, vapp: constraint_dict[(vinfra, vapp)]
        # # relation = NAryFunctionRelation(fconstraint, [v_infra,
        #                                               # v_app],
        #                                 # "coordconstraint_{}".format(idx))
        # relation = NAryMatrixRelation([v_infra, v_app],
        #                               compute_constraint_matrix_coord(constraint_dict),
        #                               "coordconstraint_{}".format(idx))
        # dcop.add_constraint(relation)

        # constraint_dict_app = app_man.compute_app_constraint(expected_cost,
        #                                                      expected_satisfaction, ts)
        # # fconstraint_app = lambda vapp : constraint_dict_app[vapp]
        # # relation_app = UnaryFunctionRelation("coordconstraint_app_{}".format(idx),
        # #                                      v_app, fconstraint_app)
        # relation_app = NAryMatrixRelation([v_app],
        #                                   compute_constraint_matrix_app(constraint_dict_app),
        #                                   "coordconstraint_app_{}".format(idx))
        # dcop.add_constraint(relation_app)

        # variables.append(v_app)
        # relations.append(relation)
        # relations.append(relation_app)

        # print(v_name, app_man.cost_pref, " ", app_man.satisfaction_pref, " ", constraint_dict, constraint_dict_app)
        # dcop.add_agents(create_agents('a', list(range(len(variables))), capacity=50))
        # print("created agents!")
        # metrics = solve(dcop, 'dpop', 'oneagent')
        # # print(metrics)
        # assignment = metrics["assignment"]
        # print(metrics, assignment)
        # # sys.exit()

        # self.infra_manager.set_policy(assignment["infra_var"])
        # self.app_managers[idx].set_policy(assignment["app_{}_var".format(idx)])


        # cap = 200

        # if (ts - self.last_infra_switch) > self.infra_switch_period:
        #     if self.infra_switch_idx == 0:
        #         self.infra_switch_idx = 1
        #         self.infra_manager.set_policy(InfrastructureManager.Strategy.PERFORMANCE)
        #     else:
        #         self.infra_switch_idx = 0
        #         self.infra_manager.set_policy(InfrastructureManager.Strategy.ENERGY_MIN)

        #     self.last_infra_switch = ts


        infra_strat = self.infra_manager.get_policy()
        expected_infra_cost = expected_cost[infra_strat]
        expected_infra_satisf = expected_satisfaction[infra_strat]
        # for app_man in self.app_managers[1:]:
        for app_man in self.app_managers:
            constr = {}
            for app_strat in ApplicationManager.Strategy:
                if not app_man.do_act:
                    cost = (expected_infra_cost + expected_cost[app_strat])
                    satisfaction = (expected_infra_satisf + expected_satisfaction[app_strat])
                    constr[app_strat] = app_man.cost_pref * cost + app_man.satisfaction_pref * satisfaction
                else:
                    cap = app_man.satisfaction_pref
                    cost = (expected_infra_cost + expected_cost[app_strat])
                    satisfaction = (expected_infra_satisf + expected_satisfaction[app_strat])
                    # act_satisf = app_man.satisfaction_pref * satisfaction
                    act_satisf = satisfaction
                    if act_satisf < cap:
                        act_satisf = cap
                    constr[app_strat] = app_man.cost_pref * cost + act_satisf

            policy = min(constr, key=constr.get)
            app_man.set_policy(policy)
            print(constr, policy)
        # TODO what about infra?

    def _do_adapt_coordination(self, ts, dispatcher, workers, job=None):
        dcop = DCOP("coord")
        d_app = Domain("app_domain", "appstrat", list(ApplicationManager.Strategy))
        d_infra = Domain("infra_domain", "infrastrat", list(InfrastructureManager.Strategy))

        variables = []
        relations = []

        v_infra = Variable("infra_var", d_infra, InfrastructureManager.Strategy.PERFORMANCE)
        variables.append(v_infra)

        def fun_gpu_constraint(gpu_number, **args):
            # TODO print args for gpu constraints
            # print("*args", args)
            gpu_users = sum([1 for i in args if i is
                             ApplicationManager.Strategy.USER_EXP_MAX_NN])
            cost = 0
            if gpu_users > gpu_number:
                cost = 10000
            return cost


            

        test_exp_costs, test_exp_satisf = self._get_expectation(ts)

        app_vars = {}
        for idx, app_man in enumerate(self.app_managers):
            v_name = "app_{}_var".format(idx)
            v_app = Variable(v_name, d_app,
                             ApplicationManager.Strategy.NO_ADAPT)
            app_vars[app_man.get_name()] = v_app
            constraint_dict = app_man.compute_infra_constraint(test_exp_costs,
                                                               test_exp_satisf, ts)

            relation = NAryMatrixRelation([v_infra, v_app],
                                          compute_constraint_matrix_coord(constraint_dict),
                                          "coordconstraint_{}".format(idx))
            # fconstraint = lambda vinfra, vapp: constraint_dict[(vinfra, vapp)]
            # relation = NAryFunctionRelation(fconstraint, [v_infra,
            #                                               v_app],
            #                                 "coordconstraint_{}".format(idx))
            dcop.add_constraint(relation)

            constraint_dict_app = app_man.compute_app_constraint(test_exp_costs,
                                                                 test_exp_satisf, ts)
            # fconstraint_app = lambda vapp : constraint_dict_app[vapp]
            # relation_app = UnaryFunctionRelation("coordconstraint_app_{}".format(idx),
            #                                      v_app, fconstraint_app)
            relation_app = NAryMatrixRelation([v_app],
                                              compute_constraint_matrix_app(constraint_dict_app),
                                              "coordconstraint_app_{}".format(idx))
            dcop.add_constraint(relation_app)

            variables.append(v_app)
            relations.append(relation)
            relations.append(relation_app)

            print(v_name, app_man.cost_pref, " ", app_man.satisfaction_pref, " ", constraint_dict, constraint_dict_app)

        print("!!!! Constraints:", dcop.constraints)

        # Add collocation constraints
        added_experiment_five = False
        if added_experiment_five:
            collocated_managers_apps = [app_vars[man.get_name()] for man
                                        in self.app_managers if man.iscollocated()]
            fconstraint_gpu = lambda **managers: fun_gpu_constraint(self.gpu_resources, **managers)
            relation = NAryFunctionRelation(fconstraint_gpu, collocated_managers_apps,
                                            "collocationconstraint")
            print("Collocated managers apps: ", collocated_managers_apps)
            print("relation arity ", relation.arity)
            dcop.add_constraint(relation)
            relations.append(relation)

        # display_graph(variables, relations)
        # display_bipartite_graph(variables, relations)

        # dcop.add_agents(create_agents('a', list(range(len(variables))), capacity=50))
        # dcop.add_agents(create_agents('a', list(range(len(relations) + len(variables))), capacity=500))
        dcop.add_agents(create_agents('a', list(range(len(variables))), capacity=50))
        print("created agents!")
        cg = pseudotree.build_computation_graph(dcop)
        print(cg)
        dist = oneagent.distribute(cg, dcop.agents.values())

        # metrics = solve(dcop, 'dpop', 'oneagent')
        metrics = solve(dcop, 'dpop', dist, cg)
        # print(metrics)
        assignment = metrics["assignment"]
        print("DCOP results: ", metrics, assignment)
        # sys.exit()

        self.infra_manager.set_policy(assignment["infra_var"])
        for idx, app_man in enumerate(self.app_managers):
            app_man.set_policy(assignment["app_{}_var".format(idx)])

        validation_costs = {}
        validation_app_strats = {}
        for infrastrat in InfrastructureManager.Strategy:
            validation_costs[infrastrat] = 0
            validation_app_strats[infrastrat] = {}
            for idx, app_man in enumerate(self.app_managers):
                constraint_dict = app_man.compute_infra_constraint(test_exp_costs,
                                                               test_exp_satisf, ts)
                constraint_dict_app = app_man.compute_app_constraint(test_exp_costs,
                                                                     test_exp_satisf, ts)
                print(infrastrat, app_man.name)
                mincost = 100000
                beststrat = None
                for appstrat in ApplicationManager.Strategy:
                    cost1 = constraint_dict[(infrastrat, appstrat)]
                    cost2 = constraint_dict_app[appstrat]
                    cost = cost1 + cost2
                    # cost = (constraint_dict[(infrastrat, appstrat)])
                    print(appstrat, cost1, cost2, cost)
                    if cost < mincost:
                        mincost = cost
                        beststrat = appstrat
                validation_costs[infrastrat] += mincost
                validation_app_strats[infrastrat][app_man.name] = beststrat
        print("Validation_costs: ", validation_costs)
        print("Validation strats: ", validation_app_strats)
        # sys.exit()

    def _do_adapt_app(self, ts, dispatcher, workers, job=None):
        if self.processed_jobs % self.app_managers_number == 0:
            self._app_handle_gpus()
        man_id = self.processed_jobs % self.app_managers_number

        if job is None:
            for app_M in self.app_managers:
                app_M.do_adapt(ts, dispatcher, workers, job)
        else:
            self.app_managers[man_id].do_adapt(ts, dispatcher, workers, job)

    def _do_adapt_infra(self, ts, dispatcher, workers, job=None):
        self.infra_manager.do_adapt(ts, dispatcher, workers, job)

    def _app_check_satisfaction(self, ts, dispatcher, workers, job=None):
        for app_man in self.app_managers:
            satisfaction = app_man.get_satisfaction()
            configuration = app_man.get_policy()
            name = app_man.get_name()
            self.dict_manager_satisfaction["configuration"].append(configuration)
            self.dict_manager_satisfaction["satisfaction"].append(satisfaction)
            self.dict_manager_satisfaction["name"].append(name)
            self.dict_manager_satisfaction["ts"].append(ts)

        self.dict_manager_satisfaction["configuration"].append(self.infra_manager.get_policy())
        self.dict_manager_satisfaction["satisfaction"].append(True)
        self.dict_manager_satisfaction["name"].append(self.infra_manager.get_name())
        self.dict_manager_satisfaction["ts"].append(ts)

    def _app_handle_gpus(self):
        # self.gpu_requestors.clear()
        if True:
            for app_man in self.app_managers:
                app_man.set_gpu_available(True)
        else:
            collocated_gpu_requestors = []
            for app_man in self.app_managers:
                app_man.set_gpu_available(False)
                if app_man.requests_gpu():
                    if app_man.is_collocated():
                        collocated_gpu_requestors.append(app_man)
                    else:
                        app_man.set_gpu_available(True)

            for idx, app_man in enumerate(collocated_gpu_requestors):
                true_chance = 1.0 - len(collocated_gpu_requestors) * 0.1
                false_chance = 1.0 - true_chance
                choice = random.choices([True, False], k=1, weights=[true_chance, false_chance])
                app_man.set_gpu_available(choice[0])

    def save_logs(self):
        # df = ManagerLog.get_empty_df()
        satdf = pd.DataFrame(self.dict_manager_satisfaction)
        #  print(len(self.dict_manager_satisfaction))
        print("#### Manager Satisfaction")
        print(satdf)
        #  print(satdf[satdf["satisfaction"] == False])
        satdf.to_pickle("manager_satisfaction.pkl")
        dfs = []
        for app_man in self.app_managers:
            dfs.append(app_man.get_logs())

        dfs.append(self.infra_manager.get_logs())
        df = pd.concat(dfs)
        print("### Manager Logs")
        print(df)
        df.to_pickle("manager_logs.pkl")

        with open("coordination_tses.pkl", "wb") as fp:
            pickle.dump(self.coordination_tses, fp)

        df = pd.DataFrame(self.strategy_logs)
        print(df)
        df.to_pickle("strategy_logs.pkl")

    def collocate_first_two_app_managers(self):
        arbitrary_value = False
        if arbitrary_value:
            self.app_managers[0].iscollocated = True
            self.app_managers[1].iscollocated = True
            print("Collocated App_Managers 0 and 1 ")
