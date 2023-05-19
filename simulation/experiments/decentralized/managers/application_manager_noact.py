import copy
import random
import tensorflow as tf
import numpy as np
from enum import Enum

from experiments.decentralized.managers.infrastructure_manager import InfrastructureManager
from experiments.decentralized.managers.costs import cost_function, energy_cost_performance, energy_cost_energymin
from experiments.decentralized.managers.manager_log import ManagerLog
from jobs import JobDurationIndex


class ApplicationManager:
    class Strategy(Enum):
        NO_ADAPT = 0
        USER_EXP_MAX = 1
        USER_EXP_MAX_NN = 2

        def __str__(self):
            return self.name

    def __init__(self, max_long_queues, ref_jobs, index, name="app", iscollocated=False, cost_varies=False, cost_pref=0.5):
        # For app
        self.max_long_queues = max_long_queues

        self.ref_jobs_for_user_experience = ref_jobs[:]
        self.ref_jobs_for_user_experience.reverse()
        self.ref_jobs_for_nn = copy.deepcopy(ref_jobs)
        self.ref_jobs_for_nn.reverse()



        self.gpu_available = True
        # FIXME init strategy app man
        self.policy = ApplicationManager.Strategy.NO_ADAPT
        self.name = name
        self.iscollocated = iscollocated

        self.data_logger = ManagerLog(manager_name=self.name)

        # start with no real preference
        self.cost_pref = cost_pref
        self.satisfaction_pref = 1 - self.cost_pref

        # constr = self.compute_infra_constraint(test_exp_compute, test_exp_satisf)
        # print(constr)

        self.policy_as_expected = True
        self.cost_varies = cost_varies
        self.use_jobs_for_all_strategies = False
        self.duration_index = JobDurationIndex()
        # random.seed(index)

        # from NN strategy, standard values used
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

        self.layers_widths = [64]
        self.batch_size = 5000
        self.batch_epochs = 5
        self.buffer = []
        self.model = None
        self.predictor = None

        self.do_act = False

    def init(self, ts, dispatcher, workers):
        if self.model is None:
            self._init_model()

        if self.policy is ApplicationManager.Strategy.USER_EXP_MAX:
            self._app_update_duration_index(ts)
        elif self.policy is ApplicationManager.Strategy.USER_EXP_MAX_NN:
            self._advance_ts(ts)

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self.data_logger.log_selected_policy(self.policy)
        self.policy_as_expected = True

        real_policy = None

        if self.policy is ApplicationManager.Strategy.USER_EXP_MAX:
            self.data_logger.log_real_policy(self.policy)
            self._app_update_duration_index(ts)
            if job and job.compilation_ok:
                self.add_job_to_duration_index(job)

            if job:
                estimate = self.duration_index.estimate_duration(job.exercise_id, job.runtime_id)
                job.estimate = estimate if estimate is not None else job.limits / 2.0

            real_policy = self.policy
        elif (self.policy is ApplicationManager.Strategy.USER_EXP_MAX_NN) and (self.gpu_available is True):
            self.data_logger.log_real_policy(self.policy)
            self._advance_ts(ts)
            if job and job.compilation_ok:
                self.buffer.append(job)

            if job:
                self._train_batch(job)
                estimate = self._predict_estimate(job)
                # estimate = self.duration_index.estimate_duration(job.exercise_id, job.runtime_id)
                job.estimate = estimate if estimate is not None else job.limits / 2.0

            real_policy = self.policy

        # elif (self.policy is ApplicationManager.Strategy.USER_EXP_MAX_NN) and (self.gpu_available is False):
        #     self.data_logger.log_real_policy(ApplicationManager.Strategy.NO_ADAPT)
        #     self.policy_as_expected = False
        #     print(self.get_satisfaction())
        else:
            self.data_logger.log_real_policy(ApplicationManager.Strategy.NO_ADAPT)
            real_policy = ApplicationManager.Strategy.NO_ADAPT
            if job:
                job.estimate = job.limits / 2.0

        if self.use_jobs_for_all_strategies and job and job.compilation_ok:
            self._add_job_all_lists(job)

        if real_policy != self.policy:
            self.policy_as_expected = False
            # print(self.get_satisfaction())

        self.data_logger.log_gpu_available(self.gpu_available)
        self.data_logger.advance_time(ts)

        if job is not None:
            job.manager_name = self.name
            job.manager_real_strategy = str(real_policy)
            job.manager_selected_strategy = str(self.policy)
            job.cost_pref = self.cost_pref
            job.satisf_pref = self.satisfaction_pref
            job.cost = cost_function(ts) if self.cost_varies else 0.5

    def _train_batch(self, job):
        # train batch code from NN code
        if len(self.buffer) > self.batch_size:
            print("Training batch...")
            x_as_list = list(map(lambda j: [j.exercise_id, j.runtime_id], self.buffer))
            y_as_list = list(map(lambda j: [j.duration], self.buffer))
            x = tf.convert_to_tensor(x_as_list, dtype=tf.int32)
            y = tf.convert_to_tensor(y_as_list, dtype=tf.float32)

            self.model.fit(x, y, batch_size=len(self.buffer), epochs=self.batch_epochs, verbose=False)
            self.buffer = []  # reset the job buffer at the end

    def _init_model(self):
        all_inputs = tf.keras.Input(shape=(2,), dtype='int32')
        encoded_features = []
        domain_sizes = [1875, 20]
        for idx in range(0, 2):
            encoding_layer = lambda feature: tf.one_hot(feature, domain_sizes[idx] + 1)
            encoded_col = encoding_layer(all_inputs[:, idx])
            encoded_features.append(encoded_col)

        last_layer = tf.keras.layers.Concatenate()(encoded_features)
        for width in self.layers_widths:
            last_layer = tf.keras.layers.Dense(int(width), activation=tf.keras.activations.relu)(last_layer)
        output = tf.keras.layers.Dense(1, tf.keras.activations.exponential)(last_layer)

        self.model = tf.keras.Model(inputs=all_inputs, outputs=output)
        learning_rate = tf.keras.experimental.CosineDecay(0.01, 10000000)
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.Poisson())

    def _predict_estimate(self, job):
        if self.predictor is None:
            return job.limits / 2.0
        else:
            @tf.function
            def predict(input_var):
                return self.model(input_var, training=False)[0]

            x = np.array([[job.exercise_id, job.runtime_id]], dtype='int32')
            return predict(x).numpy()[0]

    def set_gpu_available(self, gpu_available):
        # print("Setting GPU available to ", gpu_available)
        self.gpu_available = gpu_available

    def set_policy(self, policy):
        self.policy = policy

    def requests_gpu(self):
        ret = False
        if self.policy is ApplicationManager.Strategy.USER_EXP_MAX_NN:
            # print("Requesting GPU")
            ret = True
        else:
            ret = False
        return ret

    def _app_update_duration_index(self, ts):
        while len(self.ref_jobs_for_user_experience) > 0 and self.ref_jobs_for_user_experience[-1].spawn_ts + self.ref_jobs_for_user_experience[-1].duration <= ts:
            job = self.ref_jobs_for_user_experience.pop()
            if job.compilation_ok:
                self.add_job_to_duration_index(job)

    def _advance_ts(self, ts):
        while len(self.ref_jobs_for_nn) > 0 and self.ref_jobs_for_nn[-1].spawn_ts + self.ref_jobs_for_nn[-1].duration <= ts:
            job = self.ref_jobs_for_nn.pop()
            if job.compilation_ok:
                self.buffer.append(job)

    def get_logs(self):
        return self.data_logger.get_data()

    def is_collocated(self):
        return self.iscollocated

    def compute_infra_constraint(self, exp_costs, exp_satisfaction, ts):
        cap = self.satisfaction_pref
        # self.cost_pref = 0.5
        # self.satisfaction_pref = 1 - self.cost_pref
        constraints_dict = {}
        for infra, service in [(x, y) for x in InfrastructureManager.Strategy for y in ApplicationManager.Strategy]:
            if not self.do_act:
                cost = exp_costs[infra]
                satisfaction = exp_satisfaction[infra]
                constraints_dict[(infra, service)] = (self.cost_pref * cost +
                                                      self.satisfaction_pref * satisfaction)
            else:
                # cost = (exp_costs[infra] + exp_costs[service])
                cost = exp_costs[infra]
                satisfaction = (exp_satisfaction[infra] + exp_satisfaction[service])
                # act_satisf = self.satisfaction_pref * satisfaction
                act_satisf = satisfaction
                if act_satisf < cap:
                    act_satisf = cap
                constraints_dict[(infra, service)] = self.cost_pref * cost + act_satisf

        return constraints_dict

    def compute_app_constraint(self, exp_costs, exp_satisfaction, ts):
        constraints_dict = {}
        for service in [y for y in ApplicationManager.Strategy]:
            cost = exp_costs[service]
            if not self.do_act:
                satisfaction = exp_satisfaction[service]
            else:
                satisfaction = 0
            constraints_dict[service] = (self.cost_pref * cost +
                                         self.satisfaction_pref *
                                         satisfaction)
        return constraints_dict

    def get_expected_cost(self, app_policy, infra_policy, exp_costs, exp_satisfaction, ts):
        infra_constraint = self.compute_infra_constraint(exp_costs, exp_satisfaction, ts)
        app_constraint = self.compute_app_constraint(exp_costs, exp_satisfaction, ts)

        return infra_constraint[(infra_policy, app_policy)] + app_constraint[app_policy]

    def get_policy(self):
        return self.policy

    def get_satisfaction(self):
        return self.policy_as_expected

    def get_name(self):
        return self.name

    def set_cost_variance(self, cost_varies):
        self.cost_varies = cost_varies

    def add_job_to_duration_index(self, job):
        self.duration_index.add(job)

    def change_preferences(self):
        # prefrences for cost
        # possible_preferences = [0.1, 0.3, 0.40, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95]

        # possible_satisf_preferences = np.array([0.1, 0.3, 0.45, 0.50, 0.70, 0.80, 0.90, 0.95]) * 1.25
        possible_satisf_preferences = np.array([0.1, 0.3, 0.45, 0.50, 0.70, 0.80, 0.90, 0.95])

        # possible_satisf_preferences = np.array([0.50, 0.70, 0.80, 0.90, 0.95]) * 1.25
        # possible_preferences = np.arange(0.1, 0.4, 0.1)
        possible_preferences = np.arange(0.1, 0.9, 0.1)
        # possible_preferences.remove(self.cost_pref)
        self.cost_pref = random.choice(possible_preferences)
        # self.satisfaction_pref = 1 - self.cost_pref
        self.satisfaction_pref = random.choice(possible_satisf_preferences)
        print("new cost_pref: ", str(self.cost_pref))
        print("new satisfaction_pref: ", str(self.satisfaction_pref))

    def set_usage_of_job_lists(self, exp_same_jobs):
        self.use_jobs_for_all_strategies = exp_same_jobs

    def _add_job_all_lists(self, job):
        if self.policy is ApplicationManager.Strategy.NO_ADAPT:
            self.buffer.append(job)
            self.add_job_to_duration_index(job)
        elif self.policy is ApplicationManager.Strategy.USER_EXP_MAX:
            self.buffer.append(job)
        else:
            self.add_job_to_duration_index(job)
        pass
