from enum import Enum

from experiments.decentralized.managers.manager_log import ManagerLog


class InfrastructureManager:
    class Strategy(Enum):
        PERFORMANCE = 0
        ENERGY_MIN = 1

        def __str__(self):
            return self.name

    def __init__(self, name="infra"):
        # 0 - Performance
        # 1 - Energy saving
        ## FIXME init strategy Infrastructure
        self.policy = InfrastructureManager.Strategy.PERFORMANCE
        self.name = name
        self.last_ts = None
        self.period = 0.0
        self.uptime = 0.0

        self.data_logger = ManagerLog(manager_name=self.name)

    def init(self, ts, dispatcher, workers):
        # 0 - Performance
        # 1 - Energy saving
        # self.policy = InfrastructureManager.Strategy.PERFORMANCE
        # self.policy = InfrastructureManager.Strategy.ENERGY_MIN

        # At the beginning, make only the first worker active
        if self.policy == 1:
            for worker in workers:
                worker.set_attribute("active", False)
            workers[0].set_attribute("active", True)

    def set_policy(self, policy):
        self.policy = policy

    def do_adapt(self, ts, dispatcher, workers, job=None):
        self.data_logger.log_selected_policy(self.policy)
        self.data_logger.log_real_policy(self.policy)

        if self.policy is InfrastructureManager.Strategy.PERFORMANCE:
            for worker in workers:
                worker.set_attribute("active", True)
        elif self.policy == InfrastructureManager.Strategy.ENERGY_MIN:
            # analyse the state of the worker queues
            active = 0
            overloaded = 0
            empty = []  # active yet empty work queues
            inactive = []  # inactive work queues
            for worker in workers:
                if worker.get_attribute("active"):
                    active += 1
                    if worker.jobs_count() == 0:
                        empty.append(worker)
                    elif worker.jobs_count() > 1:
                        overloaded += 1
                else:
                    inactive.append(worker)

            # take an action if necessary
            if len(empty) > 1 and overloaded == 0 and empty:
                empty[0].set_attribute("active", False)  # put idle worker to sleep
            elif inactive and overloaded > 0:
                inactive[0].set_attribute("active", True)  # wake inactive worker
        self.data_logger.advance_time(ts)

        if job is not None:
            job.infrastructure_strategy = str(self.policy)
            job.active_workers = len(list(filter(lambda w: w.get_attribute("active"), workers)))


    def measure_uptime(self, ts, workers, job):
        if job is not None and self.last_ts:
            dt = max(ts - self.last_ts, 0)
            active_workers = len(list(filter(lambda w: w.get_attribute("active", workers))))
            self.period += dt
            self.uptime += dt * float(active_workers)
            # job
        self.last_ts = ts

    def get_logs(self):
        return self.data_logger.get_data()

    def get_policy(self):
        return self.policy

    def get_name(self):
        return self.name
