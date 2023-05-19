import pandas as pd

class ManagerLog:
    def __init__(self, manager_name="manager"):
        self.data = {"selected_policy": [], "real_policy": [], "name":
                     [], "ts": [], "gpu_available": [], "expected_cost": []}
        self.manager_name = manager_name

        self.temp_selected_policy = None
        self.temp_real_policy = None
        self.temp_gpu_available = None
        self.temp_expected_cost = None

    def advance_time(self, ts):
        self.data["ts"].append(ts)
        self.data["selected_policy"].append(self.temp_selected_policy)
        self.data["real_policy"].append(self.temp_real_policy)
        self.data["gpu_available"].append(self.temp_gpu_available)
        self.data["name"].append(self.manager_name)
        self.data["expected_cost"].append(self.temp_expected_cost)

        self.temp_selected_policy = None
        self.temp_real_policy = None
        self.temp_gpu_available = None

    def log_selected_policy(self, policy):
        self.temp_selected_policy = str(policy)

    def log_real_policy(self, policy):
        self.temp_real_policy = str(policy)

    def log_gpu_available(self, available):
        self.temp_gpu_available = available

    def log_expected_cost(self, expected_cost):
        self.temp_expected_cost = expected_cost

    def get_data(self):
        df = pd.DataFrame(self.data)
        return df
