from interfaces import AbstractMetricsCollector
from jobs import JobDurationIndex

import pandas as pd


class UserExperienceMetricsCollector(AbstractMetricsCollector):
    """User experience metrics attempts to asses user (dis)satisfaction with job delays.

    In general jobs that are expected to be quick (interactive) should not be delayed too long.
    On the other hand, long jobs can be delayed since the used would not wait for them interactively.
    Whether a job is expected to be long or short is also based on ref. solution jobs.
    All jobs are divided into 3 classes -- "ontime" :), "delayed" :|, and "late" :(
    """

    def _get_expected_duration(self, job):
        estimate = self.duration_index.estimate_duration(job.exercise_id, job.runtime_id)
        if estimate:
            return estimate

        return job.duration if job.compilation_ok else job.limits

    def __init__(self, ref_jobs, thresholds=[1.0, 2.0]):
        if (ref_jobs is None):
            raise RuntimeError("User experience metrics require ref. jobs to be loaded.")

        # create an index structure for job duration estimation
        self.duration_index = JobDurationIndex()
        for job in ref_jobs:
            if job.compilation_ok:
                self.duration_index.add(job)

        # category thresholds as multipliers of expected durations
        self.threshold_ontime, self.threshold_delayed = thresholds

        # collected counters
        self.jobs_ontime = 0
        self.jobs_delayed = 0
        self.jobs_late = 0

        self.jobs = []

    def job_finished(self, job):
        delay = job.start_ts - job.spawn_ts
        expected_duration = self._get_expected_duration(job)
        job.expected_duration = expected_duration
        ontime = max(expected_duration * self.threshold_ontime, 10.0)  # ontime / delayed threshold must be at least 10s
        delayed = max(expected_duration * self.threshold_delayed, 30.0)  # delayed / late threshold must be at least 30s
        if delay <= ontime:
            self.jobs_ontime += 1
            job.delayed = "ontime"
        elif delay <= delayed:
            self.jobs_delayed += 1
            job.delayed = "delayed"
        else:
            self.jobs_late += 1
            job.delayed = "late"

        # the metrics is adjusting the expectations of the job duration dynamically
        if job.compilation_ok:
            self.duration_index.add(job)

        self.jobs.append(job)

    def get_total_jobs(self):
        return self.jobs_ontime + self.jobs_delayed + self.jobs_late

    def print(self):
        print("Total jobs: {}, on time: {}, delayed: {}, late: {}".format(
            self.get_total_jobs(), self.jobs_ontime, self.jobs_delayed, self.jobs_late))
        print("Percentage on time: {:.2f}%, delayed : {:.2f}%, late: {:.2f}%".format(
            100 * (self.jobs_ontime / self.get_total_jobs()),
            100 * (self.jobs_delayed / self.get_total_jobs()),
            100 * (self.jobs_late / self.get_total_jobs())))

        job_df = pd.DataFrame(self.jobs)
        job_df.to_pickle("job_logs.pkl")

        print("### Jobs")
        print(job_df)
        #print(job_df["manager_name"])
        #print(job_df["manager_real_strategy"])
        #print(job_df["manager_selected_strategy"])
