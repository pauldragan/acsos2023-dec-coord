from interfaces import AbstractDispatcher
from jobs import JobDurationIndex

def duration_filter(est_duration):
    def fnc(w):
        limit = w.get_attribute("limit")
        return limit is None or limit >= est_duration
    return fnc

class DecentralizedDispatcher(AbstractDispatcher):
    """Dispatches new jobs into the shortest active queue."""

    def __init__(self):
        # the dispatching algorithm only reads the index, SA strategy is responsible for filling the data
        self.duration_index = JobDurationIndex()

    def init(self, ts, workers):
        pass

    def dispatch(self, job, workers):
        # we need to estimate the duration of the job first (! no peeking to job.duration !)
        estimate = job.estimate if job.estimate != -1 else 0

        # get all workers marked as active
        active_workers = list(filter(lambda w: w.get_attribute("active"), workers))
        if not active_workers:
            raise RuntimeError("No active workers available, unable to dispatch job.")

        # select workers where the job would fit (estimate duration is under worker limit)
        best_workers = list(filter(duration_filter(estimate), active_workers))
        if len(best_workers) == 0:
            best_workers = active_workers  # fallback, if no worker passes the limit

        best_workers.sort(key=lambda x: x.jobs_count())
        target = best_workers[0]
        target.enqueue(job)

    def add_ref_job(self, job):
        """External interface for SA strategy (which can add jobs to index to modify dispatcher behavior)."""
        self.duration_index.add(job)
