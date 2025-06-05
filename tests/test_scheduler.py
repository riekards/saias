from src import scheduler as sched_module

class DummyTrainer:
    def __init__(self):
        self.fine_tune_called = False
    def fine_tune(self):
        self.fine_tune_called = True

class DummyScheduler:
    def __init__(self):
        self.jobs = []
        self.started = False
    def add_job(self, func, trigger, minutes=None, next_run_time=None):
        self.jobs.append((func, trigger, minutes, next_run_time))
    def start(self):
        self.started = True
    def shutdown(self):
        self.started = False


def test_periodic_fine_tune(monkeypatch):
    trainer = DummyTrainer()
    created = {}
    def scheduler_factory():
        created['sched'] = DummyScheduler()
        return created['sched']

    monkeypatch.setattr(sched_module, "Trainer", lambda config_path="": trainer)
    monkeypatch.setattr(sched_module, "BackgroundScheduler", scheduler_factory)

    def raise_interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(sched_module.time, "sleep", raise_interrupt)

    sched_module.periodic_fine_tune(config_path="configs/default.yaml")

    sched = created['sched']
    assert not sched.started  # should be shutdown
    assert len(sched.jobs) == 1
