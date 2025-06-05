from src.self_improve import MemoryHandler

class DummyTrainer:
    def __init__(self):
        self.memory = type('M', (), {'buffer': []})()
        self.fine_tune_called = False
    def fine_tune(self):
        self.fine_tune_called = True


def test_memory_handler_triggers():
    trainer = DummyTrainer()
    trainer.memory.buffer.extend([1, 2])
    handler = MemoryHandler(trainer=trainer, threshold=2)
    handler.on_created(None)
    assert trainer.fine_tune_called
