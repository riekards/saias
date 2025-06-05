from src.trainer import Trainer


def test_trainer_init():
    t = Trainer(config_path="configs/default.yaml")
    assert t.batch_size == 32
    # learning_rate comes from YAML as a string
    assert str(t.learning_rate) == "1e-4"
    assert t.num_epochs == 5
    assert hasattr(t, "memory")


def test_trainer_fine_tune(capsys):
    t = Trainer(config_path="configs/default.yaml")
    t.memory.add({"role": "user", "text": "hi"})
    t.fine_tune()
    captured = capsys.readouterr()
    assert (
        "PPO training" in captured.out
        or "No memory" in captured.out
        or "No dialogue pairs" in captured.out
    )
