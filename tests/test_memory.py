import os
import shutil
from src.memory import Memory


def test_memory_add_and_limit(tmp_path):
    mem_dir = tmp_path / "mem"
    m = Memory(max_size=2, storage_path=str(mem_dir))

    m.add({"role": "user", "text": "hi"})
    m.add({"role": "assistant", "text": "there"})
    # buffer should have two entries
    assert len(m.buffer) == 2

    # Adding another entry should drop the oldest
    m.add({"role": "user", "text": "again"})
    assert len(m.buffer) == 2
    assert m.buffer[0]["text"] == "there"


def test_memory_save(tmp_path):
    mem_dir = tmp_path / "memsave"
    m = Memory(max_size=2, storage_path=str(mem_dir))
    m.add({"role": "user", "text": "hi"})
    m.add({"role": "assistant", "text": "bye"})

    m.save()
    files = list(mem_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("memory_") and files[0].suffix == ".json"
