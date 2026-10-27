import json
import h5py

import numpy as np
import pytest

from pretrain.pipeline.data import PreparedWriter, Windows
from pretrain.pipeline.preprocessing import preprocess, profile
from pretrain.pipeline.dataset import read_metadata


def prepared_fixture(root):
    writer = PreparedWriter(
        root.with_suffix(".h5"),
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified", "kind": "synthetic_fixture"},
        preprocessing=profile(),
    )
    for subject in range(10):
        signal = np.random.default_rng(subject).normal(size=(2, 1024)).astype("float32")
        writer.add(
            signal,
            subject=str(subject),
            recording=f"record-{subject}",
            source={"sha256": str(subject) * 64},
            events=[(0, 0), (256, 1), (512, 0), (768, 1)],
        )
    writer.finish()
    return writer.path


def test_prepared_windows_preserve_disjoint_subjects_and_labels(tmp_path):
    path = prepared_fixture(tmp_path / "prepared")
    views = {split: Windows(path, split) for split in ("train", "validation", "test")}
    assert {name: len(view) for name, view in views.items()} == {
        "train": 28,
        "validation": 4,
        "test": 8,
    }
    for view in views.values():
        assert view[0]["eeg"].shape == (2, 256)
    metadata = read_metadata(path)
    assert len(metadata["subjects"]) == 10


def test_early_loader_exit_drains_only_inflight_batches(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    source = prepared_fixture(tmp_path / "source")
    trace = tmp_path / "reads.txt"
    script = tmp_path / "loader_shutdown.py"
    script.write_text(textwrap.dedent('''
        import sys
        import torch
        from torch.utils.data import Dataset
        from pretrain.pipeline.data import Windows, batches

        class CountedWindows(Dataset):
            def __init__(self, path, trace):
                self.view = Windows(path, "train")
                self.trace = trace
            def __len__(self):
                return len(self.view) * 64
            def __getitem__(self, index):
                item = self.view[index % len(self.view)]
                item["eeg"] = item["eeg"].repeat(7, 1)
                item["channel_ids"] = torch.arange(14)
                with open(self.trace, "a") as handle:
                    handle.write("x\\n")
                return item

        if __name__ == "__main__":
            for _ in range(3):
                for batch in batches(CountedWindows(sys.argv[1], sys.argv[2]), batch_size=64,
                                     seed=241, workers=2, shuffle=False):
                    break
            print("completed", flush=True)
    '''))
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    completed = subprocess.run([sys.executable, str(script), str(source), str(trace)],
                               cwd=root, env=env, capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "completed" in completed.stdout
    assert "terminate called without an active exception" not in completed.stderr
    assert "is killed by signal: Aborted" not in completed.stderr
    assert len(trace.read_text().splitlines()) <= 3 * 64 * 6


@pytest.mark.parametrize("workers, shuffle", [(0, False), (0, True), (2, True)])
def test_managed_loader_preserves_native_batch_order_and_rng(tmp_path, workers, shuffle):
    import torch
    from torch.utils.data import DataLoader

    from pretrain.pipeline.data import batches, collate

    view = Windows(prepared_fixture(tmp_path / "source"), "train")
    reference = DataLoader(
        view, batch_size=7, shuffle=shuffle, drop_last=False, collate_fn=collate,
        generator=torch.Generator().manual_seed(241), num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
    )
    selected = batches(view, batch_size=7, seed=241, workers=workers, shuffle=shuffle)
    assert len(reference) == len(selected)
    state = torch.get_rng_state().clone()
    for _ in range(2):
        left, right = list(reference), list(selected)
        assert len(left) == len(right)
        for expected, actual in zip(left, right):
            assert expected["window_id"] == actual["window_id"]
            for key in ("eeg", "channel_ids", "label"):
                assert torch.equal(expected[key], actual[key])
    assert torch.equal(state, torch.get_rng_state())


def test_prepared_signal_mutation_is_rejected(tmp_path):
    path = prepared_fixture(tmp_path / "prepared")
    row = int(Windows(path, "train").rows[0])
    with h5py.File(path, "r+") as handle:
        handle["X"][row] = 0
    with pytest.raises(ValueError, match="changed"):
        for split in ("train", "validation", "test"):
            Windows(path, split)[0]


def test_unknown_labels_do_not_become_verified(tmp_path):
    with pytest.raises(ValueError, match="Verified label"):
        PreparedWriter(
            tmp_path / "bad.h5",
            task="fixture",
            window_samples=256,
            channels=["C1", "Pz"],
            label_evidence={},
            preprocessing=profile(),
        )


def test_shared_windows_linking_every_subject_cannot_form_independent_splits(tmp_path):
    writer = PreparedWriter(
        tmp_path / "duplicates.h5",
        task="fixture",
        window_samples=256,
        channels=["C1", "Pz"],
        label_evidence={"status": "verified"},
        preprocessing=profile(),
    )
    shared = np.random.default_rng(123).normal(size=(2, 256)).astype("float32")
    for subject in range(10):
        values = np.random.default_rng(subject).normal(size=(2, 1280)).astype("float32")
        values[:, :256] = shared
        writer.add(
            values,
            subject=str(subject),
            recording=f"subject-{subject}",
            source={"sha256": "a" * 64},
            events=[(0, 0), (256, 0), (512, 1)],
        )
    with pytest.raises(ValueError, match="duplicate groups"):
        writer.finish()
    assert not writer.path.exists()


def test_preprocessing_is_recording_level_and_resampled():
    rate = 256
    time = np.arange(rate * 20) / rate
    amplitude = np.where(time < 10, 1e-6, 10e-6)
    signal = amplitude * np.sin(2 * np.pi * 10 * time)
    output = preprocess(np.stack((signal, -signal)), rate, ["C1", "Pz"])
    assert output.shape == (2, 2560)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert output[0, :1000].std() < output[0, -1000:].std() / 5
    np.testing.assert_allclose(output.mean(-1), 0, atol=1e-6)
    np.testing.assert_allclose(output.std(-1), 1, atol=1e-6)


def test_seed_bank_pools_official_partitions_and_preserves_signal_hashes(tmp_path):
    from pretrain.pipeline.splits import select_split

    source = prepared_fixture(tmp_path / "source")
    original = read_metadata(source)
    assignments = []
    for seed in (241, 242, 243, 244, 245):
        manifest = select_split(source, tmp_path / str(seed), seed)
        metadata = read_metadata(manifest)
        assignments.append(metadata["subjects"])
        assert metadata["split_seed"] == seed
        assert [r["sha256"] for r in metadata["records"]] == [
            r["sha256"] for r in original["records"]
        ]
        groups = [
            {s for s, role in metadata["subjects"].items() if role == name}
            for name in ("train", "validation", "test")
        ]
        assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    assert len({json.dumps(group, sort_keys=True) for group in assignments}) == 5


def test_upstream_rasr_evidence_is_optional_but_input_contract_is_checked(tmp_path):
    from pretrain.pipeline.splits import validate_prepared_input

    manifest = prepared_fixture(tmp_path / "data")
    metadata = read_metadata(manifest)
    metadata.pop("non_rasr_evidence", None)
    validate_prepared_input(metadata)
    metadata["preprocessing"]["clip"] = 999
    with pytest.raises(ValueError, match="preprocessing differs"):
        validate_prepared_input(metadata)
