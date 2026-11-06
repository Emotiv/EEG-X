import pickle

import numpy as np
import pytest

from pretrain.pipeline.artifacts import inspect_artifact, summarize


@pytest.mark.parametrize("protocol", [4, 5])
def test_legacy_arrays_are_inspected_and_mapped_without_loading_payload(tmp_path, protocol):
    path = tmp_path / "artifact.npy"
    arrays = {
        "X_train": np.arange(120_000, dtype=np.float32).reshape(100, 3, 400),
        "Y_train": np.arange(100) % 2,
        "id_train": np.arange(100) // 10,
    }
    with path.open("wb") as handle:
        pickle.dump(arrays, handle, protocol=protocol)
    values = inspect_artifact(path)
    mapped = values["X_train"].read(path)
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped, arrays["X_train"])
    assert summarize(path)["Y_train"]["values"] == [0, 1]


def test_arbitrary_pickle_globals_are_rejected(tmp_path):
    path = tmp_path / "unsupported.npy"
    path.write_bytes(pickle.dumps({"value": Exception("not array metadata")}))
    with pytest.raises(pickle.UnpicklingError):
        inspect_artifact(path)
