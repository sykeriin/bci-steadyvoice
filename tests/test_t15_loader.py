import h5py, numpy as np, os
from steadyvoice.data.t15 import load_t15


def _write(path, n, rng):
    with h5py.File(path, "w") as f:
        for i in range(n):
            g = f.create_group(f"trial_{i:04d}")
            T, L = int(rng.integers(40, 60)), int(rng.integers(5, 9))
            g.create_dataset("input_features", data=rng.normal(size=(T, 512)).astype("float32"))
            ids = np.zeros(500, int); ids[:L] = rng.integers(1, 41, L)
            g.create_dataset("seq_class_ids", data=ids); g.attrs["seq_len"] = L


def test_loader_on_fake_schema(tmp_path):
    rng = np.random.default_rng(0)
    for name in ["t15.2023.09.01", "t15.2023.08.11", "t15.2023.08.13"]:      # unsorted on purpose
        os.makedirs(tmp_path / name)
        _write(tmp_path / name / "data_train.hdf5", 4, rng); _write(tmp_path / name / "data_val.hdf5", 2, rng)
    days = load_t15(str(tmp_path))
    assert [d.date for d in days] == [0, 2, 21]                              # sorted by calendar date
    assert days[0].adapt_x[0].shape[1] == 512 and 0 not in days[0].adapt_y[0] and len(days[0].eval_x) == 2


def test_loader_fails_loudly_on_wrong_schema(tmp_path):
    import pytest
    os.makedirs(tmp_path / "t15.2023.08.11")
    for s in ("train", "val"):
        with h5py.File(tmp_path / "t15.2023.08.11" / f"data_{s}.hdf5", "w") as f:
            f.create_group("trial_0000").create_dataset("features", data=np.zeros((3, 3)))
    with pytest.raises(KeyError, match="Schema differs"):
        load_t15(str(tmp_path))
