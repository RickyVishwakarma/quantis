import numpy as np
import pytest

from quantis.ai.models import RidgeSignalModel, make_model


def test_ridge_recovers_planted_linear_signal():
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(size=(n, 3))
    y = 0.05 * X[:, 0] - 0.03 * X[:, 1] + rng.normal(0, 0.01, n)  # f3 is noise
    model = RidgeSignalModel(["f1", "f2", "f3"], alpha=1.0).fit(X, y)
    preds = model.predict(X)
    assert np.corrcoef(preds, y)[0, 1] > 0.9
    # learned structure: f1 positive, f2 negative, f3 ~ zero
    assert model.coef_[0] > 0 and model.coef_[1] < 0
    assert abs(model.coef_[2]) < abs(model.coef_[0]) / 3


def test_ridge_attribution_sums_to_prediction():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 4))
    y = X @ np.array([0.1, -0.05, 0.02, 0.0]) + rng.normal(0, 0.01, 500)
    model = RidgeSignalModel(["a", "b", "c", "d"]).fit(X, y)
    x = X[10]
    attrib = model.attribution(x)
    reconstructed = sum(attrib.values()) + model.intercept_
    assert reconstructed == pytest.approx(model.predict(x.reshape(1, -1))[0], abs=1e-4)
    # attribution is sorted by absolute contribution
    vals = list(attrib.values())
    assert all(abs(vals[i]) >= abs(vals[i + 1]) for i in range(len(vals) - 1))


def test_make_model_unknown_type():
    with pytest.raises(ValueError, match="unknown model type"):
        make_model("transformer", ["f1"])


def test_ridge_pickles(tmp_path):
    import pickle

    rng = np.random.default_rng(0)
    model = RidgeSignalModel(["f1"]).fit(rng.normal(size=(100, 1)),
                                         rng.normal(size=100))
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps(model))
    loaded = pickle.loads(p.read_bytes())
    X = rng.normal(size=(5, 1))
    assert np.allclose(loaded.predict(X), model.predict(X))
