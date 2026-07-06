import numpy as np
import pytest

from quantis.ai.models import RidgeSignalModel
from quantis.ai.registry import ModelRegistry, PromotionError, Stage


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models")


def register_dummy(registry, name="alpha"):
    model = RidgeSignalModel(["f1", "f2"])
    model.fit(np.random.default_rng(0).normal(size=(50, 2)),
              np.random.default_rng(1).normal(size=50))
    return registry.register(name=name, model=model,
                             metrics={"ic": 0.05}, feature_names=["f1", "f2"],
                             label="label_fwd_ret_5d",
                             signal_bounds=(-0.1, 0.1))


def test_register_and_roundtrip(registry):
    entry = register_dummy(registry)
    assert entry["stage"] == "EXPERIMENTAL"
    assert entry["version"] == 1
    assert entry["feature_schema_version"]
    loaded_entry, model = registry.load_model(entry["model_id"])
    assert loaded_entry["model_id"] == entry["model_id"]
    assert model.feature_names == ["f1", "f2"]


def test_versions_increment_per_name(registry):
    e1 = register_dummy(registry)
    e2 = register_dummy(registry)
    e3 = register_dummy(registry, name="beta")
    assert (e1["version"], e2["version"], e3["version"]) == (1, 2, 1)


def test_lifecycle_transitions_enforced(registry):
    e = register_dummy(registry)
    mid = e["model_id"]
    # cannot skip straight to PRODUCTION
    with pytest.raises(PromotionError, match="not a legal transition"):
        registry.promote(mid, Stage.PRODUCTION, approved_by="ricky")
    registry.promote(mid, Stage.CANDIDATE)
    with pytest.raises(PromotionError, match="not a legal transition"):
        registry.promote(mid, Stage.PRODUCTION, approved_by="ricky")
    registry.promote(mid, Stage.SHADOW)


def test_production_requires_human_and_shadow_report(registry):
    e = register_dummy(registry)
    mid = e["model_id"]
    registry.promote(mid, Stage.CANDIDATE)
    registry.promote(mid, Stage.SHADOW)
    with pytest.raises(PromotionError, match="human sign-off"):
        registry.promote(mid, Stage.PRODUCTION)
    with pytest.raises(PromotionError, match="shadow report"):
        registry.promote(mid, Stage.PRODUCTION, approved_by="ricky")
    registry.attach_shadow_report(mid, {"hypothetical_sharpe": 1.0})
    entry = registry.promote(mid, Stage.PRODUCTION, approved_by="ricky")
    assert entry["stage"] == "PRODUCTION"
    assert entry["approved_by"] == "ricky"


def test_single_production_per_name(registry):
    def to_production(entry):
        mid = entry["model_id"]
        registry.promote(mid, Stage.CANDIDATE)
        registry.promote(mid, Stage.SHADOW)
        registry.attach_shadow_report(mid, {"ok": 1})
        return registry.promote(mid, Stage.PRODUCTION, approved_by="ricky")

    e1 = to_production(register_dummy(registry))
    e2 = to_production(register_dummy(registry))
    assert registry.get(e1["model_id"])["stage"] == "RETIRED"
    assert registry.get(e2["model_id"])["stage"] == "PRODUCTION"
    assert registry.resolve("production:alpha")["model_id"] == e2["model_id"]


def test_retired_is_terminal(registry):
    e = register_dummy(registry)
    registry.promote(e["model_id"], Stage.RETIRED)
    with pytest.raises(PromotionError):
        registry.promote(e["model_id"], Stage.CANDIDATE)
