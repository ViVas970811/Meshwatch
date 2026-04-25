"""Tests for ``fraud_detection.training.callbacks``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from fraud_detection.training.callbacks import EarlyStopping, ModelCheckpoint


def test_early_stopping_does_not_trigger_on_improvement():
    es = EarlyStopping(patience=3)
    for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        stop = es.step(score, epoch=i)
        assert not stop
    assert es.best_score == 0.5
    assert es.best_epoch == 4
    assert es.counter == 0


def test_early_stopping_triggers_after_patience_exhausted():
    es = EarlyStopping(patience=2)
    es.step(0.5, epoch=0)  # best
    assert not es.step(0.4, epoch=1)  # 1 bad
    assert es.step(0.4, epoch=2)  # 2 bad -> stop
    assert es.should_stop


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=2, mode="min")  # for loss
    es.step(1.0, epoch=0)
    es.step(0.8, epoch=1)  # improved
    assert not es.step(0.9, epoch=2)
    assert es.step(0.85, epoch=3)
    assert es.should_stop


def test_early_stopping_min_delta_blocks_tiny_changes():
    es = EarlyStopping(patience=2, mode="max", min_delta=0.01)
    es.step(0.5, epoch=0)
    # Improvements of 0.001 each don't count.
    assert not es.step(0.501, epoch=1)
    assert es.step(0.502, epoch=2)
    assert es.should_stop


def test_early_stopping_reset():
    es = EarlyStopping(patience=2)
    es.step(0.5, epoch=0)
    es.step(0.4, epoch=1)
    es.reset()
    assert es.best_score is None
    assert es.counter == 0
    assert not es.should_stop


def test_checkpoint_first_step_saves():
    model = nn.Linear(3, 1)
    cp = ModelCheckpoint()
    assert cp.step(0.5, epoch=0, model=model)
    assert cp.best_score == 0.5
    assert cp.best_epoch == 0
    assert cp.best_state_dict is not None


def test_checkpoint_only_saves_on_improvement():
    model = nn.Linear(3, 1)
    cp = ModelCheckpoint(mode="max")
    cp.step(0.5, epoch=0, model=model)
    # New step with WORSE score should not update.
    assert not cp.step(0.4, epoch=1, model=model)
    assert cp.best_epoch == 0
    # Better score should.
    assert cp.step(0.6, epoch=2, model=model)
    assert cp.best_epoch == 2
    assert cp.best_score == 0.6


def test_checkpoint_restore_loads_state(tmp_path: Path):
    model = nn.Linear(3, 1)
    original = {k: v.clone() for k, v in model.state_dict().items()}

    cp = ModelCheckpoint()
    cp.step(0.5, epoch=0, model=model)

    # Mutate the model -- weights should differ from snapshot.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))

    after_mutation = {k: v.clone() for k, v in model.state_dict().items()}
    cp.restore(model)
    restored = model.state_dict()

    for k, v in original.items():
        torch.testing.assert_close(restored[k], v)
    # Sanity: it differs from the mutated state we just had.
    for k in original:
        assert not torch.equal(restored[k], after_mutation[k])


def test_checkpoint_writes_to_disk_when_path_given(tmp_path: Path):
    model = nn.Linear(3, 1)
    path = tmp_path / "ckpt.pt"
    cp = ModelCheckpoint(path=path)
    cp.step(0.5, epoch=0, model=model)
    assert path.exists()
    payload = torch.load(path, weights_only=False)
    assert payload["epoch"] == 0
    assert payload["score"] == 0.5
    assert "state_dict" in payload


def test_checkpoint_min_mode():
    model = nn.Linear(3, 1)
    cp = ModelCheckpoint(monitor="val_loss", mode="min")
    cp.step(1.0, epoch=0, model=model)
    assert not cp.step(1.1, epoch=1, model=model)
    assert cp.step(0.9, epoch=2, model=model)
