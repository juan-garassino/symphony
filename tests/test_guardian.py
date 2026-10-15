"""Regression tests added by the guardian run (CPU only, no dataset needed)."""
import os
import subprocess
import sys

import numpy as np
import pretty_midi
import pytest
import setuptools
import tensorflow as tf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _notes_ds(n):
    notes = np.stack([np.arange(n) % 128, np.ones(n), np.ones(n)], axis=1).astype("float64")
    return tf.data.Dataset.from_tensor_slices(notes)


def test_create_sequences_as_called_by_train_keeps_all_windows():
    # train.py calls create_sequences(notes_ds, seq_length, vocab_size); vocab_size must
    # not be silently used as a sample cap (that truncated training to 128 sequences).
    from symphony.sourcing.sourcing import create_sequences

    n, seq_length, vocab_size = 300, 25, 128
    seq_ds = create_sequences(_notes_ds(n), seq_length, vocab_size)
    count = sum(1 for _ in seq_ds)
    assert count == n - seq_length
    inputs, labels = next(iter(seq_ds))
    assert inputs.shape == (seq_length, 3)
    assert float(inputs[1, 0]) == pytest.approx(1 / vocab_size)
    assert set(labels) == {"pitch", "step", "duration"}


def test_create_sequences_num_samples_cap():
    from symphony.sourcing.sourcing import create_sequences

    seq_ds = create_sequences(_notes_ds(300), 25, num_samples=10)
    assert sum(1 for _ in seq_ds) == 10


def test_setup_finds_packages():
    pkgs = setuptools.find_packages(REPO, exclude=["tests"])
    for name in ("symphony", "symphony.model", "symphony.model.loss",
                 "symphony.sourcing", "symphony.preproccesing", "symphony.manager"):
        assert name in pkgs


def test_mymodel_loss_penalises_underprediction():
    from symphony.model import model as model_mod
    from symphony.model.loss.loss import mse_with_positive_pressure

    y_true, y_pred = tf.constant([1.0]), tf.constant([0.0])
    assert float(model_mod.mse_with_positive_pressure(y_true, y_pred)) > 0
    assert model_mod.mse_with_positive_pressure is mse_with_positive_pressure


def test_midi_to_notes_roundtrip_and_empty(tmp_path):
    from symphony.preproccesing.preprocessing import midi_to_notes

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=0.5, end=1.0))
    pm.instruments.append(inst)
    path = tmp_path / "a.mid"
    pm.write(str(path))
    df = midi_to_notes(str(path))
    assert list(df["pitch"]) == [60, 62]
    assert df["step"].iloc[1] == pytest.approx(0.5, abs=1e-2)

    empty = pretty_midi.PrettyMIDI()
    empty.instruments.append(pretty_midi.Instrument(program=0))
    epath = tmp_path / "empty.mid"
    empty.write(str(epath))
    with pytest.raises(ValueError):
        midi_to_notes(str(epath))


def test_train2_help_exits_cleanly():
    # A bare `except:` caught argparse's SystemExit and tried to start ipdb.
    proc = subprocess.run([sys.executable, "-m", "symphony.train2", "--help"],
                          cwd=REPO, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--data_dir" in proc.stdout
