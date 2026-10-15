# symphony

LSTM that learns pitch, step and duration from piano MIDI and generates new note sequences, written in TensorFlow/Keras.

## What and why

`symphony` is a small experiment in symbolic music generation. It reads MIDI files (the code is written for the [MAESTRO v2.0.0](https://magenta.tensorflow.org/datasets/maestro) piano dataset), turns each file into a table of notes, and trains a single-layer LSTM to predict the next note from the previous 25. Generation samples one note at a time, feeds it back in, and writes the result to a `.mid` file. The approach follows the TensorFlow "Generate music with an RNN" tutorial. It is an older experiment, not a maintained library.

## Highlights

- Notes are represented as `(pitch, step, duration)`, where `step` is the time since the previous note start ([`preprocessing.py`](symphony/preproccesing/preprocessing.py)).
- One LSTM(128) with three heads: a 128-way pitch logit layer and two regression heads for step and duration ([`model.py`](symphony/model/model.py)).
- Custom `mse_with_positive_pressure` loss penalises negative step/duration predictions ([`loss.py`](symphony/model/loss/loss.py)).
- Temperature-controlled sampling for pitch; step and duration are clamped to be non-negative ([`generate.py`](symphony/model/generate.py)).
- `AudioManager` plots piano rolls and note distributions, and renders MIDI to audio in a notebook via FluidSynth ([`manager.py`](symphony/manager/manager.py)).

## Quickstart

Requirements: Python 3.11 (see `.python-version`), TensorFlow, `pretty_midi`, pandas, numpy, matplotlib, seaborn, IPython. For audio playback, `pyfluidsynth` plus the FluidSynth system library. Training needs the MAESTRO MIDI files, which are not in the repo.

```bash
git clone https://github.com/juan-garassino/symphony && cd symphony
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ipython
```

Run everything from the repo root. The package has no `__init__.py` files, so `pip install -e .` installs no modules; imports only work from the repo root.

Download MAESTRO v2.0.0 (MIDI only), then:

```bash
python -m symphony.train2 --data_dir path/to/maestro-v2.0.0 --num_files 5 --epochs 3
```

`train2.py` is the argparse version; it trains, saves checkpoints to `./training_checkpoints/`, and writes `output.mid`. `train.py` is the earlier script version with a hardcoded local path (`data_dir`) that you must edit.

Verified in a CPU sandbox: dependencies install, the model builds with outputs `pitch`, `step`, `duration`, and `notes_to_midi` writes MIDI. Not verified: a full training run (needs the MAESTRO download).

## Project structure

```
symphony/
  train.py                  original training + generation script (hardcoded data path)
  train2.py                 argparse version of the same pipeline
  model/model.py            LSTM with pitch / step / duration heads
  model/generate.py         predict_next_note() with temperature sampling
  model/loss/loss.py        positive-pressure MSE
  preproccesing/            midi_to_notes(), notes_to_midi()
  sourcing/                 MAESTRO download, MIDI loading, sequence windows (sourcing2.py = CLI variant)
  manager/manager.py        plots and audio playback helpers
notebooks/test.ipynb        scratch notebook, no saved results
requirements.txt, setup.py
```

## Results

No metrics or evaluation exist in the repo. The only artifact is [`output.mid`](output.mid), a generated clip: 120 notes, about 1.6 s long (read with `pretty_midi`). Checkpoints are in `training_checkpoints/` (weights only).

## Tech stack

Python, TensorFlow/Keras, pretty_midi, pandas, NumPy, matplotlib, seaborn.

## Status

Old experiment (commit history is a handful of snapshots). Known rough edges: hardcoded paths, missing `__init__.py`, duplicated scripts (`train`/`train2`, `sourcing`/`sourcing2`), no tests, no evaluation. No license file is present.
