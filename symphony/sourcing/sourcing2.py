import argparse
import glob
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
from symphony.preproccesing.preprocessing import midi_to_notes, notes_to_midi

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download and process the maestro dataset")
    parser.add_argument("--colab", action='store_true', help="Use Colab data directory")
    parser.add_argument("--local", action='store_true', help="Use local data directory")
    parser.add_argument("--num_files", type=int, default=5, help="Number of MIDI files to use for training")
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=128, help="Vocabulary size")
    return parser.parse_args()

def download_maestro_dataset(colab=False):
    if colab:
        data_dir = pathlib.Path('/content/drive/MyDrive/Colab/data/maestro-v2.0.0')
    else:
        data_dir = pathlib.Path.home() / "data/maestro-v2.0.0"

    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
    else:
        print("Data directory already exists.")

def load_midi_files(data_dir, num_files):
    directory = str(f'{data_dir}/**/*.mid*')
    filenames = glob.glob(directory)

    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)
    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    return tf.data.Dataset.from_tensor_slices(train_notes)

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size=128) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    key_order = ['pitch', 'step', 'duration']

    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == "__main__":
    args = parse_arguments()

    if args.colab and args.local:
        raise ValueError("Please specify only one data source.")

    if args.colab:
        data_dir = '/content/drive/MyDrive/Colab/data/maestro-v2.0.0'
    elif args.local:
        data_dir = '/Users/juan-garassino/Code/le-wagon/symphony/data/maestro-v2.0.0'
    else:
        raise ValueError("Please specify a data source.")

    download_maestro_dataset(args.colab)

    num_files = args.num_files
    seq_length = args.seq_length
    vocab_size = args.vocab_size

    dataset = load_midi_files(data_dir, num_files)
    sequence_dataset = create_sequences(dataset, seq_length, vocab_size)
