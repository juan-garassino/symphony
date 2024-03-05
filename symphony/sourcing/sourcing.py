import pathlib
import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import pretty_midi
import pathlib
from symphony.preproccesing.preprocessing import midi_to_notes

def download_maestro_dataset(colab=False):
    if colab:
        data_dir = pathlib.Path('/content/symphony/data/maestro-v2.0.0')  # Change this for your Colab directory
    else:
        data_dir = pathlib.Path.home() / "../data/maestro-v2.0.0"  # Assumes a local directory within the user's home directory

    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
        print(f'downloaded maestro-v2.0.0-midi.zip and extracted it to data directory. @ {data_dir}')
    else:
        print("Data directory already exists.")

def load_midi_files(data_dir, num_files):
    directory = str(f'{data_dir}/**/*.mid*')
    print(dir)
    filenames = glob.glob(directory)
    print('Number of files:', len(filenames))

    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)

    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    return tf.data.Dataset.from_tensor_slices(train_notes)

import tensorflow as tf

def create_sequences(dataset: tf.data.Dataset, seq_length: int, num_samples: int, vocab_size=128) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    key_order = ['pitch', 'step', 'duration']

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.take(num_samples).map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)



if __name__ == "__main__":

    data_dir = '../../data/maestro-v2.0.0'

    download_maestro_dataset(data_dir)

    data_dir = '/content/symphony/data/maestro-v2.0.0' # CHANGE THIS

    # Example usage:
    num_files = 5
    seq_length = 100
    vocab_size = 128

    dataset = load_midi_files(data_dir, num_files)
    sequence_dataset = create_sequences(dataset, seq_length, vocab_size)
