import argparse
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import pathlib
from symphony.model.loss.loss import mse_with_positive_pressure
from symphony.model.generate import predict_next_note
from symphony.sourcing.sourcing import load_midi_files, create_sequences
from symphony.preproccesing.preprocessing import midi_to_notes, notes_to_midi


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the neural network")
    parser.add_argument("--data_dir", type=str, default='/Users/juan-garassino/Code/le-wagon/symphony/data/maestro-v2.0.0',
                        help="Directory containing MIDI files")
    parser.add_argument("--num_files", type=int, default=5,
                        help="Number of MIDI files to use for training")
    parser.add_argument("--seq_length", type=int, default=25,
                        help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=128,
                        help="Vocabulary size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    return parser.parse_args()

def train_model(train_ds, model, callbacks, epochs):
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

def main():
    args = parse_arguments()

    print(args.data_dir, args.num_files)

    #filenames = load_midi_files(args.data_dir, args.num_files)
    filenames = glob.glob(str(f'{args.data_dir}/**/*.mid*'))

    all_notes = []
    key_order = ['pitch', 'step', 'duration']

    # print(filenames)

    for f in filenames:
        # print(f)
        notes = midi_to_notes(f)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)

    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    seq_ds = create_sequences(notes_ds, args.seq_length, args.vocab_size)

    buffer_size = n_notes - args.seq_length

    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(args.batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    input_shape = (args.seq_length, 3)
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    x = layers.LSTM(128)(inputs)

    outputs = {
      'pitch': layers.Dense(128, name='pitch')(x),
      'step': layers.Dense(1, name='step')(x),
      'duration': layers.Dense(1, name='duration')(x),
    }

    model = Model(inputs, outputs)

    loss = {
          'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),
          'step': mse_with_positive_pressure,
          'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)

    model.summary()

    losses = model.evaluate(train_ds, return_dict=True)
    print(losses)

    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration':1.0,
        },
        optimizer=optimizer,
    )

    model.evaluate(train_ds, return_dict=True)

    history = train_model(train_ds, model, callbacks, args.epochs)

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()

    temperature = 2.0
    num_predictions = 120

    sample_file = filenames[1]

    raw_notes = midi_to_notes(sample_file)
    raw_notes.head()

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    input_notes = (
        sample_notes[:args.seq_length] / np.array([args.vocab_size, 1, 1]))

    generated_notes = []

    prev_start = 0

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    instrument_name = 'Acoustic Grand Piano'

    out_file = 'output.mid'
    # out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)
    # AudioManager.display_audio(out_pm)

if __name__ == "__main__":
    main()
