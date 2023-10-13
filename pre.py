import os
import json
import music21 as m21
import numpy as np
import tensorflow as tf
from tensorflow import keras


DATASET_PATH = "deutschl/dva"
SAVE_DIR = "Save"
SINGLE_FILE = "Files/file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "Files/mapping.json"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

def load_songs(dataset_path):
    songs = []
    for path,subdirs,files in os.walk(dataset_path):
        for file in files:
            if file[-3:] =="krn":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)
    return songs

def acceptable_duration(song,duration):
    pass
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in duration:
            return False
    return True

def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    if not isinstance(key,m21.key.Key):
        key = song.analyze("key")

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))

    transposed_song = song.transpose(interval)
    return transposed_song

def encode(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event , m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event,m21.note.Rest):
            symbol = "r"

        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step ==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    encoded_song = " ".join(map(str,encoded_song))
    return encoded_song


def process(dataset_path):
    pass
    print("loading songs...")
    songs = load_songs(dataset_path)
    print(f"Loaded {len(songs)} songs")

    for i, song in enumerate(songs):
        if not acceptable_duration(song,ACCEPTABLE_DURATIONS):
            continue
        song = transpose(song)

        encoded_song = encode(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def single_dataset(dataset_path, file_dataset_path,sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song +" " + new_song_delimiter
    songs = songs[:-1]

    with open(file_dataset_path,"w") as fp:
        fp.write(songs)

    return songs

def create_map(songs,mapping_path):
    mappings = {}
    songs = songs.split()
    vocab = list(set(songs))
    for i,symbol in enumerate(vocab):
        mappings[symbol] = i

    with open(mapping_path,"w") as fp:
        json.dump(mappings,fp, indent=4)

def convert_to_int(songs):
    int_songs = []
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    songs = songs.split()
    for symbol in songs:
        int_songs.append((mappings[symbol]))

    return int_songs

def generating_training_sequences(sequence_length):
    songs = load(SINGLE_FILE)
    int_songs = convert_to_int(songs)
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append((int_songs[i+sequence_length]))
    vocab_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocab_size)
    targets = np.array(targets)

    return inputs,targets







def main():
    process(DATASET_PATH)
    songs = single_dataset(SAVE_DIR, SINGLE_FILE, SEQUENCE_LENGTH)
    create_map(songs, MAPPING_PATH)
    inputs, targets = generating_training_sequences(SEQUENCE_LENGTH)
    a = 1




if __name__=="__main__":
    main()

