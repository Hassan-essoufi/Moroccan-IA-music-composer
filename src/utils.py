import os
import random
import shutil
import json
import yaml
import pretty_midi
from pathlib import Path

def load_midi(midi_path):
    """
    Loading MIDI file.
    """

    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"midi file not found: {midi_path}")

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        raise ValueError(f"failed to load midi: {e}")

    return midi

def load_events(input_path):
    """
    Load event sequence from JSON file.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    except Exception as e:
        raise ValueError(f"failed to load events: {e}")

    return events

def load_config(path):
    """
    Loading config
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"config file is empty: {path}")

        return config

    except Exception as e:
        raise ValueError(f"Error in loading config {path}: {e}")

def load_vocab(vocab_path):
    """
    Load vocabulary(json file)
    """
    try:
        with open(vocab_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load vocab from {vocab_path}: {e}")
    
def split_midi_dataset(
    dataset_dir,train_dir,
    val_dir

):
    """
    Split a MIDI dataset into train, val.

    """
    train_ratio=0.8
    seed=42
    copy_files=True
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.exists(), "Dataset directory does not exist"

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    midi_files = list(dataset_dir.glob("*.mid")) + list(dataset_dir.glob("*.midi"))
    assert len(midi_files) > 0, "No MIDI files found"

    random.seed(seed)
    random.shuffle(midi_files)

    split_idx = int(len(midi_files) * train_ratio)
    train_files = midi_files[:split_idx]
    val_files = midi_files[split_idx:]

    for f in train_files:
        target = train_dir / f.name
        if copy_files:
            shutil.copy2(f, target)
        else:
            shutil.move(f, target)

    for f in val_files:
        target = val_dir / f.name
        if copy_files:
            shutil.copy2(f, target)
        else:
            shutil.move(f, target)

    print(f"Train: {len(train_files)} files")
    print(f"Val  : {len(val_files)} files")


