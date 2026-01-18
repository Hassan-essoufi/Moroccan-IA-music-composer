import os
import sys
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import tokenizer

def test_build_vocab_and_encode():
    """
    Test the build_vocab, encode, and pad_sequences functions
    """
    # Example
    event_sequences = [
        ["note_on_60", "note_off_60", "time_shift_10"],
        ["note_on_62", "note_off_62", "velocity_3"]
    ]

    # Constructing the vocab
    token_to_id, id_to_token = tokenizer.build_vocab(event_sequences)
    assert isinstance(token_to_id, dict)
    assert isinstance(id_to_token, dict)
    assert len(token_to_id) == len(id_to_token)
    print("Vocabulary built successfully.")

    # Encoding vocab
    encoded = tokenizer.encode(event_sequences[0], token_to_id)
    assert all(isinstance(x, int) for x in encoded)
    print(f"Encoded sequence: {encoded}")

    # Padding
    padded = tokenizer.pad_sequences([encoded], max_len=5)
    assert padded.shape == (1, 5)
    print(f"Padded sequence: {padded}")

def test_save_and_load_vocab(tmp_dir="tmp_vocab_test"):
    """
    Test saving and loading the vocab
    """
    os.makedirs(tmp_dir, exist_ok=True)
    events = [
        ["note_on_60", "note_off_60"],
        ["note_on_61", "note_off_61"]
    ]
    token_to_id, id_to_token = tokenizer.build_vocab(events)
    
    tokenizer.save_vocab(token_to_id, tmp_dir)
    vocab_path = os.path.join(tmp_dir, "vocab.json")
    assert os.path.exists(vocab_path)

    with open(vocab_path, "r") as f:
        loaded_vocab = json.load(f)
    assert loaded_vocab == token_to_id
    print("Vocabulary saved and loaded successfully.")

def test_events_to_npz(tmp_dir="tmp_vocab_test",npz_file="tokens.npz"):
    """
    Test events_to_npz function
    """
    os.makedirs(tmp_dir, exist_ok=True)
    events = [
        ["note_on_60", "note_off_60", "time_shift_10"],
        ["note_on_62", "note_off_62", "velocity_3"]
    ]

    padded_sequences, token_to_id, id_to_token = tokenizer.events_to_npz(events, tmp_dir, npz_file, max_seq_len=10)
    npz_path = os.path.join(tmp_dir, npz_file)
    assert os.path.exists(npz_path)
    assert padded_sequences.shape[1] == 10
    print("Events converted to NPZ successfully.")

if __name__ == "__main__":
    test_build_vocab_and_encode()
    test_save_and_load_vocab()
    test_events_to_npz()
    print("All tokenizer tests passed!")
