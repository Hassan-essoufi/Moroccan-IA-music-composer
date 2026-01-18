import os
from pathlib import Path
import numpy as np
import midi_neural_processor.processor as midi_tokenizer

def encode_midi_task(midi_path):
    """Tokenizing a MIDI file."""
    return midi_tokenizer.encode_midi(midi_path)


def save_to_npz(input_dir, output_dir, npz_file, max_seq_len=2048):
    """
    Tokenize all MIDI files in input_dir and save to a .npz file.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_dir = Path(output_dir)
    npz_file = Path(npz_file)
    midi_files = [f for f in os.listdir(input_dir) if f.endswith(".mid") or f.endswith(".midi")]

    all_tokens = []
    file_names = []

    for midi_name in midi_files:
        midi_path = os.path.join(input_dir, midi_name)
        try:
            tokens = encode_midi_task(midi_path)

            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]

            elif len(tokens) < max_seq_len:
                tokens += [0] * (max_seq_len - len(tokens))  # 0 = PAD token

            all_tokens.append(tokens)
            file_names.append(midi_name)
        except Exception as e:
            print(f"Skipping {midi_name}: {e}")

    # Convert to numpy array
    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Save npz
    output_path = output_dir/ npz_file
    np.savez_compressed(output_path, x=all_tokens, file_names=np.array(file_names))
    print(f"Saved {len(all_tokens)} sequences to {output_path}")
