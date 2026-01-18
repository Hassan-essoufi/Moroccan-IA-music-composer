import os
from src.utils import load_midi

def get_midi_duration(midi):
    """
    Return MIDI duration in seconds.
    """
    try:
        return midi.get_end_time()
    except Exception:
        return 0.0


def count_notes(midi):
    """
    Count the total of notes in a MIDI.
    """
    total = 0
    for instrument in midi.instruments:
        total += len(instrument.notes)
    return total


def is_valid_midi(midi_path):
    """
    Validating MIDI file based on duration & notes.
    """
    try:
        midi = load_midi(midi_path)
    except Exception as e:
        print(f"{os.path.basename(midi_path)} → {e}")
        return False

    duration = get_midi_duration(midi)
    if duration < 3.0:
        print(f"{os.path.basename(midi_path)} → duration {duration:.2f}s")
        return False

    n_notes = count_notes(midi)
    if n_notes < 1:
        print(f"{os.path.basename(midi_path)} → no notes found")
        return False

    return True


def validate_directory(input_dir, output_dir=None):
    """
    Validate all MIDI files in a directory.
    """

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    valid_count = 0
    total_count = 0

    for file in os.listdir(input_dir):
        if not (file.lower().endswith(".midi") or file.lower().endswith(".mid")):
            continue

        total_count += 1
        src_path = os.path.join(input_dir, file)

        if is_valid_midi(src_path):
            valid_count += 1

            if output_dir is not None:
                dst_path = os.path.join(output_dir, file)
                try:
                    with open(src_path, "rb") as f_src, open(dst_path, "wb") as f_dst:
                        f_dst.write(f_src.read())
                except Exception as e:
                    print(f"failed to copy {file}: {e}")

    return f"{valid_count} / {total_count}"
