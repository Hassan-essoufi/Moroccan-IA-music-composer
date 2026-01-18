import os
import sys 
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.midi_dataset import MidiDataset
from src.preprocessing import tokenizer

def create_dummy_dataset(tmp_dir="tmp_dataset_test",npz_file="dataset.npz", num_sequences=10, max_len=10):
    """
    Create a dataset to test MidiDataset
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Simulated events
    event_sequences = [
        [f"note_on_{60+i}", f"note_off_{60+i}", "time_shift_10"] for i in range(num_sequences)]

    # Generate dataset & vocab
    padded_sequences, token_to_id, id_to_token = tokenizer.events_to_npz(event_sequences, tmp_dir, npz_file,  max_seq_len=max_len)

    # save vocab
    tokenizer.save_vocab(token_to_id, tmp_dir)

    return tmp_dir

def test_midi_dataset():
    tmp_dir = create_dummy_dataset()
    batch_size = 4
    max_seq_len = 10

    dataset = MidiDataset(tokens_path=tmp_dir,
                          data_file="dataset.npz",
                          batch_size=batch_size,
                          max_seq_len=max_seq_len,
                          shuffle=True)

    # Verify dataset
    expected_len = int(np.ceil(10 / batch_size))
    assert len(dataset) == expected_len
    print(f"Dataset length OK: {len(dataset)} batches")

    X, y = dataset[0]
    assert X.shape[0] <= batch_size
    assert X.shape[1] == max_seq_len - 1  
    assert y.shape[1] == max_seq_len - 1 
    print(f"Batch shapes OK: X={X.shape}, y={y.shape}")

    idx_before = dataset.indexes.copy()
    dataset.on_epoch_end()
    idx_after = dataset.indexes
    assert not np.array_equal(idx_before, idx_after)  
    print("Shuffle on epoch end OK")

if __name__ == "__main__":
    test_midi_dataset()
    print("All MidiDataset tests passed!")
