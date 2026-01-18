import os
import numpy as np
from tensorflow.keras.utils import Sequence
from pathlib import Path
class MidiDataset(Sequence):
    """
    Keras Sequence for loading MIDI tokenized sequences.
    """

    def __init__(self, tokens_path,  data_file, batch_size, max_seq_len, shuffle=True, **kwargs):
        """
        Initialize the dataset.
        tokens_path : path to folder containing dataset.npz and vocab.json
        batch_size  : number of sequences per batch
        max_seq_len : maximum sequence length (padding)
        shuffle     : whether to shuffle data each epoch
        """
        super().__init__(**kwargs)
        self.tokens_path = tokens_path
        self.data_file = data_file
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle

        # Load dataset
        self.data = self._load_dataset()
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_dataset(self):
        dataset_file = Path(self.tokens_path) / self.data_file
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"{dataset_file} not found.")
        
        loaded = np.load(dataset_file, allow_pickle=True)
        sequences = loaded["x"]

        return sequences
                        
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sequences = self.data[batch_indexes]
        X = batch_sequences[:, :-1]
        y = batch_sequences[:, 1:]
        return X, y

    def on_epoch_end(self):
        """
        Shuffle indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
