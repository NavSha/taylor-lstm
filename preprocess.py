import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LyricsDataset(Dataset):
    def __init__(self, text, seq_length=100):
        self.seq_length = seq_length

        # Build vocabulary
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # Encode full text
        self.encoded = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int64)

    def __len__(self):
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.encoded[idx : idx + self.seq_length])
        y = torch.from_numpy(self.encoded[idx + 1 : idx + self.seq_length + 1])
        return x, y

    def save_vocab(self, path):
        vocab = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": {str(k): v for k, v in self.idx_to_char.items()},
        }
        with open(path, "w") as f:
            json.dump(vocab, f)

    @staticmethod
    def load_vocab(path):
        with open(path) as f:
            vocab = json.load(f)
        char_to_idx = vocab["char_to_idx"]
        idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
        return char_to_idx, idx_to_char


def get_dataloader(text_path, seq_length=100, batch_size=64, shuffle=True):
    with open(text_path, "r") as f:
        text = f.read()

    dataset = LyricsDataset(text, seq_length=seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataset, loader
