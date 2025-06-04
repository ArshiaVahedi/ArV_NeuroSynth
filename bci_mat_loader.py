import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, eeg_data, segment_length):
        self.eeg_data = eeg_data
        self.segment_length = segment_length
        if not isinstance(self.eeg_data, torch.Tensor):
            self.eeg_data = torch.tensor(self.eeg_data, dtype=torch.float32)

    def __len__(self):
        return self.eeg_data.shape[1] // self.segment_length

    def __getitem__(self, idx):
        start_idx = idx * self.segment_length
        end_idx = start_idx + self.segment_length
        segment = self.eeg_data[:, start_idx:end_idx]
        return segment,

def load_bci_mat(filepath, num_channels=4, segment_length=256, batch_size=32):
    mat = scipy.io.loadmat(filepath)
    X = mat['data'][0, 0][0, 0]['X']  # shape (29683, 25)
    eeg = X[:, :num_channels].T       # shape (num_channels, time_steps)
    eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
    eeg_dataset = EEGDataset(eeg_tensor, segment_length)
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, eeg_tensor
