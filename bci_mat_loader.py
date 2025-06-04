
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_bci_mat(filepath, num_channels=4, segment_length=256, batch_size=32):
    import scipy.io
    import torch
    from torch.utils.data import DataLoader

    mat = scipy.io.loadmat(filepath)
    data_struct = mat['data'][0, 0][0, 0]
    X = data_struct['X']  # shape (29683, 25)
    eeg = X[:, :num_channels].T  # shape (num_channels, time_steps)

    eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
    eeg_dataset = EEGDataset(eeg_tensor, segment_length)
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, eeg_tensor
