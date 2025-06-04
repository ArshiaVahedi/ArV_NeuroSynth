
import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_bci_mat(filepath, num_channels=4, segment_length=256, batch_size=32):
    mat = scipy.io.loadmat(filepath)

    # Identify the main data structure key
    data = None
    for key in mat:
        if isinstance(mat[key], np.ndarray) and mat[key].dtype.names:
            struct = mat[key][0, 0]
            for name in struct.dtype.names:
                if 'EEG' in name or 'data' in name.lower():
                    data = struct[name]
                    break
            if data is not None:
                break

    if data is None:
        raise ValueError("Could not locate EEG data structure. Check the .mat file keys manually.")

    print(f"Loaded EEG shape: {data.shape}")
    eeg = data[:num_channels, :]

    # Segment into non-overlapping windows
    segments = []
    for i in range(0, eeg.shape[1] - segment_length, segment_length):
        segments.append(eeg[:, i:i+segment_length])

    eeg_tensor = torch.tensor(np.stack(segments), dtype=torch.float32)  # [N, C, T]
    dataloader = DataLoader(TensorDataset(eeg_tensor), batch_size=batch_size, shuffle=True)
    return dataloader, eeg_tensor
