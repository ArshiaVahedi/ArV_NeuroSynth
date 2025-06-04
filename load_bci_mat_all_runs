def load_bci_mat_all_runs(filepath, num_channels=4, segment_length=256, batch_size=32):
    import scipy.io
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    mat = scipy.io.loadmat(filepath)
    data = mat['data']
    eeg_list = []
    for i in range(data.shape[1]):
        run_struct = data[0, i][0, 0]
        X = run_struct['X']
        eeg_list.append(X)
    eeg_all = np.concatenate(eeg_list, axis=0)
    eeg = eeg_all[:, :num_channels].T

    class EEGDataset(Dataset):
        def __init__(self, eeg_data, segment_length):
            self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            self.segment_length = segment_length

        def __len__(self):
            return self.eeg_data.shape[1] // self.segment_length

        def __getitem__(self, idx):
            start = idx * self.segment_length
            end = start + self.segment_length
            return self.eeg_data[:, start:end],

    dataset = EEGDataset(eeg, segment_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, eeg