# ArV_NeuroSynth
combined models and statistical methods for enhancement GANs stimulation for EEG functional connectivity analysis

## Usage

1. Place your BCI .mat file (e.g., A01T.mat) in the project folder.
2. In your notebook, use:
    ```python
    from bci_mat_loader import load_bci_mat_all_runs
    dataloader, eeg_tensor = load_bci_mat_all_runs("A01T.mat")
    ```
3. Train your model using the provided ArV_NeuroSynth class.
4. Save and load models/results using the provided utility functions.