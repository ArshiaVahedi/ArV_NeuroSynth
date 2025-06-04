# ArV_NeuroSynth
combined models and statistical methods for enhancement GANs stimulation for EEG functional connectivity analysis

# How this supports the review

This notebook directly supports the review by providing a practical implementation of the ArV_NeuroSynth model discussed in the article. It offers concrete examples of how GANs can be applied to EEG data, demonstrating their potential for data augmentation, denoising, and preserving functional connectivity. By showcasing the code and its applications, it validates the claims made in the review and provides researchers with a tangible resource for exploring GANs in EEG analysis.

## Usage

1. Place your BCI .mat file (e.g., A01T.mat) in the project folder.
2. In your notebook, use:
    ```python
    from bci_mat_loader import load_bci_mat_all_runs
    dataloader, eeg_tensor = load_bci_mat_all_runs("A01T.mat")
    ```
3. Train your model using the provided ArV_NeuroSynth class.
4. Save and load models/results using the provided utility functions.