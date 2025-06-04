"""
ArV_NeuroSynth: An Auto-regressive Variational Model for EEG Synthesis

This module implements the ArV_NeuroSynth model, which combines variational autoencoding
and adversarial training for EEG data generation and analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Utility Functions
# --------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(real_samples.device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# --------------------
# Model Components
# --------------------
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.GRU(32, 64, batch_first=True)
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        _, h = self.rnn(x)
        h = h[-1]
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, seq_length):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64)
        self.rnn = nn.GRU(64, 32, batch_first=True)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, output_channels, kernel_size=4, stride=2, padding=1),
        )
        self.seq_length = seq_length

    def forward(self, z):
        z = self.fc(z).unsqueeze(1).repeat(1, self.seq_length // 4, 1)
        out, _ = self.rnn(z)
        out = out.permute(0, 2, 1)
        return self.deconv(out)

class LatentGenerator(nn.Module):
    def __init__(self, noise_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z):
        return self.model(z)

class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.model(z)

# --------------------
# Main Model
# --------------------
class ArV_NeuroSynth(nn.Module):
    """
    ArV_NeuroSynth Model

    An auto-regressive variational model for EEG synthesis, featuring an encoder, decoder,
    latent generator, and latent discriminator.
    """
    model_name = "ArV_NeuroSynth"
    def __init__(self, input_channels=4, seq_length=256, latent_dim=32, noise_dim=50):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels, seq_length)
        self.G_latent = LatentGenerator(noise_dim, latent_dim)
        self.D_latent = LatentDiscriminator(latent_dim)
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

    def forward(self, x=None, mode='vae', noise_for_g_latent=None):
        if mode == 'vae':
            mu, logvar = self.encoder(x)
            z = reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar, z

        elif mode == 'encode_to_latent':
            mu, logvar = self.encoder(x)
            z = reparameterize(mu, logvar)
            return z, mu, logvar

        elif mode == 'generate_latent_from_noise':
            return self.G_latent(noise_for_g_latent)

        elif mode == 'generate_eeg_from_noise':
            z = self.G_latent(noise_for_g_latent)
            return self.decoder(z), z

        elif mode == 'discriminate_latent':
            return self.D_latent(x)

        else:
            raise ValueError(f"Invalid mode: {mode}")
