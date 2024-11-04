import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dismix_loss import ELBOLoss, BarlowTwinsLoss


class Conv1DEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, norm_layer, activation):
        super(Conv1DEncoder, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.norm = norm_layer(output_channels) if norm_layer else None
        self.activation = activation() if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = x.transpose(1, 2)  # Change shape from (batch, channels, sequence) to (batch, sequence, channels)
            x = self.norm(x)        # Apply LayerNorm to the last dimension (channels)
            x = x.transpose(1, 2)  # Change back shape to (batch, channels, sequence)
        if self.activation:
            x = self.activation(x)
        return x



class MixtureQueryEncoder(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=768,
        output_dim=64,
    ):
        super(MixtureQueryEncoder, self).__init__()
        self.encoder_layers = nn.Sequential(
            Conv1DEncoder(input_dim, hidden_dim, 3, 1, 0, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 4, 2, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, hidden_dim, 3, 1, 1, nn.LayerNorm, nn.ReLU),
            Conv1DEncoder(hidden_dim, output_dim, 1, 1, 1, None, None)
        )

    def forward(self, x):
        x = self.encoder_layers(x)
        return torch.mean(x, dim=-1)  # Mean pooling along the temporal dimension




class StochasticBinarizationLayer(nn.Module):
    def __init__(self):
        super(StochasticBinarizationLayer, self).__init__()
    
    def forward(self, logits):
        """
        Forward pass of the stochastic binarization layer.
        
        Parameters:
        - logits: The raw outputs from the pitch encoder (before sigmoid activation).
        
        Returns:
        - binarized_output: Binary tensor after stochastic binarization.
        """
        
        probabilities = torch.sigmoid(logits) # Apply sigmoid to get probabilities
        h = torch.rand_like(probabilities) # Sample a uniform random threshold for binarization
        binarized_output = (probabilities > h).float() # Binarize based on the threshold h
        
        return binarized_output # Return the binarized output with a straight-through estimator



class TimbreEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        hidden_dim=256, 
        output_dim=64  # Latent space dimension for timbre
    ):
        super(TimbreEncoder, self).__init__()

        # Shared architecture with Eφν (PitchEncoder)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Gaussian parameterization layers
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mean, logvar):
        """Reparameterization trick to sample from N(mean, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, em, eq):
        # Concatenate the mixture and query embeddings
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate along feature dimension
        
        # Shared layers forward pass
        hidden_state = self.shared_layers(concat_input)  # Shared hidden state output

        # Calculate mean and log variance
        mean = self.mean_layer(hidden_state)  # Mean of the Gaussian distribution
        logvar = self.logvar_layer(hidden_state)  # Log variance of the Gaussian distribution

        # Sample the timbre latent using the reparameterization trick
        timbre_latent = self.reparameterize(mean, logvar)
        
        return timbre_latent, mean, logvar




# Pitch Encoder Implementation from Table 5
class PitchEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=128, 
        hidden_dim=256, 
        pitch_classes=52, # true labels not 0-51
        output_dim=64
    ):
        super(PitchEncoder, self).__init__()

        # Transcriber: Linear layers for pitch classification
        self.transcriber = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pitch_classes)  # Output logits for pitch classification
            # No Sigmoid here, SB layer will handle it
        )

        # Stochastic Binarization (SB) Layer: Converts pitch logits to a binary representation
        self.sb_layer = StochasticBinarizationLayer()

        # Projection Layer: Project the binarized pitch representation to the latent space
        self.fc_proj = nn.Sequential(
            nn.Linear(pitch_classes, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )


    def forward(self, em, eq):
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate em and eq
        pitch_logits = self.transcriber(concat_input)
        y_bin = self.sb_layer(pitch_logits)  # Apply binarisation
        pitch_latent = self.fc_proj(y_bin)
        return pitch_latent, pitch_logits
    
    

class FiLM(nn.Module):
    def __init__(self, pitch_dim, timbre_dim):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(timbre_dim, pitch_dim)
        self.shift = nn.Linear(timbre_dim, pitch_dim)
    
    def forward(self, pitch_latent, timbre_latent):
        scale = self.scale(timbre_latent)
        shift = self.shift(timbre_latent)
        return scale * pitch_latent + shift


class DisMixDecoder(nn.Module):
    def __init__(
        self, 
        pitch_dim=64, 
        timbre_dim=64, 
        gru_hidden_dim=256, 
        output_dim=128, 
        num_frames=32, #10,
        num_layers=2
    ):
        super(DisMixDecoder, self).__init__()
        self.num_frames = num_frames
        
        self.film = FiLM(pitch_dim, timbre_dim)
        self.gru = nn.GRU(input_size=pitch_dim, hidden_size=gru_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(gru_hidden_dim * 2, output_dim)  # Bi-directional GRU output dimension is doubled
        self.output_transform = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=1),  # Convert back to 1D waveform or spectrogram
        )
    
    def forward(self, pitch_latents, timbre_latents):
        # FiLM layer: modulates pitch latents based on timbre latents
        source_latents = self.film(pitch_latents, timbre_latents)
        source_latents = source_latents.unsqueeze(1).repeat(1, self.num_frames, 1) # Expand source_latents along time axis if necessary
        
        output, _ = self.gru(source_latents)
        output = self.linear(output).transpose(1, 2)
        output = self.output_transform(output)
        
        return output # reconstructed spectrogram



class DisMixModel(pl.LightningModule):
    def __init__(
        self, 
        input_dim=128, 
        latent_dim=64, 
        hidden_dim=256, 
        gru_hidden_dim=256,
        num_frames=32, #10,
        pitch_classes=52,
        output_dim=128,
        learning_rate=4e-4,
        num_layers=2,
        clip_value=0.5,
    ):
        super(DisMixModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pitch_classes = pitch_classes
        self.clip_value = clip_value

        # Model components
        self.mixture_encoder = MixtureQueryEncoder(
            input_dim=input_dim,
            hidden_dim=768,
            output_dim=latent_dim,
        )
        self.query_encoder = MixtureQueryEncoder(
            input_dim=input_dim,
            hidden_dim=768,
            output_dim=latent_dim,
        )
        self.pitch_encoder = PitchEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            pitch_classes=pitch_classes,
            output_dim=latent_dim,
        )
        self.timbre_encoder = TimbreEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=latent_dim,
        )
        self.decoder = DisMixDecoder(
            pitch_dim=latent_dim, 
            timbre_dim=latent_dim, 
            gru_hidden_dim=gru_hidden_dim, 
            output_dim=output_dim, 
            num_frames=num_frames,
            num_layers=num_layers,
        )

        # Loss functions
        self.elbo_loss_fn = ELBOLoss() # For ELBO
        self.ce_loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()  # For pitch supervision
        self.bt_loss_fn = BarlowTwinsLoss() # Barlow Twins
        
        # Pitch Priors
        self.pitch_prior = self.pitch_encoder.fc_proj
        

    def forward(self, mixture, query):
        # Encode mixture and query
        em = self.mixture_encoder(mixture)
        eq = self.query_encoder(query)

        # Encode pitch and timbre latents
        pitch_latent, pitch_logits = self.pitch_encoder(em, eq)
        timbre_latent, timbre_mean, timbre_logvar = self.timbre_encoder(em, eq)
        
        # Decode to reconstruct the mixture
        reconstructed_spectrogram = self.decoder(pitch_latent, timbre_latent)
        return reconstructed_spectrogram, pitch_latent, pitch_logits, timbre_latent, timbre_mean, timbre_logvar, eq


    def training_step(self, batch, batch_idx):
        spec, note_tensors, pitch_annotation, _ = batch
        batch_size = spec.size(0)  # Extract batch size
        note_tensors, pitch_annotation = torch.stack(note_tensors), torch.stack(pitch_annotation)
        
        # Forward pass
        rec_mixture, pitch_latent, pitch_logits, timbre_latent, \
            timbre_mean, timbre_logvar, eq = self(spec, note_tensors)

        # Get pitch priors
        ohe_pitch_annotation = F.one_hot(pitch_annotation, num_classes=self.pitch_classes).float()
        pitch_priors = self.pitch_prior(ohe_pitch_annotation)
        
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            spec, rec_mixture,
            timbre_mean, timbre_logvar,
            pitch_latent, pitch_priors,
        )
        
        ce_loss = self.ce_loss_fn(pitch_logits, pitch_annotation)
        bt_loss = self.bt_loss_fn(eq, timbre_latent)

        # Total loss
        total_loss = elbo_loss + ce_loss + bt_loss
        
        # Log losses with batch size
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('train_elbo_loss', elbo_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('train_ce_loss', ce_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('train_bt_loss', bt_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        spec, note_tensors, pitch_annotation, _ = batch
        batch_size = spec.size(0)  # Extract batch size
        note_tensors, pitch_annotation = torch.stack(note_tensors), torch.stack(pitch_annotation)
        
        # Forward pass
        rec_mixture, pitch_latent, pitch_logits, timbre_latent, \
            timbre_mean, timbre_logvar, eq = self(spec, note_tensors)

        # Get pitch priors
        ohe_pitch_annotation = F.one_hot(pitch_annotation, num_classes=self.pitch_classes).float()
        pitch_priors = self.pitch_prior(ohe_pitch_annotation)
        
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            spec, rec_mixture,
            timbre_mean, timbre_logvar,
            pitch_latent, pitch_priors,
        )
        
        ce_loss = self.ce_loss_fn(pitch_logits, pitch_annotation)
        bt_loss = self.bt_loss_fn(eq, timbre_latent)

        # Total loss
        total_loss = elbo_loss + ce_loss + bt_loss
        
        # Log validation loss with batch size
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val_elbo_loss', elbo_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_ce_loss', ce_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_bt_loss', bt_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)

        return total_loss


    def test_step(self, batch, batch_idx):
        spec, note_tensors, pitch_annotation, _ = batch
        batch_size = spec.size(0)  # Extract batch size
        note_tensors, pitch_annotation = torch.stack(note_tensors), torch.stack(pitch_annotation)
        
        # Forward pass
        rec_mixture, pitch_latent, pitch_logits, timbre_latent, \
            timbre_mean, timbre_logvar, eq = self(spec, note_tensors)

        # Get pitch priors
        ohe_pitch_annotation = F.one_hot(pitch_annotation, num_classes=self.pitch_classes).float()
        pitch_priors = self.pitch_prior(ohe_pitch_annotation)
        
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            spec, rec_mixture,
            timbre_mean, timbre_logvar,
            pitch_latent, pitch_priors,
        )
        
        ce_loss = self.ce_loss_fn(pitch_logits, pitch_annotation)
        bt_loss = self.bt_loss_fn(eq, timbre_latent)

        # Total loss
        total_loss = elbo_loss + ce_loss + bt_loss
        
        # Log validation loss with batch size
        print("Test loss:", total_loss)
        return total_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Example usage
    batch_size = 8
    em = torch.randn(batch_size, 64)  # Example mixture encoder output
    eq = torch.randn(batch_size, 64)  # Example query encoder output

    model = TimbreEncoder()
    timbre_embedding, mean, logvar = model(em, eq)

    print("Timbre Embedding:", timbre_embedding.shape)  # Should be [batch_size, 64]
    print("Mean:", mean.shape)  # Should be [batch_size, 64]
    print("Log Variance:", logvar.shape)  # Should be [batch_size, 64]
