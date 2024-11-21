import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

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
        """
        prob = torch.sigmoid(logits)
        if self.training:
            h = torch.rand_like(prob)  # Use random threshold during training
        else:
            h = torch.full_like(prob, 0.5)  # Fixed threshold of 0.5 during inference
            
        return (prob > h).float()  # Binarize based on the threshold h
        



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
            nn.Linear(hidden_dim, hidden_dim), # First layer
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
            nn.Linear(hidden_dim, hidden_dim), # Last layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Gaussian parameterization layers
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.logvar_layer = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mean, logvar):
        """
            Reparameterization trick to sample from N(mean, var)
            Sampling by μφτ (·) + ε σφτ (·)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + (eps * std)

    def forward(self, em, eq):
        # Concatenate the mixture and query embeddings
        concat_input = torch.cat([em, eq], dim=-1)  # Concatenate along feature dimension
        
        # Shared layers forward pass
        hidden_state = self.shared_layers(concat_input)  # Shared hidden state output

        # Calculate mean and log variance
        mean, logvar = self.mean_layer(hidden_state), self.logvar_layer(hidden_state)  # Gaussian distribution

        # Sample the timbre latent using the reparameterization trick
        timbre_latent = self.reparameterize(mean, logvar)
       
        # # sample z from q
        # std = torch.exp(logvar / 2)
        # q = torch.distributions.Normal(mean, std)
        # timbre_latent = q.rsample()
        
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
            nn.Linear(hidden_dim, pitch_classes),  # Output logits for pitch classification
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
        
        # Transcriber
        pitch_logits = self.transcriber(concat_input)

        # SB Layer
        y_bin = self.sb_layer(pitch_logits)  # Apply binarisation
        
        # Projection Layer
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
    
    def forward(self, pitch_latents, timbre_latents):
        # FiLM layer: modulates pitch latents based on timbre latents
        source_latents = self.film(pitch_latents, timbre_latents)
        source_latents = source_latents.unsqueeze(1).repeat(1, self.num_frames, 1) # Expand source_latents along time axis if necessary
        output, _ = self.gru(source_latents)
        output = self.linear(output).transpose(1, 2) # torch.Size([32, 10, 64])
        
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
        
        # Add storage for stored test timbre latents and instrument labels
        self.test_timbre_latents = []
        self.test_instrument_labels = []
        

    def forward(self, mixture, query):
        # Encode mixture and query
        em = self.mixture_encoder(mixture)
        eq = self.query_encoder(query)

        # Encode pitch and timbre latents
        pitch_latent, pitch_logits = self.pitch_encoder(em, eq)
        timbre_latent, timbre_mean, timbre_logvar = self.timbre_encoder(em, eq)
        
        # Decode to reconstruct the mixture
        rec_source_spec = self.decoder(pitch_latent, timbre_latent)
        return rec_source_spec, pitch_latent, pitch_logits, timbre_latent, timbre_mean, timbre_logvar, eq


    def training_step(self, batch, batch_idx):
        spec, note_tensors, pitch_annotation, _ = batch
        batch_size = spec.size(0)  # Extract batch size
        note_numbers = [i.shape[0] for i in note_tensors] # [4, 4, 4, 3, 4, 4, 4, 3, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4]

        """ Reconstruct spec, note_tensors, pitch_annotation """
        repeated_slices = [spec[i].repeat(count, 1, 1) for i, count in enumerate(note_numbers)]
        repeated_spec = torch.cat(repeated_slices, dim=0)
        note_tensors = torch.cat(note_tensors, dim=0)
        pitch_annotation = torch.cat(pitch_annotation, dim=0)
        
        # Forward pass
        rec_source_spec, pitch_latent, pitch_logits, timbre_latent, \
            timbre_mean, timbre_logvar, eq = self(repeated_spec, note_tensors)

        # Get pitch priors
        ohe_pitch_annotation = F.one_hot(pitch_annotation, num_classes=self.pitch_classes).float()
        pitch_priors = self.pitch_prior(ohe_pitch_annotation)
        
        # Get reconstruct mixture by summing each split along the first dimension and concatenate
        splits = torch.split(rec_source_spec, note_numbers, dim=0)
        summed_splits = [s.sum(dim=0, keepdim=True) for s in splits]
        rec_mixture = torch.cat(summed_splits, dim=0)
                
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            spec, rec_mixture,
            timbre_latent, timbre_mean, timbre_logvar,
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

    def evaluate(self, batch, stage='val'):
        spec, note_tensors, pitch_annotation, instrument_label = batch
        batch_size = spec.size(0)  # Extract batch size
        note_numbers = [i.shape[0] for i in note_tensors] # [4, 3, 4, ..., 4]

        """ Reconstruct spec, note_tensors, pitch_annotation """
        repeated_slices = [spec[i].repeat(count, 1, 1) for i, count in enumerate(note_numbers)]
        repeated_spec = torch.cat(repeated_slices, dim=0)
        note_tensors = torch.cat(note_tensors, dim=0)
        pitch_annotation = torch.cat(pitch_annotation, dim=0)
        instrument_label = torch.cat(instrument_label, dim=0)
        
        # Forward pass
        rec_source_spec, pitch_latent, pitch_logits, timbre_latent, \
            timbre_mean, timbre_logvar, eq = self(repeated_spec, note_tensors)

        self.test_timbre_latents.append(timbre_latent.detach().cpu())
        self.test_instrument_labels.append(instrument_label.detach().cpu())
            
        # Get pitch priors
        ohe_pitch_annotation = F.one_hot(pitch_annotation, num_classes=self.pitch_classes).float()
        pitch_priors = self.pitch_prior(ohe_pitch_annotation)
        
        # Get reconstruct mixture by summing each split along the first dimension and concatenate
        splits = torch.split(rec_source_spec, note_numbers, dim=0)
        summed_splits = [s.sum(dim=0, keepdim=True) for s in splits]
        rec_mixture = torch.cat(summed_splits, dim=0)
                
        # Compute losses
        elbo_loss = self.elbo_loss_fn(
            spec, rec_mixture,
            timbre_latent, timbre_mean, timbre_logvar,
            pitch_latent, pitch_priors,
        )
        
        ce_loss = self.ce_loss_fn(pitch_logits, pitch_annotation)
        bt_loss = self.bt_loss_fn(eq, timbre_latent)

        # Total loss
        total_loss = elbo_loss + ce_loss + bt_loss
        
        # Get accuracy
        predicted_classes = torch.argmax(pitch_logits, dim=1)
        correct_predictions = (predicted_classes == pitch_annotation).float().sum()
        accuracy = correct_predictions / len(predicted_classes)

        
        # Log losses and metrics
        self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}_elbo_loss', elbo_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}_ce_loss', ce_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}_acc', accuracy, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log(f'{stage}_bt_loss', bt_loss, on_epoch=True, batch_size=batch_size, sync_dist=True)
        return total_loss



    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, stage='test')


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
    
    def on_validation_epoch_end(self):
        return self.plotting(stage='val')
    
    def on_test_epoch_end(self):
        return self.plotting(stage='test')
    
    def plotting(self, stage='val'):
        # Concatenate stored latents and labels
        timbre_latents = torch.cat(self.test_timbre_latents, dim=0).numpy()
        instrument_labels = torch.cat(self.test_instrument_labels, dim=0).numpy()
        
        # Reset storage lists for the next epoch
        self.test_timbre_latents = []
        self.test_instrument_labels = []

        # Perform T-SNE or PCA
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(timbre_latents)

        # Define unique labels and colormap
        unique_labels = np.unique(instrument_labels)
        colors = plt.cm.tab10.colors[:len(unique_labels)]  # Use colors from tab10 colormap
        cmap = ListedColormap(colors)
        
        # Plotting
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            plt.scatter(latent_2d[instrument_labels == label, 0], 
                        latent_2d[instrument_labels == label, 1], 
                        color=cmap(i), 
                        label=f"Instrument {label}", 
                        alpha=0.7)
        
        plt.title(f"T-SNE of Timbre Latent Embeddings ({stage})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title="Instrument Label", loc="best")  # Add legend instead of colorbar
        
        # Save the figure to a file
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"timbre_latent_tsne_{stage}.png"), format="png", dpi=300)

        # Show the plot
        plt.show()



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
