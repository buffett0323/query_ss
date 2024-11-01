
# # Main Model
# class DisMixModel(nn.Module):
#     def __init__(
#         self, 
#         input_dim=128, 
#         latent_dim=64, 
#         hidden_dim=256, 
#         gru_hidden_dim=256,
#         num_frames=10,
#         pitch_classes=52,
#         output_dim=128,
#     ):
#         super(DisMixModel, self).__init__()
#         self.mixture_encoder = MixtureQueryEncoder(
#             input_dim=input_dim,
#             hidden_dim=768,
#             output_dim=latent_dim,
#         )
#         self.query_encoder = MixtureQueryEncoder(
#             input_dim=input_dim,
#             hidden_dim=768,
#             output_dim=latent_dim,
#         )
#         self.pitch_encoder = PitchEncoder(
#             input_dim=input_dim, 
#             hidden_dim=hidden_dim, 
#             pitch_classes=pitch_classes, # true labels not 0-51
#             output_dim=latent_dim
#         )
#         self.timbre_encoder = TimbreEncoder(
#             input_dim=input_dim, 
#             hidden_dim=hidden_dim, 
#             output_dim=latent_dim
#         )
#         self.decoder = DisMixDecoder(
#             pitch_dim=latent_dim, 
#             timbre_dim=latent_dim, 
#             gru_hidden_dim=gru_hidden_dim, 
#             output_dim=output_dim, 
#             num_frames=num_frames,
#             num_layers=2
#         )
        
#     def forward(self, mixture, query):
        
#         # Encode mixture and query
#         em = self.mixture_encoder(mixture)
#         eq = self.query_encoder(query)

#         # Encode pitch and timbre latents
#         pitch_latent, pitch_logits = self.pitch_encoder(em, eq)
#         timbre_latent, timbre_mean, timbre_logvar = self.timbre_encoder(em, eq)
        
#         # Decode to reconstruct the mixture
#         reconstructed_spectrogram = self.decoder(pitch_latent, timbre_latent)
#         return reconstructed_spectrogram, pitch_latent, pitch_logits, timbre_latent, timbre_mean, timbre_logvar, eq
