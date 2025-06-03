Cocochorales website:
https://magenta.tensorflow.org/datasets/cocochorales

Structure:
string_track000001
   |-- metadata.yaml
   |-- mix.mid
   |-- stems_MIDI
   |    |-- 0_violin.mid
   |    |-- 1_violin.mid
   |    |-- 2_viola.mid
   |    |-- 3_cello.mid
   |-- mix.wav
   |-- stems_audio
        |-- 0_violin.wav
        |-- 1_violin.wav
        |-- 2_viola.wav
        |-- 3_cello.wav

There are 28,179 samples of mixtures split into the train, validation, and test sets with a ratio of 70/20/10. The wave- forms are converted into mel spectrograms using 128 mel- filter bands, a window size of 1,024, and a hop length of 512. We crop a 320ms segment, or 10 spectral frames, from the sustain phase of each sample.


- The framework is based on the fully-supervised query-based source separation framework. So in addition to the reconstruction loss wrt the mixture, you would also need to have a reconstruction loss for each individual source, just like how one would train a supervised source separation model. -- In testing period.

- The pitch prior (purple highlighted) is not critical so you might want to leave it. Table 1 in the paper reports the results without the pitch prior, and Table 6 shows minor improvements with different choices of prior. -- Temporarily leave it

- Applied BCE before SB, because back propagating the pitch transcription gradient through SB could be more unstable. Though the argument is hand-wavy and requires experiments to verify, it could be contributing factor to unstable trainings. -- V


- ELBO Loss implementation:
1. reconstruction loss for each individual source -- V
2. KL Loss check


## LDM
Question:
1. How does Timbre read 8x100x16? turn it into 128x100?
2. concatenate? FiLM?


The LDM framework in the DisMix model works by leveraging pre-trained VAE from AudioLDM2 to improve compute efficiency during diffusion modeling. Here’s the detailed process for getting \( z_{s,0} \) after obtaining \( s_i \) from the concatenation of pitch latent (\( \nu^{(i)} \)) and timbre latent (\( \tau^{(i)} \)):

1. **Projection to Latent Space**:
   - Each individual source \( x^{(i)}_s \) is encoded into a latent space using a VAE encoder, \( z^{(i)}_s = E_{\text{vae}}(x^{(i)}_s) \), where \( z^{(i)}_s \) is a compact representation of the source's spectrogram.

2. **Conditioning Mechanism**:
   - The set of source-level representations \( S = \{ s^{(i)} \} \) is prepared from the concatenated pitch (\( \nu^{(i)} \)) and timbre (\( \tau^{(i)} \)) latents for all sources. Each \( s^{(i)} \) integrates these latents to encapsulate the source-specific characteristics.

3. **Partitioning**:
   - Both the latent representation \( z_s^{(i)} \) and the source-level representation \( s^{(i)} \) are partitioned into smaller patches for temporal alignment using partitioning functions. This creates \( z_{m,t} \) and \( s_c \), facilitating transformer-based processing while preserving temporal and feature dimensions.

4. **Diffusion Transformer**:
   - The latent representation \( z_{m,t} \) undergoes iterative reverse diffusion steps \( T \) (from \( z_{m,T} \) to \( z_{m,0} \)) conditioned on \( s_c \). This conditioning integrates source-specific information into the reconstruction process.

5. **Reconstruction of \( z_{m,0} \)**:
   - The resulting \( z_{m,0} \) is reconstructed by the latent diffusion model, completing the process. At the final stage, this latent representation can be decoded back to a mixture or individual sources.

6. **Decoding to Obtain \( x^{(i)}_s \)**:
   - \( z_{m,0} \) is converted back to mel spectrograms using the pre-trained VAE decoder \( D_{\text{vae}}(z_{m,0}) \), and audio signals are subsequently reconstructed.

This stepwise process allows for efficient and high-quality reconstruction while maintaining disentanglement between pitch and timbre attributes.


HIFI-GAN Usage example:
https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_hifigan.ipynb#scrollTo=b02dd30d


  | Name           | Type             | Params
----------------------------------------------------
0 | Q_Encoder      | QueryEncoder     | 5.6 M
1 | M_Encoder      | MixtureEncoder   | 22.4 M
2 | combine_conv   | Conv2d           | 528
3 | pitch_encoder  | PitchEncoder     | 14.2 M
4 | timbre_encoder | TimbreEncoder    | 279 K
5 | dit            | DiT              | 72.2 M
6 | elbo_loss_fn   | ELBOLoss         | 1
7 | ce_loss_fn     | CrossEntropyLoss | 0
8 | bt_loss_fn     | BarlowTwinsLoss  | 0
