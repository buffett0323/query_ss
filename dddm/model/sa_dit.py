import math
import torch 
import inspect
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from model.base import BaseModule
from model.diffusion_module import *
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.attention_processor import StableAudioAttnProcessor2_0
from diffusers.schedulers import EDMDPMSolverMultistepScheduler
from typing import Callable, List, Optional, Union

class SADiT(BaseModule):
    def __init__(
        self, 
        cond_add_dim = 2048,
        sample_size=1024,
        in_channels=80, # ✅ Revised
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        num_key_value_attention_heads=12,
        out_channels=80, # ✅ Revised
        cross_attention_dim=768,
        time_proj_dim=256,
        global_states_input_dim=1536,
        cross_attention_input_dim=80, # ✅ Revised
    ):
        super(SADiT, self).__init__()
        self.dit = StableAudioDiTModel(
            sample_size=sample_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_attention_heads=num_key_value_attention_heads,
            out_channels=out_channels,
            cross_attention_dim=cross_attention_dim,
            time_proj_dim=time_proj_dim,
            global_states_input_dim=global_states_input_dim,
            cross_attention_input_dim=cross_attention_input_dim,
        )
        
        # Revise the Stable Audio DiT Model
        self.dit.global_proj = nn.Sequential(
            nn.Linear(cond_add_dim, self.dit.inner_dim, bias=False),  # Changed from 1536 to 2048
            nn.SiLU(),
            nn.Linear(self.dit.inner_dim, self.dit.inner_dim, bias=False),
        )
        self.scheduler = EDMDPMSolverMultistepScheduler()
    
    def forward(
        self, 
        latents, 
        mask, 
        src_out, 
        spk, 
        num_inference_steps=100,
        do_classifier_free_guidance=False,
        guidance_scale=7.0,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: Optional[float] = 0.0,
    ):
        
        # Conditions
        extracted_condition = src_out
        extracted_condition = extracted_condition.transpose(1, 2)
        global_condition = spk.transpose(1, 2)
        
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        timesteps = self.scheduler.timesteps
        
        
        # Rotary Embedding
        rotary_embed_dim = self.dit.config.attention_head_dim // 2  # 32
        rotary_embedding = get_1d_rotary_pos_embed(
            rotary_embed_dim,
            latents.shape[2] + global_condition.shape[1], # 200 + 1 = 201
            use_real=True,
            repeat_interleave_real=False,
        )
        
        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if t.dim() == 0:
                t = t.unsqueeze(0)

            # predict the noise residual
            noise_pred = self.dit(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=extracted_condition,
                global_hidden_states=global_condition, # Pass timbre embedding (2048-D) through global projection
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        return latents
    
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    
    def get_alphas_sigmas(self, t):
        """Returns the scaling factors for the clean image (alpha) and for the
        noise (sigma), given a timestep."""
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
        
    def compute_loss(self, latents, mask, src_out, spk):
        # ======= Set Parameters =======
        # (x0, mask, src_out, spk) -> (latents, mask, src_out, spk)
        # x0 -> latents : torch.Size([4, 80, 200])
        # src_out -> src_out : torch.Size([4, 80, 200])
        # spk -> condition : torch.Size([4, 2048, 1])
        
        # ======= Start DiT =======
        # TODO: mask
        b = latents.shape[0]
        t = torch.sigmoid(torch.randn(b)).to(latents.device)
        
        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = self.get_alphas_sigmas(t)  # get_alphas_sigmas should be defined as in the wrapper
        alphas = alphas[:, None, None]  # Shape to match latents
        sigmas = sigmas[:, None, None]
        
        # Sample noise and add it to the latents
        noise = torch.randn_like(latents)
        noisy_latents = latents * alphas + noise * sigmas
        
        # Determine the target for v_prediction
        targets = alphas * noise - sigmas * latents
        
        
        # Conditions
        extracted_condition = src_out
        extracted_condition = extracted_condition.transpose(1, 2)
        global_condition = spk.transpose(1, 2)
        

        # Rotary Embedding
        rotary_embed_dim = self.dit.config.attention_head_dim // 2  # 32
        rotary_embedding = get_1d_rotary_pos_embed(
            rotary_embed_dim,
            latents.shape[2] + global_condition.shape[1], # 200 + 1 = 201
            use_real=True,
            repeat_interleave_real=False,
        )
        
        # Diffusion Transformer
        # TODO: Use Fun-Duo's settings
        model_pred = self.dit(
            hidden_states=noisy_latents,
            timestep=t,
            encoder_hidden_states=extracted_condition,
            global_hidden_states=global_condition, # Pass timbre embedding (2048-D) through global projection
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        
        loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")
        return loss