# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import librosa
import librosa.display
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spec_aug.sparse_image_warp_pytorch import sparse_image_warp
import torch
import torch.nn as nn



class SpecAugment(nn.Module):
    def __init__(self,
                 time_warping_para=80,
                 frequency_masking_para=27,
                 time_masking_para=100,
                 frequency_mask_num=1,
                 time_mask_num=1,
                 p_time_warp=0.4,
                 p_mask=0.5):
        super(SpecAugment, self).__init__()
        self.time_warping_para = time_warping_para
        self.frequency_masking_para = frequency_masking_para
        self.time_masking_para = time_masking_para
        self.frequency_mask_num = frequency_mask_num
        self.time_mask_num = time_mask_num
        self.p_time_warp = p_time_warp
        self.p_mask = p_mask

    def time_warp(self, spec):
        """Apply sparse image warp on a single [F, T] spectrogram."""
        F, T = spec.shape
        y = F // 2

        x_pos = random.randint(self.time_warping_para, T - self.time_warping_para - 1)
        dist = random.randint(-self.time_warping_para, self.time_warping_para)

        src_pts = torch.tensor([[[y, x_pos]]], dtype=torch.float32, device=spec.device)
        dst_pts = torch.tensor([[[y, x_pos + dist]]], dtype=torch.float32, device=spec.device)

        # Add batch dimension [1, F, T]
        spec = spec.unsqueeze(0)
        warped, _ = sparse_image_warp(spec, src_pts, dst_pts)  # output: [1, F, T]
        return warped.squeeze(0)

    def forward(self, mel_batch):
        """
        Input: mel_batch of shape [B, F, T]
        Output: same shape [B, F, T] with spec augment applied
        """
        B, F, T = mel_batch.shape
        device = mel_batch.device
        out = mel_batch.clone()

        # # Step 1: Time warp (per sample)
        # TODO: Add time warp, due to implement issues, we skip this step
        # if random.random() <= self.p_time_warp:
        #     for i in range(B):
        #         out[i] = self.time_warp(out[i])

        # # Step 2: Frequency masking
        # TODO: Remove this since it might affect timbre learning
        # if random.random() <= self.p_mask:
        #     for _ in range(self.frequency_mask_num):
        #         f = random.randint(0, self.frequency_masking_para)
        #         f0s = torch.randint(0, F - f, (B,), device=device)
        #         for i in range(B):
        #             out[i, f0s[i]:f0s[i] + f, :] = 0

        # Step 3: Time masking
        if random.random() <= self.p_mask:
            for _ in range(self.time_mask_num):
                t = random.randint(0, self.time_masking_para)
                t0s = torch.randint(0, T - t, (B,), device=device)
                for i in range(B):
                    out[i, :, t0s[i]:t0s[i] + t] = 0
        
        return out


def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=1, 
                 p1=0.4, p2=0.5):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    if random.random() <= p1:
        warped_mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)
    else:
        warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    if random.random() <= p2:
        for _ in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f, :] = 0
    

    # Step 3 : Time masking
    if random.random() <= p2:
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau-t)
            warped_mel_spectrogram[:, :, t0:t0+t] = 0

    return warped_mel_spectrogram


def visualization_spectrogram(mel_spectrogram, title, save_path=None):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    if save_path:
        plt.savefig(save_path)
        plt.close()