import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential
from mamba.mamba_ssm.modules.mamba2 import Mamba2
from ptflops import get_model_complexity_info

class MambaBlock(nn.Module):
    def __init__(self, in_channels):
        super(MambaBlock, self).__init__()
        self.forward_mamba2 = Mamba2(
            d_model=in_channels,
            d_state=128,
            d_conv=4,
            expand=4,
            headdim=64,
        )

        self.backward_mamba2 = Mamba2(
            d_model=in_channels,
            d_state=128,
            d_conv=4,
            expand=4,
            headdim=64,
        )
    def forward(self, input):
        forward_f = input
        forward_f_output = self.forward_mamba2(forward_f)
        backward_f = torch.flip(input, [1])
        backward_f_output = self.backward_mamba2(backward_f)
        backward_f_output2 = torch.flip(backward_f_output, [1])
        output = torch.cat([forward_f_output + input, backward_f_output2+input], -1)
        return output

class TAC(nn.Module):
    """
    A transform-average-concatenate (TAC) module.
    """
    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()

        self.input_size = input_size
        self.eps = torch.finfo(torch.float32).eps

        self.input_norm = nn.GroupNorm(1, input_size, self.eps)
        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.Tanh()
                                      )
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.Tanh()
                                     )
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size*2, input_size),
                                        nn.Tanh()
                                       )

    def forward(self, input):
        # input shape: batch, group, N, *

        batch_size, G, N = input.shape[:3]
        output = self.input_norm(input.view(batch_size*G, N, -1)).view(batch_size, G, N, -1)
        T = output.shape[-1]

        # transform
        group_input = output  # B, G, N, T
        group_input = group_input.permute(0,3,1,2).contiguous().view(-1, N)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size*T, -1)  # B*T, H
        group_mean = self.TAC_mean(group_mean).unsqueeze(1).expand(batch_size*T, G, group_mean.shape[-1]).contiguous()  # B*T, G, H

        # concate
        group_output = group_output.view(batch_size*T, G, -1)  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = group_output.view(batch_size, T, G, -1).permute(0,2,3,1).contiguous()  # B, G, N, T
        output = input + group_output.view(input.shape)

        return output

class ResMamba(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0., bidirectional=True):
        super(ResMamba, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = MambaBlock(input_size)
        self.proj = nn.Linear(input_size*2 ,input_size)
        # linear projection layer

    def forward(self, input):
        # input shape: batch, dim, seq
        rnn_output =  self.rnn(self.dropout(self.norm(input)).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0], input.shape[2], input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()

class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = ResMamba(self.feature_dim, self.feature_dim*2)
        self.band_comm = ResMamba(self.feature_dim, self.feature_dim*2)
        self.channel_comm = TAC(self.feature_dim, self.feature_dim*3)

    def forward(self, input):
        # input shape: B, nch, nband*N, T
        B, nch, N, T = input.shape

        band_output = self.band_rnn(input.view(B*nch*self.nband, self.feature_dim, -1)).view(B*nch, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0,3,2,1).contiguous().view(B*nch*T, -1, self.nband)
        output = self.band_comm(band_output).view(B*nch, T, -1, self.nband).permute(0,3,2,1).contiguous()

        # channel comm
        output = output.view(B, nch, self.nband, -1, T).transpose(1,2).contiguous().view(B*self.nband, nch, -1, T)
        output = self.channel_comm(output).view(B, self.nband, nch, -1, T).transpose(1,2).contiguous()

        return output.view(B, nch, N, T)





class Separator(nn.Module):
    def __init__(
        self,
        sr=44100,
        win=2048,
        stride=512,
        feature_dim=128, # N
        num_repeat_mask=4, #8,
        num_repeat_map=2, #4,
        num_output=4
    ):
        super(Separator, self).__init__()

        self.sr = sr
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.num_output = num_output
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (50 hop), 1k-2k (100 hop), 2k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_50 = int(np.floor(50 / (sr / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        bandwidth_1k = int(np.floor(1000 / (sr / 2.) * self.enc_dim))
        bandwidth_2k = int(np.floor(2000 / (sr / 2.) * self.enc_dim))
        self.band_width = [bandwidth_50]*20
        self.band_width += [bandwidth_100]*10
        self.band_width += [bandwidth_250]*8
        self.band_width += [bandwidth_500]*8
        self.band_width += [bandwidth_1k]*8
        self.band_width += [bandwidth_2k]*2
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)

        self.BN_mask = nn.ModuleList([])
        for i in range(self.nband):
            self.BN_mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i]*2, self.eps),
                    nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)
                )
            )
        self.BN_map = nn.ModuleList([])
        for i in range(self.nband):
            self.BN_map.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i]*2, self.eps),
                    nn.Conv1d(self.band_width[i]*2, self.feature_dim, 1)
                )
            )

        self.separator_mask = []
        for i in range(num_repeat_mask):
            self.separator_mask.append(BSNet(self.nband*self.feature_dim, self.nband))
        self.separator_mask = nn.Sequential(*self.separator_mask)

        self.separator_map = []
        for i in range(num_repeat_map):
            self.separator_map.append(BSNet(self.nband * self.feature_dim, self.nband))
        self.separator_map = nn.Sequential(*self.separator_map)

        self.in_conv = nn.Conv1d(self.feature_dim*2, self.feature_dim, 1)
        self.Tanh = nn.Tanh()
        self.mask = nn.ModuleList([])
        self.map = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim*1*self.num_output, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim*1*self.num_output, self.feature_dim*1*self.num_output, 1, groups=self.num_output),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim*1*self.num_output, self.band_width[i]*4*self.num_output, 1, groups=self.num_output)
                    )
                )
            self.map.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim*1*self.num_output, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim*1*self.num_output, self.feature_dim*1*self.num_output, 1, groups=self.num_output),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim*1*self.num_output, self.band_width[i]*4*self.num_output, 1, groups=self.num_output)
                    )
                )

    def forward(self, input):

        batch_size, nch, nsample = input.shape # input shape: (B, C, T)
        input = input.view(batch_size*nch, -1)

        # STFT
        spec = torch.stft(input, n_fft=self.win, hop_length=self.stride,
                          window=torch.hann_window(self.win).to(input.device).type(input.type()),
                          return_complex=True)

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(spec_RI[:,:,band_idx:band_idx+self.band_width[i]].contiguous())
            subband_spec.append(spec[:,band_idx:band_idx+self.band_width[i]])  # B*nch, BW, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature_mask = []
        for i in range(len(self.band_width)):
            subband_feature_mask.append(self.BN_mask[i](subband_spec_RI[i].view(batch_size*nch, self.band_width[i]*2, -1)))
        subband_feature_mask = torch.stack(subband_feature_mask, 1)  # B, nband, N, T -> torch.Size([2, 57, 128, 87])

        subband_feature_map = []
        for i in range(len(self.band_width)):
             subband_feature_map.append(self.BN_map[i](subband_spec_RI[i].view(batch_size*nch, self.band_width[i] * 2, -1)))
        subband_feature_map = torch.stack(subband_feature_map, 1)  # B, nband, N, T -> torch.Size([2, 57, 128, 87])

        # separator
        sep_output = checkpoint_sequential(self.separator_mask, 2, subband_feature_mask.view(batch_size, nch, self.nband*self.feature_dim, -1), use_reentrant=False)  # B, nband*N, T
        sep_output = sep_output.view(batch_size*nch, self.nband, self.feature_dim, -1)

        combined = torch.cat((subband_feature_map, sep_output), dim=2)
        combined = combined.reshape(batch_size * nch * self.nband,self.feature_dim*2,-1)
        combined = self.Tanh(self.in_conv(combined))
        combined = combined.reshape(batch_size * nch, self.nband,self.feature_dim,-1)
        sep_output2 = checkpoint_sequential(self.separator_map, 2, combined.view(batch_size, nch, self.nband*self.feature_dim, -1), use_reentrant=False)  # 1B, nband*N, T
        sep_output2 = sep_output2.view(batch_size * nch, self.nband, self.feature_dim, -1)


        return sep_output2



if __name__ == "__main__":
    x = torch.randn(1, 2, 44100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    model = Separator().to(device)
    res = model(x)
    print(res.shape)


    # with torch.no_grad():
    #     macs, params = get_model_complexity_info(
    #         model, (2, 44100), as_strings=False, print_per_layer_stat=True, verbose=False
    #     )
    # macs = macs/1e9
    # params = params/1e6
    # print(f"MACs: {macs}")
    # print(f"Params: {params}")
