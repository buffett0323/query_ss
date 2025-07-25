import torch
import typing as tp


class SAD:
    """
    SAD(Source Activity Detector)
    """

    def __init__(
            self,
            sr: int,
            window_size_in_sec: int = 6,
            overlap_ratio: float = 0.5,
            n_chunks_per_segment: int = 10,
            eps: float = 1e-5,
            gamma: float = 1e-3,
            threshold_max_quantile: float = 0.15,
            threshold_segment: float = 0.5,
    ):
        self.sr = sr
        self.n_chunks_per_segment = n_chunks_per_segment
        self.eps = eps
        self.gamma = gamma
        self.threshold_max_quantile = threshold_max_quantile
        self.threshold_segment = threshold_segment

        self.window_size = sr * window_size_in_sec
        self.step_size = int(self.window_size * overlap_ratio)

    def chunk(self, y: torch.Tensor):
        """
        Input shape: [n_channels, n_frames]
        Output shape: []
        """
        y = y.unfold(-1, self.window_size, self.step_size)
        y = y.chunk(self.n_chunks_per_segment, dim=-1)
        y = torch.stack(y, dim=-2)
        return y

    @staticmethod
    def calculate_rms(y: torch.Tensor):
        """Calculates torch tensor rms from audio signal
        RMS = sqrt(1/N * sum(x^2))
        """
        y_squared = torch.pow(y, 2) # need to square signal before mean and sqrt
        y_mean = torch.mean(torch.abs(y_squared), dim=-1, keepdim=True)
        y_rms = torch.sqrt(y_mean)
        return y_rms

        # # Version from `augmentations.py`
        # return torch.sqrt(
        #     torch.mean(torch.square(y), dim=-1, keepdim=keepdim)
        # )


    def calculate_thresholds(self, rms: torch.Tensor):
        """
        """
        rms[rms == 0.] = self.eps
        rms_threshold = torch.quantile(
            rms,
            self.threshold_max_quantile,
            dim=-2,
            keepdim=True,
        )
        rms_threshold[rms_threshold < self.gamma] = self.gamma
        rms_percentage = torch.mean(
            (rms > rms_threshold).float(),
            dim=-2,
            keepdim=True,
        )
        rms_mask = torch.all(rms_percentage > self.threshold_segment, dim=0).squeeze()
        return rms_mask

    def calculate_salient(self, y: torch.Tensor, mask: torch.Tensor):
        """
        """
        y = y[:, mask, ...]
        C, D1, D2, D3 = y.shape
        y = y.view(C, D1, D2*D3)
        return y

    def __call__(
            self,
            y: torch.Tensor,
            segment_saliency_mask: tp.Optional[torch.Tensor] = None
    ):
        """
        Stacks signal into segments and filters out silent segments.
        :param y: Input signal.
            Shape [n_channels, n_frames]
               segment_saliency_mask: Optional precomputed mask
            Shape [n_channels, n_segments, 1, 1]
        :return: Salient signal folded into segments of length 'self.window_size' and step 'self.step_size'.
            Shape [n_channels, n_segments, frames_in_segment]
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        if segment_saliency_mask is None:
            segment_saliency_mask = self.calculate_thresholds(rms)
        y_salient = self.calculate_salient(y, segment_saliency_mask)
        return y_salient, segment_saliency_mask

    def calculate_salient_indices(
            self,
            y: torch.Tensor
    ):
        """
        Returns start indices of salient regions of audio
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        mask = self.calculate_thresholds(rms)
        indices = torch.arange(mask.shape[-1])[mask] * self.step_size
        return indices.tolist()


if __name__ == "__main__":
    import torchaudio

    sr = 44100
    example_path = '/mnt/gestalt/home/ddmanddman/MUSDB/test/Louis Cressy Band - Good Time/vocals.wav'

    sad = SAD(sr=sr)
    y, sr = torchaudio.load(example_path)
    print("Original sample rate: ", sr)
    y_salience = sad(y)

    indices = sad.calculate_salient_indices(y)
    print(f"Initial shape: ({y[0].shape}, {y[1].shape}).\nShape after source activity detection: ({y_salience[0].shape}, {y_salience[1].shape})")
    print(f"Salient indices: {indices}")
