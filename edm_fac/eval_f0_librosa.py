import librosa
import numpy as np

def f0_rmse(
    pred_wav, 
    gt_wav, 
    sr=44100, 
    fmin=50.0,
    fmax=2000.0, 
    frame_length=2048, 
    hop_length=512
):
    """
    Compute F0 RMSE between predicted audio and ground truth audio using librosa.pyin.
    
    Parameters:
    -----------
    pred_wav : str
        Path to predicted audio file
    gt_wav : str
        Path to ground-truth audio file
    sr : int
        Target sampling rate (default: 44100 Hz)
    fmin, fmax : float
        Minimum and maximum frequency range for F0 estimation
    frame_length : int
        Frame length for STFT
    hop_length : int
        Hop length for STFT
    
    Returns:
    --------
    rmse : float
        Root-mean-square error of F0 (Hz) between predicted and ground truth
    """

    # Load audio
    y_pred, _ = librosa.load(pred_wav, sr=sr)
    y_gt, _   = librosa.load(gt_wav, sr=sr)

    # Extract F0 with librosa.pyin (NaN for unvoiced frames)
    f0_pred, _, _ = librosa.pyin(y_pred, fmin=fmin, fmax=fmax,
                                 sr=sr, frame_length=frame_length, hop_length=hop_length)
    f0_gt, _, _   = librosa.pyin(y_gt, fmin=fmin, fmax=fmax,
                                 sr=sr, frame_length=frame_length, hop_length=hop_length)

    # Align lengths (pad shorter one with NaNs)
    min_len = min(len(f0_pred), len(f0_gt))
    f0_pred, f0_gt = f0_pred[:min_len], f0_gt[:min_len]

    # Only evaluate on frames where both have valid F0
    mask = ~np.isnan(f0_pred) & ~np.isnan(f0_gt)
    if np.sum(mask) == 0:
        return np.nan

    rmse = np.sqrt(np.mean((f0_pred[mask] - f0_gt[mask])**2))
    return rmse


# Example usage
if __name__ == "__main__":
    pred_file = "predicted.wav"
    gt_file   = "ground_truth.wav"
    rmse_val = f0_rmse(pred_file, gt_file)
    print(f"F0 RMSE: {rmse_val:.2f} Hz")