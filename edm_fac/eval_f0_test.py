import torch, torchaudio, torchcrepe

f1_vals = []
rmse_vals = []
corr_vals = []
iter_cnt = 200000

for i in range(32):
    print(f"Processing {i:02d}")
    ref_path = f"/home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/sample_audio/iter_{iter_cnt}/conv_both/{i:02d}_gt.wav"
    est_path = f"/home/buffett/nas_data/EDM_FAC_LOG/0804_proposed/sample_audio/iter_{iter_cnt}/conv_both/{i:02d}_recon.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_ref, sr = torchaudio.load(ref_path)   # 參考音檔
    wav_est, _  = torchaudio.load(est_path)   # 轉調後音檔（或你的模型輸出）
    wav_ref = wav_ref.to(device); wav_est = wav_est.to(device)

    # 1) GPU 重採樣到 16k
    resampler = torchaudio.transforms.Resample(44100, 16000).to(device)
    x = resampler(wav_ref)  # (1,T')
    y = resampler(wav_est)

    # 2) batched 一次丟進 torchcrepe（維度要 (B,1,T)）
    hop = 160; fmin, fmax = 50., 1100.; th = 0.5

    def crepe_f0(wav_1x1T):
        f0, pd = torchcrepe.predict(
            wav_1x1T, sample_rate=16000, hop_length=hop,
            fmin=fmin, fmax=fmax, model='tiny',
            device=device, pad=True, return_periodicity=True
        )  # (1,F)
        pd = torchcrepe.filter.median(pd, win_length=3)
        return f0, pd

    f0r, pdr = crepe_f0(x)
    f0e, pde = crepe_f0(y)

    # 3) 共同有效遮罩
    vr = pdr >= th
    ve = pde >= th
    joint = vr & ve

    # 4) 在 joint 上計算
    def hz_to_cents(hz): return 1200.0 * torch.log2(hz)
    valid = joint & torch.isfinite(f0r) & torch.isfinite(f0e) & (f0r>0) & (f0e>0)

    if valid.any():
        r = f0r[valid]; e = f0e[valid]
        # RMSE in cents
        rmse_c = torch.sqrt(torch.mean((hz_to_cents(e) - hz_to_cents(r))**2))
        # Pearson corr in Hz
        r0 = r - r.mean(); e0 = e - e.mean()
        corr = (r0 @ e0) / (torch.sqrt((r0**2).sum()*(e0**2).sum()) + 1e-12)
        print("F0RMSE_cents:", rmse_c.item(), "F0CORR:", corr.item())
        rmse_vals.append(rmse_c.item())
        corr_vals.append(corr.item())
    else:
        print("No valid voiced-overlap frames; report Voicing metrics instead.")

    # 5) 也回報 Voicing（選擇性）
    # 例：F1 = 2*TP / (2*TP + FP + FN)，這裡把 ref 的 voiced 當正類
    tp = (vr & ve).sum().item(); fp = (~vr & ve).sum().item(); fn = (vr & ~ve).sum().item()
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    print("Voicing F1:", f1, "\n")
    f1_vals.append(f1)

print("Mean F1:", sum(f1_vals) / len(f1_vals))
print("Mean RMSE cents:", sum(rmse_vals) / len(rmse_vals))
print("Mean CORR:", sum(corr_vals) / len(corr_vals))
