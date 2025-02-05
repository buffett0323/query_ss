python inference.py \
    --src_path './examples/orig_y.wav' \
    --trg_path './examples/orig_y2.wav' \
    --ckpt_model './logs/MD/G_180000.pth' \
    --ckpt_voc './checkpoints/voc_ckpt.pth' \
    --output_dir './converted' \
    -t 6 \
    -d 'cuda:0'