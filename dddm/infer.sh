python inference.py \
    --src_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/c6c011af-d6e8-41d5-9980-a44e5efcc972_1/other.npy' \
    --trg_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/4303f5a0-377e-4a71-8d2f-3898641df854_1/other.npy' \
    --ckpt_model '/mnt/gestalt/home/buffett/timbre_transfer_dict/G_200000.pth' \
    --ckpt_voc '/mnt/gestalt/home/buffett/hifigan_ckpt/voc_ckpt.pth' \
    --logs_path '/mnt/gestalt/home/buffett/timbre_transfer_logs' \
    --output_dir '/mnt/gestalt/home/buffett/tt_converted' \
    -t 6 \
    -d 'cuda:0'