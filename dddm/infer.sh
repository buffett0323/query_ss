python inference.py \
    --src_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/97d025c8-6070-4bc0-b602-66e392f26893_4/other.npy' \
    --trg_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/d3054b25-c025-45ca-9d78-f49b4c8b850d_1/other.npy' \
    --ckpt_model '/mnt/gestalt/home/buffett/timbre_transfer_dict/G_200000.pth' \
    --ckpt_voc '/mnt/gestalt/home/buffett/hifigan_ckpt/voc_ckpt.pth' \
    --logs_path '/mnt/gestalt/home/buffett/timbre_transfer_logs' \
    --output_dir '/mnt/gestalt/home/buffett/tt_converted' \
    -t 6 \
    -d 'cuda:0'