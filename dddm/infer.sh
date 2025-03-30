python inference.py \
    --src_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/aa4d58e8-4718-4fdf-812d-e9cba20821da_1/other.npy' \
    --trg_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/496a7e07-43ab-41c2-8415-431410d7d72f_1/other.npy' \
    --ckpt_model '/mnt/gestalt/home/buffett/tt_training/timbre_transfer_te_train_dict/G_110000.pth' \
    --ckpt_voc '/mnt/gestalt/home/buffett/hifigan_ckpt/voc_ckpt.pth' \
    --logs_path '/mnt/gestalt/home/buffett/timbre_transfer_logs' \
    --output_dir '/mnt/gestalt/home/buffett/tt_converted' \
    -t 6 \
    -d 'cuda:1'