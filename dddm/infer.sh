python inference.py \
    --src_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/orig_y.wav' \
    --trg_path '/mnt/gestalt/home/ddmanddman/beatport_analyze/chorus_audio_16000_8secs_npy/orig_y2.wav' \
    --ckpt_model '/mnt/gestalt/home/buffett/timbre_transfer_dict/G_200000.pth' \
    --ckpt_voc '/mnt/gestalt/home/buffett/hifigan_ckpt/voc_ckpt.pth' \
    --output_dir '/mnt/gestalt/home/buffett/tt_converted' \
    -t 6 \
    -d 'cuda:0'