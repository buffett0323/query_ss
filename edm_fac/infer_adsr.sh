python infer_adsr_short2long.py \
    --checkpoint /mnt/gestalt/home/buffett/EDM_FAC_LOG/0707_ss/ckpt/checkpoint_220000.pt \
    --config configs/config_ss.yaml \
    --orig_audio /mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA/rendered_ss_t_adsr_c/evaluation/T000_ADSR016_C015.wav \
    --ref_audio /mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/train/T001_ADSR001_C001.wav \
    --gt_audio /mnt/gestalt/home/buffett/EDM_FAC_NEW_DATA/rendered_ss_t_adsr_c/evaluation/T000_ADSR001_C015.wav \
    --output_dir short2long_audio/ \
    --convert_type adsr \
    --device cuda:1 \
    --prefix "mn_adsr_"
