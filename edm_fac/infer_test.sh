python infer_adsr_short2long.py \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0723_mn_cross_attn_add_adsr/ckpt/checkpoint_latest.pt \
    --config configs/config_mn_cross_attn_add_adsr.yaml \
    --orig_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR001_C028.wav \
    --ref_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR000_C028.wav \
    --gt_audio /home/buffett/nas_data/EDM_FAC_NEW_DATA/rendered_mn_t_adsr_c/evaluation/T010_ADSR000_C028.wav \
    --output_dir 0723_same_onset_long2short/ \
    --convert_type both \
    --device cuda:2 \
    --prefix "both_"
