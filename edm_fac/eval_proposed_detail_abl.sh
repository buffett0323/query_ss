python eval_proposed_detail_abl.py \
    --device cuda:3 \
    --bs 32 \
    --config configs/config_mn_ablation.yaml \
    --checkpoint /home/buffett/nas_data/EDM_FAC_LOG/0804_ablation/ckpt/checkpoint_latest.pt \
    --output_dir /home/buffett/nas_data/EDM_FAC_LOG/final_eval/0804_ablation/detail/checkpoint_latest
