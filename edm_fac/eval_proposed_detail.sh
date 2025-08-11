CUDA_VISIBLE_DEVICES=1 python eval_proposed_detail.py \
    --device cuda:0 \
    --bs 32 \
    --config configs/config_proposed_no_ca.yaml \
    --checkpoint /mnt/gestalt/home/buffett/EDM_FAC_LOG/0804_proposed_no_ca/ckpt/checkpoint_latest.pt \
    --output_dir /mnt/gestalt/home/buffett/EDM_FAC_LOG/final_eval/0804_proposed_no_ca/detail/checkpoint_latest
