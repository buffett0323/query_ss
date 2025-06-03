CUDA_VISIBLE_DEVICES=2,3,4 torchrun --nproc_per_node=3 train.py \
    --config ../rave/configs/v3.gin \
    --db_path /home/buffett/dataset/rave/beatport_data_pp/4secs/ \
    --out_path /home/buffett/dataset/rave/train_configs/ \
    --name beatport_rave \
    --channels 1 \
    --save_every 10000 \
    --batch 128 \
    --workers 24 \
    --sr 44100 \
    --devices 3 \
    --strategy ddp
    # --n_signal 131072 \ # = 1024 * 128
    # --augment configs/augmentations/compress.gin \
    # --augment configs/augmentations/gain.gin \
    # --config configs/noise.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/mute.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/compress.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/gain.gin \
