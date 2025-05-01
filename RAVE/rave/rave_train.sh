rave train \
    --config configs/v2.gin \
    --db_path /mnt/gestalt/home/buffett/rave/beatport_data_pp/ \
    --out_path /mnt/gestalt/home/buffett/rave/train_configs/ \
    --name beatport_rave \
    --channels 2 \
    --save_every 10000 \
    --gpu 3 \
    --batch 16 \
    --workers 24 \
    # --augment configs/augmentations/compress.gin \
    # --augment configs/augmentations/gain.gin \
    # --config configs/noise.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/mute.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/compress.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/gain.gin \
    
