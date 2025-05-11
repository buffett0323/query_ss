python train.py \
    --gpu 2 \
    --config ../rave/configs/v2.gin \
    --db_path /mnt/gestalt/home/buffett/rave/beatport_data_pp/4secs/ \
    --out_path /mnt/gestalt/home/buffett/rave/train_configs/ \
    --name beatport_rave \
    --channels 1 \
    --save_every 10000 \
    --batch 16 \
    --workers 24 \
    --sr 44100 \
    # --n_signal 131072 \ # = 1024 * 128
    # --augment configs/augmentations/compress.gin \
    # --augment configs/augmentations/gain.gin \
    # --config configs/noise.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/mute.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/compress.gin \
    # --augment /home/buffett/research/query_ss/RAVE/rave/configs/augmentations/gain.gin \
    
