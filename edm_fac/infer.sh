python inference_conversion.py \
    --checkpoint /home/buffett/dataset/EDM_FAC_LOG/0602/ckpt/checkpoint_latest.pt \
    --config configs/config.yaml \
    --device cuda:2 \
    --input_dir /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/ \
    --output_dir sample_audio/ \
    --midi_dir /home/buffett/dataset/EDM_FAC_DATA/single_note_midi/evaluation/midi/ \
    --mode batch_convert \
    --amount 20 \
    # --target_audio /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/Lead\ -\ Atmos\ Bell_194447.wav \
    # --content_ref /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/LD\ -\ Vintage\ Slideshow_194447.wav \
    # --timbre_ref /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/Lead\ -\ Atmos\ Bell_74909.wav \
    # --output sample_audio/Lead\ -\ Atmos\ Bell_194447.wav \

# tensorboard --logdir=/home/buffett/dataset/EDM_FAC_LOG/0602/logs
