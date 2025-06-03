python inference_conversion.py \
    --checkpoint /home/buffett/dataset/EDM_FAC_LOG/0602/ckpt/checkpoint_latest.pt \
    --config configs/config.yaml \
    --target_audio /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/Lead\ -\ Atmos\ Bell_194447.wav \
    --content_ref /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/LD\ -\ Vintage\ Slideshow_194447.wav \
    --timbre_ref /home/buffett/dataset/EDM_FAC_DATA/rendered_audio_new/lead/Lead\ -\ Atmos\ Bell_74909.wav \
    --output sample_audio/Lead\ -\ Atmos\ Bell_194447.wav \
    --device cuda:2
# tensorboard --logdir=/home/buffett/dataset/EDM_FAC_LOG/0602/logs
