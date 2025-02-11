CUDA_VISIBLE_DEVICES=0,1,2,3 python train_new.py \
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
    --fix-pred-lr \
    # [your imagenet-folder with train and val folders]