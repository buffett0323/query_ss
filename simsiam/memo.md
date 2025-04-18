### Inference note
During inference, first cd to ```BP_WGL128_CNN14.ckpt```, and run ```python zero_to_fp32.py ./ ./consolidated_model.pt```


### Training Commands
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_pl.py
```

BP Dataset 8 secs no resize
img_mean: -1.1041008234024048
img_std: 14.636213302612305

BP Dataset 8 secs slice 4 secs + resize 256
Mean: -1.100174903869629, Std: 14.353998184204102

BP Dataset 8 secs slice 4 secs + resize 224
Mean: -1.100771427154541, Std: 14.373129844665527


Simple BP Dataset 4 secs
norm_stats: [-5.36902,  3.7100384]

Best ResNet Dict path:
/mnt/gestalt/home/buffett/simsiam_model_dict/resnet_model_dict/checkpoint_0199.pth.tar