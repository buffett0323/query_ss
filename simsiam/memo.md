### Inference note
During inference, first cd to ```BP_WGL128_CNN14.ckpt```, and run ```python zero_to_fp32.py ./ ./consolidated_model.pt```


### Training Commands
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_pl.py
```

BP Dataset
Mean: -1.1039695739746094, Std: 14.636246681213379