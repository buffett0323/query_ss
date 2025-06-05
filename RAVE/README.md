# RAVE: Realtime Audio Variational autoEncoder
Author: Buffett

## Overview

RAVE is a state-of-the-art neural audio synthesis model that combines variational autoencoding with adversarial training. This guide provides comprehensive instructions for training a RAVE model using the MusicNet dataset. This is an experiment of training v2 RAVE without discriminator (no adversarial loss).<br>
Original github repo of RAVE is [here](https://github.com/acids-ircam/RAVE).<br>



## Model Architecture

The RAVE model consists of several key components:

| Component                 | Architecture Type      | Parameters |
|--------------------------|------------------------|------------|
| PQMF                     | CachedPQMF            | 16.7K      |
| Encoder                  | VariationalEncoder     | 3.9M       |
| Decoder                  | GeneratorV2            | 3.8M       |
| Discriminator            | CombineDiscriminators  | 6.8M       |
| Audio Distance           | AudioDistanceV1        | 0          |
| Multiband Audio Distance | AudioDistanceV1        | 0          |

**Total Model Statistics:**
- Trainable Parameters: 14.6M
- Non-trainable Parameters: 0
- Total Parameters: 14.6M
- Model Size: 58.284 MB

## Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Training Process

### 1. Dataset Preparation

Download and prepare the MusicNet dataset:

```bash
python downloader.py
```

### 2. Data Preprocessing

Transform the raw MusicNet data into RAVE-compatible format:

```bash
cd scripts/
bash rave_preprocess.sh
```

After preprocessing, verify the following metadata configuration:
```yaml
channels: 1
lazy: false
n_seconds: 121923.70938775511
sr: 44100
```

### 3. Model Training

Launch the training process using the v3 configuration:

```bash
cd scripts/
bash rave_train_single_gpu.sh
```

## Configuration

### Available Options

The training process can be customized through various parameters:

1. **Model Architecture**
   - Version selection (v1, v2, v3)
   - Component configurations
   - Layer specifications

2. **Training Parameters**
   - Learning rate
   - Batch size
   - Training duration
   - Loss weights

3. **Audio Processing**
   - Sample rate
   - Number of channels
   - Audio preprocessing settings

### Custom Configuration

For detailed configuration options and customization:
1. Examine the configuration files in the `configs/` directory
2. Modify the gin configuration files to adjust model behavior
3. Use command-line overrides for temporary changes

## Additional Resources

For more information about RAVE and its components, please refer to:
- Configuration files in `configs/`
- Model implementation details in source code
- Related research papers and documentation
