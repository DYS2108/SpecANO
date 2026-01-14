# Specctral-Guided Volumetric Restoration and Anomaly Detection of Medical Images



<p align="center">
<img src=assets/fig_flowchart2.png />
</p>

## Requirements
A suitable [conda](https://conda.io/) environment named `Spec-ANO` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate Spec-ANO
```

# Pretrained Models
A general list of all available checkpoints is available in via our [model zoo](#model-zoo).



To get started, install the additionally required python packages into your `Spec-ANO` environment
```shell script
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```


## Model Training


Codes for spectral bases approximation  are in './spectral embedding'

Logs and checkpoints for trained models are saved to `logs`.

### Training autoencoder models

Configs for training a KL-regularized autoencoder on chest and brain CT are provided at `configs/autoencoder`.
Training can be started by running
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/<config_spec>.yaml -t --gpus 0,    
```


### Training LDMs 

In ``configs/latent-diffusion/`` we provide configs for training LDMs on chest and brain CTs. 
Training can be started by running

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/<config_spec>.yaml -t --gpus 0,
``` 

# Model Zoo 


### Get the models

The LDMs listed above can jointly be downloaded and extracted via

```shell script
bash scripts/download_models.sh
```

The models can then be found in `models/ldm/<model_spec>`.





