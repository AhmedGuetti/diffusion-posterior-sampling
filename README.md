# Technical project based on the paper:  [Diffusion Posterior Sampling for General Noisy Inverse Problems (ICLR 2023 spotlight)][arxiv-paper]
[arxiv-paper]: https://arxiv.org/abs/2503.10237

<strong>By:</strong> Alexandra Villon Huaman & Ahmed Guetti (Université de Toulouse, France)
## 1 Resume of the methodology:
This article proposes a diffusion-based reconstruction technique for image restoration. This method relies on iterative estimation of the undegraded image $\hat{\mathbf{x}}_0$, obtained by combining two sources of information. First, a pre-trained UNet network, capable of generating realistic images, provides prior information. Second, the approach considers the likelihood of the data, determined by an image degradation model with a known direct measurement operator. Combining these pieces of information enables progressive improvement in the quality of the reconstructed image.

In summary, we are looking for the most probable images $x_{0}$ sampled according to the posterior distribution: 

$p_t(x_t|y) = \frac{p_t(x_t)p_t(y|x_t)}{p_t(y)}$.

## 2 Prerequisites
- python 3.8
- pytorch 1.11.0
- CUDA 11.3.1
- nvidia-docker (if you use a GPU in docker container)

According the autor, you can use lower version of CUDA with proper pytorch version.
Ex) CUDA 10.2 with pytorch 1.7.0

## 3 How to Run the Code:

### A) Clone the repository

```
git clone git@github.com:AhmedGuetti/diffusion-posterior-sampling.git
cd diffusion-posterior-sampling
```
### B) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

### C) Environnement Setup:

### Local environment setting

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## 4) Inference 
### 4.1) Applied to Photographic images

```
python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config={TASK-CONFIG};
```
#### a. Possible task configurations

```
# Linear inverse problems
- configs/super_resolution_config.yaml
- configs/gaussian_deblur_config.yaml
- configs/motion_deblur_config.yaml
- configs/inpainting_config.yaml

# Non-linear inverse problems
- configs/nonlinear_deblur_config.yaml
- configs/phase_retrieval_config.yaml
```
#### b. Fixing some issues 
In order to process non-linear deblurring, it is necessary to make certain modifications to the following relative paths: 
- bkse/models/backbones/resnet.py
- bkse/models/kernel_encoding/kernel_wizard.py
- bkse/options/generate_blur/default.yml
The corrected versions of the code can be found in the folder named fix.

#### Structure of task configurations
You need to write your data directory at data.root. Default is ./data/samples which contains three sample images from FFHQ validation set.

```
conditioning:
    method: # check candidates in guided_diffusion/condition_methods.py
    params:
        scale: 0.5

data:
    name: ffhq
    root: ./data/samples/

measurement:
    operator:
        name: # check candidates in guided_diffusion/measurements.py

noise:
    name:   # gaussian or poisson
    sigma:  # if you use name: gaussian, set this.
    (rate:) # if you use name: poisson, set this.
```
### 4.2) Applied to Ultrasound Images
```
python3 main.py \
--model_config configs/model_config.yaml \
--diffusion_config configs/diffusion_config.yaml \
--task_config configs/ultrasound_config.yaml
```
Here, we adapted the approach to handle ultrasound images using the forward measurement model and an estimated psf.

## 5 Test Results in Photographic Images:
### Super-resolution Task
<p align="center">
  <img width="607" height="605" alt="image" src="https://github.com/user-attachments/assets/ade5c6d8-e299-43ed-95bf-d1d62cf368ba" /> 
<br>
  From left to right: input image, reconstructed image, ideal image.
</p>

### Gaussian Deblurring Task
<p align="center">
 <img width="730" height="734" alt="image" src="https://github.com/user-attachments/assets/202f1352-52bd-40fd-a20e-fc3656b8a68f" />
 <br>
  From left to right: input image, reconstructed image, ideal image.
</p>

### Phase Retrieval Task
<p align="center">
<img width="602" height="605" alt="image" src="https://github.com/user-attachments/assets/3366cf2f-c66f-4d84-92ba-eb12de6ecc69" />
<br>
  From left to right: input image, reconstructed image, ideal image.
</p>

## 6 Test Results in Ultrasound Images
### Simulated images
<p align="center">
<img width="856" height="435" alt="image" src="https://github.com/user-attachments/assets/eb97e041-bc84-4cf6-94c4-3569e808edee" />
<br>
  From left to right: input image, reconstructed image, ideal image.
</p>

### In vivo images
<p align="center">
<img width="853" height="382" alt="image" src="https://github.com/user-attachments/assets/b9bb627b-58d8-452c-945d-02dbcbaf1637" />

<br>
  From left to right: input image, reconstructed image, ideal image.
</p>

## Citation

```
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```

