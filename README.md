# A Recycling Training Strategy for Medical Image Segmentation with Diffusion Denoising Models

:tada: This is a follow-up work of Importance of Aligning Training Strategy with
Evaluation for Diffusion Models in 3D Multiclass Segmentation
([paper](https://arxiv.org/abs/2303.06040),
[code](https://github.com/mathpluscode/ImgX-DiffSeg/tree/v0.1.0)), with better
recycling method, better network, more baseline training methods (including
self-conditioning) on four data sets (muscle ultrasound, male pelvic MR,
abdominal CT, brain MR).

:bookmark_tabs: The preprint is available on
[arXiv](https://arxiv.org/abs/2308.16355).

<div>
<img src="images/diffusion_training_strategy_diagram.png" width="600" alt="diffusion_training_strategy_diagram"></img>
</div>

---

ImgX is a Jax-based deep learning toolkit for biomedical image segmentations.

Current supported functionalities are summarized as follows.

**Data sets**

See the [readme](imgx_datasets/README.md) for details on training, validation,
and test splits.

- [x] Muscle ultrasound from
      [Marzola et al. 2021](https://data.mendeley.com/datasets/3jykz7wz8d/1).
- [x] Male pelvic MR from
      [Li et al. 2022](https://zenodo.org/record/7013610#.Y1U95-zMKrM).
- [x] AMOS CT from
      [Ji et al. 2022](https://zenodo.org/record/7155725#.ZAN4BuzP2rO).
- [x] Brain MR from [Baid et al. 2021](https://arxiv.org/abs/2107.02314).

**Algorithms**

- [x] Supervised segmentation.
- [x] Diffusion-based segmentation.
  - [x] Gaussian noise based diffusion.
  - [x] Prediction of noise or ground truth.
  - [x] Training with recycling or self-conditioning.

**Models**

- [x] U-Net with Transformers supporting 2D and 3D images.

**Training**

- [x] Patch-based training.
- [x] Multi-device training (one model per device).
- [x] Mixed precision training.
- [x] Gradient clipping and accumulation.

---

## Installation

### TPU with Docker

The following instructions have been tested only for TPU-v3-8. The docker
container uses root user.

1. Build the docker image inside the repository.

   ```bash
   sudo docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile.tpu -t imgx .
   ```

   where

   - `--build-arg` provides argument values.
   - `-f` provides the docker file.
   - `-t` tag the docker image.

2. Run the Docker container.

   ```bash
   mkdir -p $(cd ../ && pwd)/tensorflow_datasets
   sudo docker run -it --rm --privileged --network host \
   -v "$(pwd)":/app/ImgX \
   -v "$(cd ../ && pwd)"/tensorflow_datasets:/root/tensorflow_datasets \
   imgx bash
   ```

3. Install the package inside container.

   ```bash
   make pip
   ```

TPU often has limited disk space.
[RAM disk](https://www.linuxbabe.com/command-line/create-ramdisk-linux) can be
used to help.

```bash
sudo mkdir /tmp/ramdisk
sudo chmod 777 /tmp/ramdisk
sudo mount -t tmpfs -o size=256G imgxramdisk /tmp/ramdisk
cd /tmp/ramdisk/
```

### GPU with Docker

The following instructions have been tested only for CUDA == 11.4.1 and CUDNN ==
8.2.0. The docker container uses non-root user.
[Docker image used may be removed.](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md)

1. Build the docker image inside the repository.

   ```bash
   docker build --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) -f docker/Dockerfile -t imgx .
   ```

   where

   - `--build-arg` provides argument values.
   - `-f` provides the docker file.
   - `-t` tag the docker image.

2. Run the Docker container.

   ```bash
   mkdir -p $(cd ../ && pwd)/tensorflow_datasets
   docker run -it --rm --gpus all \
   -v "$(pwd)":/app/ImgX \
   -v "$(cd ../ && pwd)"/tensorflow_datasets:/home/app/tensorflow_datasets \
   imgx bash
   ```

   where

   - `--rm` removes the container once exit it.
   - `-v` maps the `ImgX` folder into container.

3. Install the package inside container.

   ```bash
   make pip
   ```

### Local with Conda

#### Install Conda for Mac M1

[Download Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
from [GitHub](https://github.com/conda-forge/miniforge) and install it.

```bash
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba
conda env update -f docker/environment_mac_m1.yml
```

#### Install Conda for Linux / Mac Intel

[Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
and then create the environment.

```bash
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba
conda env update -f docker/environment.yml
```

#### Activate Conda Environment

Activate the environment and install the package.

```bash
conda activate imgx
make pip
```

## Build Data Sets

Use the following commands to (re)build all data sets. Check the
[README](imgx_datasets/README.md) of imgx_datasets for details. Especially,
manual downloading is required for the BraTS 2021 dataset.

```bash
make build_dataset
make rebuild_dataset
```

## Experiment

### Training and Testing

Example command to use two GPUs for training, validation and testing. The
outputs are stored under `wandb/latest-run/files/`, where

- `ckpt` stores the model checkpoints and corresponding validation metrics.
- `test_evaluation` stores the prediction on test set and corresponding metrics.

```bash
# limit to two GPUs if using NVIDIA GPUs
export CUDA_VISIBLE_DEVICES="0,1"
# select data set to use
export DATASET_NAME="male_pelvic_mr"
export DATASET_NAME="amos_ct"
export DATASET_NAME="muscle_us"
export DATASET_NAME="brats2021_mr"

# Vanilla segmentation
imgx_train data=${DATASET_NAME} task=seg
imgx_valid --log_dir wandb/latest-run/
imgx_test --log_dir wandb/latest-run/

# Diffusion-based segmentation
imgx_train data=${DATASET_NAME} task=gaussian_diff_seg
imgx_valid --log_dir wandb/latest-run/ --num_timesteps 5 --sampler DDPM
imgx_test --log_dir wandb/latest-run/ --num_timesteps 5 --sampler DDPM
imgx_valid --log_dir wandb/latest-run/ --num_timesteps 5 --sampler DDIM
imgx_test --log_dir wandb/latest-run/ --num_timesteps 5 --sampler DDIM
```

```bash
imgx_test --log_dir wandb/latest-run/ --num_timesteps 5 --num_seeds 3
```

Optionally, for debug purposes, use flag `debug=True` to run the experiment with
a small dataset and smaller models.

```bash
imgx_train --config-name config_${DATASET_NAME}_seg debug=True
imgx_train --config-name config_${DATASET_NAME}_diff_seg debug=True
```

## Code Quality

### Pre-commit

Install pre-commit hooks:

```bash
pre-commit install
wily build .
```

Update hooks, and re-verify all files.

```bash
pre-commit autoupdate
pre-commit run --all-files
```

### Code Test

Run the command below to test and get coverage report. As JAX tests requires two
CPUs, `-n 4` uses 4 threads, therefore requires 8 CPUs in total.

```bash
pytest --cov=imgx -n 4 tests
```

## References

- [Segment Anything (PyTorch)](https://github.com/facebookresearch/segment-anything)
- [MONAI (PyTorch)](https://github.com/Project-MONAI/MONAI/)
- [Cross Institution Few Shot Segmentation (PyTorch)](https://github.com/kate-sann5100/CrossInstitutionFewShotSegmentation/)
- [MegSegDiff (PyTorch)](https://github.com/WuJunde/MedSegDiff/)
- [MegSegDiff (PyTorch, lucidrains)](https://github.com/lucidrains/med-seg-diff-pytorch/)
- [DeepReg (Tensorflow)](https://github.com/DeepRegNet/DeepReg/)
- [Scenic (JAX)](https://github.com/google-research/scenic/)
- [DeepMind Research (JAX)](https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc/)
- [Haiku (JAX)](https://github.com/deepmind/dm-haiku/)

## Acknowledgement

This work was supported by the EPSRC grant (EP/T029404/1), the Wellcome/EPSRC
Centre for Interventional and Surgical Sciences (203145Z/16/Z), the
International Alliance for Cancer Early Detection, an alliance between Cancer
Research UK (C28070/A30912, C73666/A31378), Canary Center at Stanford
University, the University of Cambridge, OHSU Knight Cancer Institute,
University College London and the University of Manchester, and Cloud TPUs from
Google's TPU Research Cloud (TRC).

## Citation

If you find the code base and method useful in your research, please cite the
relevant paper:

```bibtex
@article{fu2023recycling,
  title={A Recycling Training Strategy for Medical Image Segmentation with Diffusion Denoising Models},
  author={Fu, Yunguan and Li, Yiwen and Saeed, Shaheer U and Clarkson, Matthew J and Hu, Yipeng},
  journal={arXiv preprint arXiv:2308.16355},
  year={2023},
  doi={10.48550/arXiv.2308.16355},
  url={https://arxiv.org/abs/2308.16355},
}

@article{fu2023importance,
  title={Importance of Aligning Training Strategy with Evaluation for Diffusion Models in 3D Multiclass Segmentation},
  author={Fu, Yunguan and Li, Yiwen and Saeed, Shaheer U and Clarkson, Matthew J and Hu, Yipeng},
  journal={arXiv preprint arXiv:2303.06040},
  year={2023},
  doi={10.48550/arXiv.2303.06040},
  url={https://arxiv.org/abs/2303.06040},
}
```
