
# Wan2.1 MI325X AAC-platform Multi-nodes Training

## 1. Envs
### Code
```bash
git clone git@github.com:qiongz/DiffSynth-Studio.git
cd DiffSynth-Studio
```

### Data & Models
download data & models following https://github.com/qiongz/DiffSynth-Studio/blob/main/examples/wanvideo/README.md
- `data/example_video_dataset`
- `models/WanAI`
or create soft-links 
```bash
ln -s /path/to/data ./data
ln -s /path/to/models ./models
```

## 2. Slurm jobs
Use sbatch to submit jobs
```
sbatch scripts/submit_dist.slurm
```
Importance parameters:
- `--reservation=gpu-22_gpu-18_gpu-16_gpu-9_reservation`: reservation group
- `--partition=256C8G1H_MI325X_Ubuntu22`: reservation partition
- `--nodelist=gpu-18,gpu-9,gpu-22`: nodes for training
- `export IMAGE_TAR="/shared/data/FX2/rocm6.4.3_ubuntu22.04_py3.10_pytorch_2.8.0_ck_fa3_fremont_dist_250917.tar"`: DOCKER tar path, ask administrator for read permission
- `export DOCKER_IMAGE="docker.io/rocm/ali-private:rocm6.4.3_ubuntu22.04_py3.10_pytorch_2.8.0_ck_fa3_fremont_dist_250917"`: DOCKER image tag, corresponding to IMAGE_TAR
- `export TRAIN_SCRIPT_IN_CONTAINER="examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P_dist.sh"`: training script


### logs
slurm log & training log for each node
```
./logs/
├── 2025-09-18_22-30-42
│   ├── node_0
│   │   └── output.log
│   ├── node_1
│   │   └── output.log
│   └── node_2
│       └── output.log
└── slurm
   └── mi325x-dist.3877.out
```
