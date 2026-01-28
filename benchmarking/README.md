# DiffSynth‑Studio — Training Performance (Perf) One‑Click Runner & E2E Parser

---

## Quickstart (Training Performance)

```bash
# 0) Installation 
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .

# 1) Prepare Datasets & Models
## dataset 
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
## Models
(checkout "Model Asset" )

# 2) （可选）指定要跑的模型清单（与 configs/*.yaml 文件名一致，不含 .yaml）
cat > available_models.txt << 'EOF'
Wan2.1-I2V-14B-480P-81F
Wan2.1-I2V-14B-720P-121F
Wan2.1-I2V-14B-720P-49F
Wan2.1-T2V-1.3B-81F
Wan2.1-T2V-14B-81F
Wan2.2-I2V-A14B-high-noise-480P-49F
Wan2.2-I2V-A14B-high-noise-720P-161F
Wan2.2-T2V-A14B-high-nosie-480P-49F
Wan2.2-TI2V-5B-480P-81F
EOF

# 3) 跑测试对比
默认跑 8-gpu configs (日志落在 logs/<MACHINE>-8g）
# MI325X
python3 benchmarking/run_models.py --machine MI325X --configs-dir benchmarking/configs/ --available-list benchmarking/available_models.txt
# H200
python3 benchmarking/run_models.py --machine MI325X --configs-dir benchmarking/configs/ --available-list benchmarking/available_models.txt

# 4) 解析日志并生成对比表
python3 parse_e2e_and_make_table.py \
  --log-root logs \
  --prefer-gpus 8 1 \
  --last-n 10 \
  --raw-csv e2e_raw_points.csv \
  --summary-csv e2e_summary.csv \
  --summary-md e2e_summary.md \
  --by-mg-csv e2e_by_machine_gpu.csv \
  --by-mg-md e2e_by_machine_gpu.md
```

---

## Prerequisites

---

## Model Assets（手动下载 / Manual Download）

推荐将所需权重预先下载或同步至本地（推荐以下布局）：
```
DiffSynth-Studio/models/Wan-AI/
  ├── Wan2.1-T2V-1.3B/
  ├── Wan2.1-T2V-14B/
  ├── Wan2.1-I2V-14B/
  └── ...
```


按需下载
```
pip install "huggingface_hub[cli]"
cd DiffSynth-Studio
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan-AI/Wan2.1-T2V-1.3B
```

> 📌 **YAML 中的 `model_id_with_origin_paths`** 需与本地权重路径一致，确保训练脚本可正确加载。

---

## YAML Configs（最小字段速览 / Minimal Schema）

每个 `configs/<MODEL>.yaml` 将映射为训练 CLI 参数（布尔开关会转为**无值 flag**）：

```yaml
name: Wan2.1-T2V-1.3B-81F

# dataset
dataset_base_path: data/example_video_dataset
dataset_metadata_path: data/example_video_dataset/metadata.csv

# spatial/temporal
height: 480
width: 832
num_frames: 81

# training
dataset_repeat: 128
warmup_steps: 3
learning_rate: 1e-5
num_epochs: 1
seed: 10007
trainable_models: "dit"
remove_prefix_in_ckpt: "pipe.dit."
output_path: "./models/train/Wan2.1-T2V-1.3B_full"

# model weights mapping
model_id_with_origin_paths: "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"

# deepspeed & flags
deepspeed: true
deepspeed_config: "examples/wanvideo/model_training/full/deepspeed_stage1_config.json"
use_gradient_checkpointing: true
use_gradient_checkpointing_offload: false
```

---

## One‑Click Runner（`run_models.py`）

### Basic Usage

```bash
# 在 H200 上跑 8gpu
cd DiffSynth-Studio
python3 benchmarking/run_models.py --machine H200  --gpus 8 --configs-dir benchmarking/configs --available-list benchmarking/available_models.txt

# 在 MI325X 上顺序跑 1g 与 8g
cd DiffSynth-Studio
python3 benchmarking/run_models.py --machine MI325X --gpus 1 8 --configs-dir benchmarking/configs --available-list benchmarking/available_models.txt

```

###  Arguments

- `--machine {H200|MI325X}`：指定平台并用于日志路径命名（`logs/<MACHINE>-<Ng>g/...`）。不指定则尝试自动识别（NVIDIA→H200，ROCm→MI325X）。  
- `--gpus N [N2 ...]`：每个模型跑的 GPU 数，**串行**执行（例如 `--gpus 1 8`）。  
- `--configs-dir PATH`：模型 YAML 目录（默认 `configs`）。  
- `--available-list PATH`：可选，仅跑列表中的模型（与 YAML 文件名一致，去 `.yaml`）。默认 `available_models.txt`；缺省则跑 `configs/` 下所有 YAML。  
- `--dataset-base-path PATH` / `--dataset-metadata-path PATH`：全局覆盖数据路径（优先于 YAML）。  
- `--seed INT`：全局覆盖 seed。  
- `--entry ...`：覆盖训练入口（默认 `deepspeed examples/wanvideo/model_training/train.py`）。  
- `--dry-run`：仅输出命令与写入 `cmd.txt`，不实际执行。

###  Logs Layout

```
logs/
  H200-1g/
    <MODEL_NAME>/train.log
    <MODEL_NAME>/cmd.txt
  H200-8g/
    <MODEL_NAME>/train.log
    <MODEL_NAME>/cmd.txt
  MI325X-1g/
    ...
  MI325X-8g/
    ...
```

> 每次运行会将 stdout/stderr 同时输出到控制台与 `train.log`（等价 tee）。

---

## Parse & Compare（`parse_e2e_and_make_table.py`，训练性能对比）

该脚本会**自动识别**以下目录形式：`logs/<MACHINE>` 与 `logs/<MACHINE>-<Ng>g`（如 `H200-8g`）。
默认将**优先选择 8g** 结果作为主对比表数据（可通过 `--prefer-gpus` 调整顺序）。

### Usage

```bash
python3 parse_e2e_and_make_table.py \
  --log-root logs \
  --prefer-gpus 8 1 \
  --last-n 5 \
  --raw-csv e2e_perf_raw_points.csv \
  --summary-csv e2e_perf_summary.csv \
  --summary-md e2e_perf_summary.md \
  --by-mg-csv e2e_perf_by_machine_gpu.csv \
  --by-mg-md e2e_perf_by_machine_gpu.md
```

### Outputs

- `e2e_perf_raw_points.csv`：每个日志文件的 e2e 序列与中位数（最后 N 个样本，默认 N=5）。  
- `e2e_perf_summary.csv`：主对比表（每平台为每个模型选定一条结果，默认优先 8g）。  
- `e2e_perf_summary.md`：主对比表（Markdown）。**相对性能定义**：`MI325X vs. H200 (%) = (H200_e2e / MI325X_e2e) * 100`。e2e 是耗时，越小越好，故 **>100% 表示 MI325X 更快**。  
- `e2e_perf_by_machine_gpu.csv`：平台×GPU 数的长表（含所有可用组合）。  
- `e2e_perf_by_machine_gpu.md`：平台×GPU 数概览表（Markdown）。

---

## Models List

- `Wan2.1-I2V-14B-480P-81F`
- `Wan2.1-I2V-14B-720P-121F`
- `Wan2.1-I2V-14B-720P-49F`
- `Wan2.1-T2V-1.3B-81F`
- `Wan2.1-T2V-14B-81F`
- `Wan2.2-I2V-A14B-high-noise-480P-49F`
- `Wan2.2-I2V-A14B-high-noise-720P-161F`
- `Wan2.2-T2V-A14B-high-nosie-480P-49F`
- `Wan2.2-TI2V-5B-480P-81F`

**示例 `available_models.txt`**：

```text
Wan2.1-I2V-14B-480P-81F
Wan2.1-I2V-14B-720P-121F
Wan2.1-I2V-14B-720P-49F
Wan2.1-T2V-1.3B-81F
Wan2.1-T2V-14B-81F
Wan2.2-I2V-A14B-high-noise-480P-49F
Wan2.2-I2V-A14B-high-noise-720P-161F
Wan2.2-T2V-A14B-high-nosie-480P-49F
Wan2.2-TI2V-5B-480P-81F
```

---

## Training Performance Results

> 跑完并执行解析脚本后，请以实际生成的 `e2e_perf_summary.md` 与 `e2e_perf_by_machine_gpu.md` 为准。

** 测试结果 **

| Model | MI325X vs. H200 (Perf) | MI325X e2e_s (s/iter) | H200 e2e_s (s/iter) | Notes |
|---|---:|---:|---:|---|
| Wan2.1-I2V-14B-480P-81F | 100% | 16.199 | 16.255 |  |
| Wan2.1-I2V-14B-720P-121F | 124% | 110.692 | 137.195 |  |
| Wan2.1-I2V-14B-720P-49F | 98% | 27.325 | 26.671 |  |
| Wan2.1-T2V-1.3B-81F | 105% | 3.298 | 3.450 |  |
| Wan2.1-T2V-14B-81F | 98% | 15.176 | 14.841 |  |
| Wan2.2-I2V-A14B-high-noise-480P-81F | 99% | 16.138 | 16.054 |  |
| Wan2.2-I2V-A14B-high-noise-720P-161F | 119% | 180.178 | 214.628 |  |
| Wan2.2-T2V-A14B-high-nosie-480P-49F | 102% | 7.949 | 8.133 |  |
| Wan2.2-TI2V-5B-480P-81F | 116% | 1.595 | 1.845 |  |


### Docker Version

| System | docker | FA-version| 
| :--- | :--- | :--- |
| MI325X| amdagi/rocm_pytorch-training:v25.11 | FA2 (aiter-v0.1.9)|
| H200 | nvcr.io/nvidia/pytorch:25.12-py3 | FA3-hopper|

