#!/bin/bash
#SBATCH --job-name=mi325x-dist-train-example
#SBATCH --output=logs/slurm/mi325x-dist.%j.out
#SBATCH --nodes=2
# 不手写 --nodelist，避免主机名解析问题；完全依赖 reservation 自动分配
# #SBATCH --nodelist=gpu-9,gpu-22
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --reservation=gpu-22_gpu-18_gpu-16_gpu-9_reservation

set -euo pipefail

echo "JobID: $SLURM_JOB_ID"

# ========== 0) docker & 训练脚本参数 ==========
PROJECT_DIR_ON_HOST="$HOME/codes/DiffSynth-Studio"
PROJECT_DIR_IN_CONTAINER="/workdir/codes/DiffSynth-Studio"
TRAIN_SCRIPT_IN_CONTAINER="examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P_dist.sh"
DOCKER_IMAGE="docker.io/rocm/ali-private:rocm6.4.3_ubuntu22.04_py3.10_pytorch_2.8.0_ck_fa3_fremont_dist_250917"
CONTAINER_CMD="podman"

# docker 下传递 data, models 等目录软连接
export WORKDIR=$HOME

# 共享运行目录（所有节点可见）
RUN_DIR="$PROJECT_DIR_ON_HOST/.slurm_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# 时间戳（节点日志分目录）
TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "$TIME_STAMP"

# ========== 1) 获取实际分配的节点名 ==========
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#HOSTS[@]}" -lt 2 ]; then
  echo "分配到的节点不足 2 个: ${HOSTS[*]}"
  exit 1
fi
echo "分配到的节点: ${HOSTS[*]}"

# ========== 2) 解析主机名 -> IP（用 IP 更稳） ==========
HOST_IPS=()
for h in "${HOSTS[@]}"; do
  ip=$(getent hosts "$h" | awk '{print $1; exit}')
  if [[ -z "${ip:-}" ]]; then
    echo "解析失败：getent hosts $h 无返回。请让管理员在 DNS 或 /etc/hosts 配置 $h。"
    exit 1
  fi
  HOST_IPS+=("$ip")
done
echo "对应 IP: ${HOST_IPS[*]}"

# ========== 3) 设置 Master（用第一个 IP）& 生成 DeepSpeed hostfile ==========
export MASTER_ADDR="${HOST_IPS[0]}"
export MASTER_PORT="${MASTER_PORT:-29475}"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"

HOSTFILE_PATH="$PROJECT_DIR_ON_HOST/examples/wanvideo/model_training/full/hostfile"
mkdir -p "$(dirname "$HOSTFILE_PATH")"
: > "$HOSTFILE_PATH"
for ip in "${HOST_IPS[@]}"; do
  echo "$ip slots=8" >> "$HOSTFILE_PATH"
done
echo "Hostfile 内容："
cat "$HOSTFILE_PATH"

# ========== 4) 写入共享文件，供各节点读取 ==========
HOSTS_FILE="$RUN_DIR/hosts.txt"
IPS_FILE="$RUN_DIR/ips.txt"
printf "%s\n" "${HOSTS[@]}"    > "$HOSTS_FILE"
printf "%s\n" "${HOST_IPS[@]}" > "$IPS_FILE"

# ========== 5) IB/RoCE 环境变量 ==========
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

# 若要固定：提交时带 `FIXED_NIC=enp49s0f1np1 sbatch run_deepspeed_dist.sbatch`
#export FIXED_NIC="${FIXED_NIC:-enp49s0f1np1}" # lmx5
export FIXED_NIC="${FIXED_NIC:-enp49s0f0np0}"  # ROCE

# ========== 6) 启动（每节点一个容器；容器内 deepspeed --no_ssh 起本地 8 进程） ==========
# 说明：我们依赖 #SBATCH 的资源分配，这里不重复指定 --nodes/--gres
srun --kill-on-bad-exit=1 bash -c '
  set -euo pipefail

  # 读取共享文件（所有节点可见）
  HOSTS_FILE="'"$HOSTS_FILE"'"
  IPS_FILE="'"$IPS_FILE"'"

  readarray -t HOSTS < "$HOSTS_FILE"
  readarray -t HOST_IPS < "$IPS_FILE"

  export NODE_RANK="$SLURM_NODEID"
  export NNODES="$SLURM_NNODES"
  DOCKER_NAME="dist-node-${SLURM_JOB_ID}-${NODE_RANK}"

  NODE_NAME="${HOSTS[$SLURM_NODEID]}"
  NODE_IP="${HOST_IPS[$SLURM_NODEID]}"

  LOG_DIR="logs/'"$TIME_STAMP"'/node_${NODE_RANK}"
  mkdir -p "$LOG_DIR"

  echo "当前节点（NODE_RANK=$NODE_RANK）: ${NODE_NAME} (IP: ${NODE_IP})"

  # 选择对端节点的 IP 用于路由探测（避免 master==本机返回 lo）
  # 这里是 2 节点：0 的对端是 1；1 的对端是 0。若未来扩到 N 节点，可改成挑选不等于自己 IP 的第一个。
  if [ "$NODE_RANK" -eq 0 ]; then
    PEER_IP="${HOST_IPS[1]}"
  else
    PEER_IP="${HOST_IPS[0]}"
  fi

  # 探测与对端通信的网卡名；若失败或为 lo，则使用默认路由；如仍失败，要求用户 FIXED_NIC
  NIC=""
  if [ -n "'"$FIXED_NIC"'" ]; then
    NIC="'"$FIXED_NIC"'"
  else
    NIC=$(ip route get "$PEER_IP" 2>/dev/null | awk '"'"'{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'"'"' || true)
    if [[ -z "${NIC:-}" || "$NIC" == "lo" ]]; then
      NIC=$(ip -o route 2>/dev/null | awk '"'"'/default/ {print $5; exit}'"'"' || true)
    fi
  fi
  if [[ -z "${NIC:-}" || "$NIC" == "lo" ]]; then
    echo "无法自动探测到有效网卡名。请在提交时指定：FIXED_NIC=<你的网卡名> sbatch ..."
    exit 1
  fi
  echo "节点 ${NODE_RANK} 使用网卡: ${NIC}（对端: ${PEER_IP}）"

  # 动态检测是否有 IB 设备：有则挂载
  EXTRA_DEVICES=""
  if [ -d /dev/infiniband ]; then
    EXTRA_DEVICES+=" --device=/dev/infiniband"
  fi

  '"$CONTAINER_CMD"' rm -f "$DOCKER_NAME" >/dev/null 2>&1 || true
  # 启动容器（root）
  '"$CONTAINER_CMD"' run --rm \
    --name $DOCKER_NAME \
    --privileged \
    --security-opt seccomp=unconfined \
    --group-add keep-groups \
    --device=/dev/kfd \
    --device=/dev/dri \
    ${EXTRA_DEVICES} \
    --network=host \
    --shm-size=128g \
    --ulimit memlock=-1:-1 \
    -v $WORKDIR:$WORKDIR \
    -v "$HOME":/workdir \
    -v "$HOME/.ssh":/root/.ssh \
    --workdir /workdir \
    -e MASTER_ADDR="'"$MASTER_ADDR"'" \
    -e MASTER_PORT="'"$MASTER_PORT"'" \
    -e NODE_RANK \
    -e NNODES \
    -e NCCL_DEBUG="'"$NCCL_DEBUG"'" \
    -e NCCL_SOCKET_IFNAME="$NIC" \
    -e GLOO_SOCKET_IFNAME="$NIC" \
    -e NCCL_IB_HCA="'"$NCCL_IB_HCA"'" \
    -e NCCL_IB_QPS_PER_CONNECTION="'"$NCCL_IB_QPS_PER_CONNECTION"'" \
    -e NCCL_IB_TIMEOUT="'"$NCCL_IB_TIMEOUT"'" \
    -e NCCL_IB_DISABLE="'"$NCCL_IB_DISABLE"'" \
    -e NCCL_IB_RETRY_CNT="'"$NCCL_IB_RETRY_CNT"'" \
    -e NCCL_NVLS_ENABLE="'"$NCCL_NVLS_ENABLE"'" \
    -e NCCL_IB_GID_INDEX="'"$NCCL_IB_GID_INDEX"'" \
    -e USE_ROCM_AITER_ROPE_BACKEND=0 \
    -e NUM_GPUS=8 \
    '"$DOCKER_IMAGE"' /bin/bash -lc "
      set -euo pipefail
      echo \"=== 容器内：节点 \$NODE_RANK 启动（宿主: ${NODE_NAME}, IP: ${NODE_IP}）===\"

      # 写 IP->主机名，避免容器内解析问题
      {
        echo \"${HOST_IPS[0]} ${HOSTS[0]}\"
        echo \"${HOST_IPS[1]} ${HOSTS[1]}\"
      } >> /etc/hosts

      cd '"$PROJECT_DIR_IN_CONTAINER"'

      # 依赖（建议烘焙进镜像；此处做一次性缓存）
      #if [ ! -f .dist_installed ]; then
      echo \"安装项目依赖...\"
      rm -rf build/ dist/ ./*.egg-info || true
      python3 -m pip install -U pip
      python3 -m pip install typeguard tyro
      python3 -m pip install -e . 
      #  touch .dist_installed
      #fi

      echo \"节点 \$NODE_RANK: 开始训练 - 主节点: $MASTER_ADDR:$MASTER_PORT\"
      bash '"$PROJECT_DIR_IN_CONTAINER/$TRAIN_SCRIPT_IN_CONTAINER"' \$NODE_RANK $MASTER_ADDR $MASTER_PORT

      echo \"=== 节点 \$NODE_RANK 完成 ===\"
    " > "$LOG_DIR/output.log" 2>&1
'

# ========== 7) 清理 ==========
# 如需保留中间文件便于排错，可注释下面两行
rm -rf "$RUN_DIR"

