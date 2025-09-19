#!/bin/bash
#SBATCH --job-name=mi325x-dist-train-example
#SBATCH --output=logs/slurm/mi325x-dist.%j.out
#SBATCH --nodelist=gpu-18,gpu-9
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
export PROJECT_DIR_ON_HOST="$HOME/codes/DiffSynth-Studio"
export PROJECT_DIR_IN_CONTAINER="/workdir/codes/DiffSynth-Studio"
export TRAIN_SCRIPT_IN_CONTAINER="examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P_dist.sh"
export DOCKER_IMAGE="docker.io/rocm/ali-private:rocm6.4.3_ubuntu22.04_py3.10_pytorch_2.8.0_ck_fa3_fremont_dist_250917"
export CONTAINER_CMD="podman"

# docker 下传递 data, models 等目录软连接
export WORKDIR="$HOME"

# 共享运行目录（所有节点可见）
export RUN_DIR="$PROJECT_DIR_ON_HOST/.slurm_runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# 时间戳（节点日志分目录）
export TIME_STAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "$TIME_STAMP"

# ========== 1) 获取实际分配的节点名 ==========
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#HOSTS[@]}" -lt 1 ]; then
  echo "未分配到节点: ${HOSTS[*]}"
  exit 1
fi
echo "分配到的节点: ${HOSTS[*]}"

# ========== 2) 解析主机名 -> IP（用 IP 更稳） ==========
HOST_IPS=()
resolve_ip() {
  local h="$1" ip=""
  # 2) aac 系统上 gpu-9, gpu-18i 上 法解析 gpu-22，但 gpu-22 可以解析其他机器，需要找管理员排查下
  ip=$(timeout 10s getent -s files hosts "$h" | awk '{print $1; exit}')
  [[ -n "$ip" ]] && { echo "$ip"; return 0; }

  return 1
}
for h in "${HOSTS[@]}"; do
  ip="$(resolve_ip "$h")" || {
    echo "解析失败：$h DNS 无法解析" 
    exit 1
  }
  echo "$h, $ip"
  HOST_IPS+=("$ip")
done

# ========== 3) 设置 Master（用第一个 IP）& 生成 DeepSpeed hostfile ==========
export MASTER_ADDR="${HOST_IPS[0]}"
export MASTER_PORT="${MASTER_PORT:-29475}"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"

export HOSTFILE_PATH="$PROJECT_DIR_ON_HOST/examples/wanvideo/model_training/full/hostfile"
mkdir -p "$(dirname "$HOSTFILE_PATH")"
: > "$HOSTFILE_PATH"
for ip in "${HOST_IPS[@]}"; do
  echo "$ip slots=8" >> "$HOSTFILE_PATH"
done
echo "Hostfile 内容："
cat "$HOSTFILE_PATH"

# ========== 4) 写入共享文件，供各节点读取 ==========
export HOSTS_FILE="$RUN_DIR/hosts.txt"
export IPS_FILE="$RUN_DIR/ips.txt"
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

# export FIXED_NIC="${FIXED_NIC:-enp49s0f1np1}" # lmx5
export FIXED_NIC="${FIXED_NIC:-enp49s0f0np0}"  # ROCE (默认)

# ========== 6) 启动（每节点一个容器；容器内 deepspeed --no_ssh 起本地 8 进程） ==========
srun --kill-on-bad-exit=1  bash -c '
  set -euo pipefail

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

  # 选一个对端 IP（多机时挑一个不等于自己 IP 的）
  PEER_IP=""
  if [ "$NNODES" -gt 1 ]; then
    for ip in "${HOST_IPS[@]}"; do
      if [ "$ip" != "$NODE_IP" ]; then
        PEER_IP="$ip"
        break
      fi
    done
  fi

  # 探测 NIC
  NIC=""
  if [ -n "${FIXED_NIC:-}" ]; then
    NIC="$FIXED_NIC"
  else
    if [ -n "$PEER_IP" ]; then
      NIC=$(ip route get "$PEER_IP" 2>/dev/null | awk '"'"'{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'"'"' || true)
    fi
    if [[ -z "${NIC:-}" || "${NIC:-lo}" == "lo" ]]; then
      NIC=$(ip -o route 2>/dev/null | awk '"'"'/default/ {print $5; exit}'"'"' || true)
    fi
  fi
  if [[ -z "${NIC:-}" || "${NIC:-lo}" == "lo" ]]; then
    echo "无法自动探测到有效网卡名。请在提交时指定：FIXED_NIC=<你的网卡名> sbatch ..."
    exit 1
  fi
  echo "节点 ${NODE_RANK} 使用网卡: ${NIC}（对端: ${PEER_IP:-<none>}）"

  # 仅当 NIC 能直达 MASTER_ADDR 时才绑定 GLOO
  SET_GLOO=0
  if ip route get "$MASTER_ADDR" 2>/dev/null | awk -v nic="$NIC" '"'"'{for(i=1;i<=NF;i++) if($i=="dev" && $(i+1)==nic){found=1}} END{exit found?0:1}'"'"' ; then
    SET_GLOO=1
  fi

  EXTRA_DEVICES=""
  [ -d /dev/infiniband ] && EXTRA_DEVICES+=" --device=/dev/infiniband"

  $CONTAINER_CMD rm -f "$DOCKER_NAME" >/dev/null 2>&1 || true

  $CONTAINER_CMD run --rm \
    --name "$DOCKER_NAME" \
    --privileged \
    --security-opt seccomp=unconfined \
    --group-add keep-groups \
    --device=/dev/kfd \
    --device=/dev/dri \
    ${EXTRA_DEVICES} \
    --network=host \
    --shm-size=128g \
    --ulimit memlock=-1:-1 \
    -v "$WORKDIR":"$WORKDIR" \
    -v "$HOME":/workdir \
    -v "$HOME/.ssh":/root/.ssh \
    --workdir /workdir \
    -e MASTER_ADDR="$MASTER_ADDR" \
    -e MASTER_PORT="$MASTER_PORT" \
    -e NODE_RANK \
    -e NNODES \
    -e NCCL_DEBUG="$NCCL_DEBUG" \
    -e NCCL_SOCKET_IFNAME="$NIC" \
    $( [ "$SET_GLOO" -eq 1 ] && echo -e "-e GLOO_SOCKET_IFNAME=$NIC" ) \
    -e NCCL_IB_HCA="$NCCL_IB_HCA" \
    -e NCCL_IB_QPS_PER_CONNECTION="$NCCL_IB_QPS_PER_CONNECTION" \
    -e NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT" \
    -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
    -e NCCL_IB_RETRY_CNT="$NCCL_IB_RETRY_CNT" \
    -e NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
    -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
    -e USE_ROCM_AITER_ROPE_BACKEND=0 \
    -e NUM_GPUS=8 \
    "$DOCKER_IMAGE" /bin/bash -lc "
      set -euo pipefail
      echo \"=== 容器内：节点 \$NODE_RANK 启动（宿主: ${NODE_NAME}, IP: ${NODE_IP}）===\"

      {
        i=0
        while read -r ip; do
          host=\$(sed -n \"\$((i+1))p\" \"$HOSTS_FILE\")
          echo \"\$ip \$host\"
          i=\$((i+1))
        done < \"$IPS_FILE\"
      } >> /etc/hosts

      cd \"$PROJECT_DIR_IN_CONTAINER\"

      echo \"安装项目依赖...\"
      rm -rf build/ dist/ ./*.egg-info || true
      python3 -m pip install -U pip
      python3 -m pip install typeguard tyro
      python3 -m pip install -e .

      echo \"节点 \$NODE_RANK: 开始训练 - 主节点: \$MASTER_ADDR:\$MASTER_PORT\"
      bash \"$PROJECT_DIR_IN_CONTAINER/$TRAIN_SCRIPT_IN_CONTAINER\" \$NODE_RANK \$MASTER_ADDR \$MASTER_PORT

      echo \"=== 节点 \$NODE_RANK 完成 ===\"
    " > "$LOG_DIR/output.log" 2>&1
'

# ========== 7) 清理 ==========
# 如需保留中间文件便于排错，可注释下面两行
#rm -rf "$RUN_DIR"

