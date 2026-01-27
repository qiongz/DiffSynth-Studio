#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

BOOLEAN_FLAG_ALIASES = {
    "use_gradient_checkpointing": "use_gradient_checkpointing",
    "use_gradient_checkpointing_offload": "use_gradient-checkpointing_offload",
    "deepspeed": "deepspeed",
}

RESERVED_KEYS = {"model_name"}

# default keys, can be overwritten by configs
DEFAULTS = {
    "dataset_base_path": "data/example_video_dataset",
    "dataset_metadata_path": "data/example_video_dataset/metadata.csv",
    "learning_rate": 1e-5,
    "remove_prefix_in_ckpt": "pipe.dit.",
    "trainable_models": "dit",
    "seed": 10007,
    "deepspeed": True,
}

# training entry
DEFAULT_ENTRY = ["deepspeed", "examples/wanvideo/model_training/train.py"]


def load_yaml(fpath: Path) -> Dict[str, Any]:
    with fpath.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cmd_from_config(entry: List[str],
                          base_required: Dict[str, Any],
                          cfg: Dict[str, Any]) -> List[str]:
    params = []
    merged = dict(DEFAULTS)
    merged.update(base_required)  # dataset paths 等
    merged.update(cfg or {})

    for key, val in merged.items():
        if key in RESERVED_KEYS:
            continue
        if key in BOOLEAN_FLAG_ALIASES:
            if isinstance(val, bool) and val:
                params.append(f"--{BOOLEAN_FLAG_ALIASES[key]}")
            continue

        if isinstance(val, bool):
            if val:
                params.append(f"--{key}")
            continue

        if val is None:
            continue

        params.append(f"--{key}")
        params.append(str(val))

    return list(entry) + params


def detect_machine(cli_machine: str = "") -> str:
    if cli_machine:
        return cli_machine

    from shutil import which
    if which("nvidia-smi"):
        return "NVIDIA"
    if which("rocm-smi") or which("rocminfo"):
        return "AMD"

    print("[ERROR] use --machine NVIDIA or --machine AMD ", file=sys.stderr)
    sys.exit(2)


def load_model_list(config_dir: Path, available_list: Path = None) -> List[str]:
    models = []
    if available_list and available_list.exists():
        for line in available_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    else:
        for f in sorted(config_dir.glob("*.yaml")):
            models.append(f.stem)
    return models


def stream_run(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_txt = log_path.parent / "cmd.txt"
    cmd_txt.write_text(" ".join(cmd), encoding="utf-8")

    print(f"[INFO] CMD: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
            lf.flush()
        proc.wait()
        return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="One-click runner for WanVideo models with YAML configs (tee logs)."
    )
    parser.add_argument("--configs-dir", type=str, default="configs",
                        help="Directory containing per-model YAML configs.")
    parser.add_argument("--available-list", type=str, default="available_models.txt",
                        help="Optional file to restrict which models to run (one per line, matching YAML filename).")
    parser.add_argument("--machine", type=str, default="",
                        choices=["", "H200", "MI325X"],
                        help="Override machine type for log path.")
    #parser.add_argument("--gpus", type=int, nargs="+", default=[1, 8],
    parser.add_argument("--gpus", type=int, nargs="+", default=[ 8],
                                help="List of GPU counts to run per model, e.g., --gpus 1 8")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only, do not execute.")
    parser.add_argument("--entry", type=str, nargs="+", default=None,
                        help="Override training entry command, default: deepspeed examples/wanvideo/model_training/train.py")
    parser.add_argument("--dataset-base-path", type=str, default=None)
    parser.add_argument("--dataset-metadata-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_dir = Path(args.configs_dir)
    if not config_dir.exists():
        print(f"[ERROR] configs dir not found: {config_dir}", file=sys.stderr)
        sys.exit(2)

    entry = args.entry if args.entry else DEFAULT_ENTRY

    machine = detect_machine(args.machine)
    print(f"[INFO] MACHINE={machine}")

    # models to run
    available_list = Path(args.available_list) if args.available_list else None
    models = load_model_list(config_dir, available_list)
    if not models:
        print("[WARN] 未发现要运行的模型。请检查 configs/ 或 available_models.txt")
        sys.exit(0)

    print(f"[INFO] 将运行的模型（{len(models)}）：{', '.join(models)}")

    # 全局覆盖项
    base_required = {}
    if args.dataset_base_path:
        base_required["dataset_base_path"] = args.dataset_base_path
    if args.dataset_metadata_path:
        base_required["dataset_metadata_path"] = args.dataset_metadata_path
    if args.seed is not None:
        base_required["seed"] = args.seed

    # 逐模型 + 逐 GPU 数运行
    for model in models:
        yaml_path = config_dir / f"{model}.yaml"
        if not yaml_path.exists():
            print(f"[WARN] 缺少配置文件 {yaml_path}，跳过。")
            continue

        cfg = load_yaml(yaml_path) or {}
        model_name = cfg.get("name", model)
        if "output_path" not in cfg:
            cfg["output_path"] = f"./models/train/{model_name}_full"

        # 基础训练参数（不含 --num_gpus）
        base_cmd = build_cmd_from_config(entry, base_required, cfg)

        for ngpu in args.gpus:
            # 插入 --num_gpus（只对 deepspeed 有效）
            if base_cmd[0] == "deepspeed":
                cmd = base_cmd[:1] + ["--num_gpus", str(ngpu)] + base_cmd[1:]
            else:
                # 万一 entry 改为 torchrun，这里可替换为 --nproc_per_node
                cmd = base_cmd

            out_dir = Path("logs") / f"{machine}-{ngpu}g" / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "train.log"

            print(f"[INFO] === Running {model_name} on {machine} with {ngpu} GPU(s) ===")
            if args.dry_run:
                print("[DRY] ", " ".join(cmd))
                (out_dir / "cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
                continue

            rc = stream_run(cmd, log_path)
            if rc != 0:
                print(f"[WARN] {model_name} ({ngpu}g) 退出码={rc}（日志：{log_path}）")
            else:
                print(f"[INFO] {model_name} ({ngpu}g) 完成（日志：{log_path}）")

        print("[INFO] All done. 日志位于 logs/<MACHINE>/<MODEL_NAME>/train.log")


if __name__ == "__main__":
    main()
