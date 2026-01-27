#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_e2e_and_make_table.py

扫描 logs 目录，自动识别以下结构：
- logs/MI325X/<MODEL>/train.log
- logs/H200/<MODEL>/train.log
- logs/MI325X-1g/<MODEL>/train.log
- logs/MI325X-8g/<MODEL>/train.log
- logs/H200-1g/<MODEL>/train.log
- logs/H200-8g/<MODEL>/train.log
（即：<machine> 或 <machine>-<Ng>g 两种形式）

提取 e2e_s（秒/迭代），默认取“最后 N 个样本的中位数”（N 可调），生成两类输出：
1) e2e_summary.*        —— 主对比表（按 --prefer-gpus 策略为每个机器选一个卡数）
2) e2e_by_machine_gpu.* —— 平台 × GPU 数的明细（长表 + Markdown 概览）

相对性能定义（与既有口径一致）：
  MI325X vs. H200 (%) = (H200_e2e / MI325X_e2e) * 100
e2e 为耗时（越小越好），因此 >100% 表示 MI325X 更快。

用法示例：
  python3 parse_e2e_and_make_table.py \
    --log-root logs \
    --last-n 5 \
    --prefer-gpus 8 1 \
    --raw-csv e2e_raw_points.csv \
    --summary-csv e2e_summary.csv \
    --summary-md e2e_summary.md \
    --by-mg-csv e2e_by_machine_gpu.csv \
    --by-mg-md e2e_by_machine_gpu.md
"""

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# 宽松 e2e 值匹配（支持 12、12.3、1e-3、1.23E+02 等）
PATTERN_E2E = re.compile(r"e2e_s\s*=\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)")
PATTERN_OOM = re.compile(r"(out of memory|CUDA out of memory|HIP out of memory|OOM)", re.IGNORECASE)

MACHINE_CANONICAL = {"MI325X", "H200"}

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(errors="ignore")
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            return ""

def find_log_file(model_dir: Path) -> Optional[Path]:
    p = model_dir / "train.log"
    if p.exists():
        return p
    candidates = list(model_dir.glob("*.log")) + list(model_dir.glob("*.txt"))
    return candidates[0] if candidates else None

def extract_e2e_points(text: str) -> List[float]:
    vals: List[float] = []
    for m in PATTERN_E2E.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue
    return vals

def detect_machine_gpu(dirname: str) -> Tuple[Optional[str], Optional[int]]:
    """
    输入 logs 子目录名，返回 (machine, gpus)
    支持：
      MI325X        -> ("MI325X", None)
      H200          -> ("H200", None)
      MI325X-8g     -> ("MI325X", 8)
      H200-1g       -> ("H200", 1)
    其他命名返回 (None, None)
    """
    name = dirname.strip()
    # 原生
    if name in MACHINE_CANONICAL:
        return name, None
    # 后缀式
    for m in MACHINE_CANONICAL:
        prefix = m + "-"
        if name.startswith(prefix) and name.endswith("g"):
            mid = name[len(prefix):-1]
            if mid.isdigit():
                return m, int(mid)
    return None, None

def is_oom(text: str) -> bool:
    return bool(PATTERN_OOM.search(text))

def median_last_n(values: List[float], last_n: int) -> Tuple[Optional[float], int]:
    if not values:
        return None, 0
    used = values[-last_n:] if len(values) >= last_n else values
    try:
        med = statistics.median(used)
    except Exception:
        return None, 0
    return med, len(used)

def fnum(x: Optional[float], digits: int = 3, nan: str = "NaN") -> str:
    return f"{x:.{digits}f}" if x is not None else nan

def compute_relative(mi_e2e: Optional[float], h2_e2e: Optional[float]) -> Optional[float]:
    if mi_e2e is None or h2_e2e is None:
        return None
    if mi_e2e <= 0:
        return None
    return (h2_e2e / mi_e2e) * 100.0

def collect_all_runs(
    log_root: Path,
    last_n: int
) -> List[Dict[str, Any]]:
    """
    返回每条日志的明细：
    {
      "machine": "MI325X" | "H200",
      "gpus": 1 | 8 | None,
      "model_name": "xxx",
      "log_path": ".../train.log",
      "oom": bool,
      "e2e_points": [float, ...],
      "median_e2e": float|None,
      "used_points": int
    }
    """
    rows: List[Dict[str, Any]] = []
    if not log_root.exists():
        print(f"[WARN] log root not found: {log_root}")
        return rows

    for sub in sorted([d for d in log_root.iterdir() if d.is_dir()]):
        machine, gpus = detect_machine_gpu(sub.name)
        if machine is None:
            # 不识别的目录名跳过
            continue

        for model_dir in sorted([d for d in sub.iterdir() if d.is_dir()]):
            log_file = find_log_file(model_dir)
            if not log_file:
                print(f"[WARN] no log file in {model_dir}")
                continue

            text = read_text(log_file)
            points = extract_e2e_points(text)
            oom = is_oom(text)
            median, used_n = median_last_n(points, last_n)

            # OOM 视为无效 e2e
            if oom:
                median = None

            rows.append({
                "machine": machine,
                "gpus": gpus,
                "model_name": model_dir.name,
                "log_path": str(log_file),
                "oom": oom,
                "e2e_points": points,
                "median_e2e": median,
                "used_points": used_n,
            })
    return rows

def write_raw_points_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "gpus", "model_name", "log_path", "oom", "num_points", "used_points", "median_e2e", "points"])
        for r in rows:
            w.writerow([
                r["machine"],
                r["gpus"] if r["gpus"] is not None else "",
                r["model_name"],
                r["log_path"],
                "Yes" if r["oom"] else "No",
                len(r["e2e_points"]),
                r["used_points"],
                fnum(r["median_e2e"], digits=6),
                ";".join(map(str, r["e2e_points"])),
            ])

def aggregate_by_machine_gpu(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    返回层次结构：
    agg[model][machine][gpus] = {"median_e2e":..., "oom":..., "log_path":...}
    其中 gpus 可能为 None（表示未标注卡数的目录，如 logs/H200）
    """
    agg: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for r in rows:
        model = r["model_name"]
        machine = r["machine"]
        g = r["gpus"] if r["gpus"] is not None else -1  # 用 -1 表示“未标注卡数”的目录
        agg.setdefault(model, {}).setdefault(machine, {})[g] = {
            "median_e2e": r["median_e2e"],
            "oom": r["oom"],
            "log_path": r["log_path"],
        }
    return agg

def select_one_gpu_run(
    mg_dict: Dict[int, Dict[str, Any]],
    prefer_gpus: List[int]
) -> Tuple[Optional[float], Optional[bool], str]:
    """
    在某个 (model, machine) 的多 GPU 结果中，根据 prefer_gpus 选择一条作为主表用：
    - 先按 prefer_gpus 顺序挑可用的（非 NaN & 非 OOM）
    - 若都不可用，则选“最大 gpus”（或 -1）但 median 可能为 None
    返回 (median_e2e, oom_flag, note)
    note 会标注实际选择的 gpus 值
    """
    # 先收集可用集合
    available_keys = [k for k, v in mg_dict.items() if v.get("median_e2e") is not None and not v.get("oom")]
    # 1) 按偏好顺序挑
    for g in prefer_gpus:
        key = g
        if key in mg_dict and (key in available_keys):
            sel = mg_dict[key]
            return sel["median_e2e"], sel["oom"], f"gpus={g}"
    # 2) 退化：选最大 g（包括 -1），即便是 None
    if mg_dict:
        key = sorted(mg_dict.keys())[-1]
        sel = mg_dict[key]
        gnote = "unlabeled" if key == -1 else str(key)
        return sel.get("median_e2e"), sel.get("oom"), f"gpus={gnote}"
    return None, None, ""  # 不存在

def write_summary_main(
    agg: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    summary_csv: Path,
    summary_md: Path,
    prefer_gpus: List[int]
) -> None:
    """
    生成主对比表（每个平台每模型只选一条）
    """
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Model",
            "MI325X e2e_s (s/iter)",
            "H200 e2e_s (s/iter)",
            "MI325X vs. H200 (%)",
            "MI325X Note",
            "H200 Note",
        ])
        rows_for_md: List[List[str]] = []

        for model in sorted(agg.keys()):
            mi_dict = agg[model].get("MI325X", {})
            h2_dict = agg[model].get("H200", {})

            mi_e2e, mi_oom, mi_note = (None, None, "MI325X missing")
            h2_e2e, h2_oom, h2_note = (None, None, "H200 missing")

            if mi_dict:
                mi_e2e, mi_oom, mi_note = select_one_gpu_run(mi_dict, prefer_gpus)
            if h2_dict:
                h2_e2e, h2_oom, h2_note = select_one_gpu_run(h2_dict, prefer_gpus)

            rel = compute_relative(mi_e2e, h2_e2e)

            w.writerow([
                model,
                fnum(mi_e2e),
                fnum(h2_e2e),
                f"{rel:.0f}%" if rel is not None else "NaN",
                mi_note,
                h2_note,
            ])

            # Markdown 行
            notes: List[str] = []
            if "missing" in mi_note:
                notes.append("MI325X missing")
            if "missing" in h2_note:
                notes.append("H200 missing")

            linestr = f"| {model} | " \
                      f"{(f'{rel:.0f}%' if rel is not None else 'NaN')} | " \
                      f"{fnum(mi_e2e)} | {fnum(h2_e2e)} | " \
                      f"{', '.join(notes) if notes else ''} |"
            rows_for_md.append(linestr)

    # 写 Markdown
    md_lines: List[str] = []
    md_lines.append("| Model | MI325X vs. H200 (Perf) | MI325X e2e_s (s/iter) | H200 e2e_s (s/iter) | Notes |")
    md_lines.append("|---|---:|---:|---:|---|")
    md_lines.extend(rows_for_md)
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

def write_by_machine_gpu(
    agg: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    by_mg_csv: Path,
    by_mg_md: Path,
) -> None:
    """
    生成平台×GPU 数的长表 CSV 与概览 Markdown
    """
    # 收集所有出现过的 (machine, gpus) 组合（用于 MD 概览列头）
    combos: Set[Tuple[str, int]] = set()
    for model, mdict in agg.items():
        for mach, gdict in mdict.items():
            for g in gdict.keys():
                combos.add((mach, g))
    # 排序：按 machine 再按 g（-1 表示 unlabeled，置后）
    sorted_combos = sorted(combos, key=lambda x: (x[0], x[1] if x[1] >= 0 else 10**9))

    # 写 CSV（长表）
    with by_mg_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Machine", "GPUs", "e2e_s (s/iter)", "OOM", "Log"])
        for model in sorted(agg.keys()):
            for mach in ["MI325X", "H200"]:
                if mach not in agg[model]:
                    continue
                for g in sorted(agg[model][mach].keys(), key=lambda x: (x if x >= 0 else 10**9)):
                    info = agg[model][mach][g]
                    e2e = info.get("median_e2e")
                    oom = info.get("oom")
                    logp = info.get("log_path", "")
                    gstr = "unlabeled" if g == -1 else str(g)
                    w.writerow([
                        model, mach, gstr, fnum(e2e), "Yes" if oom else "No", logp
                    ])

    # 写 Markdown 概览：按组合形成列
    # 构造一个行：每个 model 一行，每列是该 (machine,g) 的 e2e
    header_cols = [f"{mach}-{('unlabeled' if g==-1 else f'{g}g')}" for (mach, g) in sorted_combos]
    md_lines: List[str] = []
    md_lines.append("| Model | " + " | ".join(header_cols) + " |")
    md_lines.append("|---" + "|---" * len(header_cols) + "|")

    for model in sorted(agg.keys()):
        vals: List[str] = []
        for mach, g in sorted_combos:
            info = agg[model].get(mach, {}).get(g)
            if info is None:
                vals.append("NaN")
            else:
                vals.append(fnum(info.get("median_e2e")))
        md_lines.append("| " + model + " | " + " | ".join(vals) + " |")

    by_mg_md.write_text("\n".join(md_lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Parse e2e_s from logs and build MI325X vs H200 tables (with GPU-awareness).")
    ap.add_argument("--log-root", type=str, default="logs", help="Root dir containing logs/<machine>[-<Ng>g]/<model>/train.log")
    ap.add_argument("--last-n", type=int, default=5, help="Median of last N e2e points per log (if fewer, use all)")
    ap.add_argument("--prefer-gpus", type=int, nargs="+", default=[8, 1],
                    help="Order of GPU counts to prefer when selecting one run per machine for the main summary; e.g., --prefer-gpus 8 1")
    ap.add_argument("--raw-csv", type=str, default="e2e_perf_raw_points.csv", help="Raw points CSV")
    ap.add_argument("--summary-csv", type=str, default="e2e_perf_summary.csv", help="Main summary CSV")
    ap.add_argument("--summary-md", type=str, default="e2e_perf_summary.md", help="Main summary Markdown")
    ap.add_argument("--by-mg-csv", type=str, default="e2e_perf_by_machine_gpu.csv", help="By (machine,gpu) CSV")
    ap.add_argument("--by-mg-md", type=str, default="e2e_perf_by_machine_gpu.md", help="By (machine,gpu) Markdown")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    rows = collect_all_runs(log_root, args.last_n)
    if not rows:
        print("[WARN] No logs found under:", log_root)
        return

    # 1) 明细原始点
    write_raw_points_csv(rows, Path(args.raw_csv))

    # 2) 聚合为 (model -> machine -> gpus -> info)
    agg = aggregate_by_machine_gpu(rows)

    # 3) 主对比表（每平台选一条，偏好指定的 GPU 数）
    write_summary_main(agg, Path(args.summary_csv), Path(args.summary_md), args.prefer_gpus)

    # 4) 平台×GPU 总览
    write_by_machine_gpu(agg, Path(args.by_mg_csv), Path(args.by_mg_md))

    print("Done. Generated:")
    print(f"  - {args.raw_csv}")
    print(f"  - {args.summary_csv}")
    print(f"  - {args.summary_md}")
    print(f"  - {args.by_mg_csv}")
    print(f"  - {args.by_mg_md}")

if __name__ == "__main__":
    main()
