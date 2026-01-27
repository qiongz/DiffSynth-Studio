#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_e2e_and_make_table.py

扫描 logs/MI325X 与 logs/H200 下的训练日志，提取每模型的 e2e_s（秒/迭代），
并生成对比表：
- 若某模型只在单侧机器跑过，则仅输出该侧 e2e；另一侧为 NaN
- 若两侧都跑过，则额外输出 "MI325X vs. H200 (%)"：
    Relative(%) = (H200_e2e / MI325X_e2e) * 100
  （e2e_s 是耗时，越小越好，该比值 >100% 表示 MI325X 更快，与既有表格口径一致）

输出文件：
- e2e_raw_points.csv
- e2e_summary.csv
- e2e_summary.md

用法：
    python3 parse_e2e_and_make_table.py
可选参数：
    --log-root logs                  # 日志根目录
    --machines MI325X H200           # 扫描的机器目录名（可改顺序/子集）
    --last-n 5                       # 取最后 N 个 e2e_s 计算中位数（不足 N 则用全部）
    --raw-csv e2e_raw_points.csv
    --summary-csv e2e_summary.csv
    --summary-md e2e_summary.md

日志匹配：
- 优先读取 <model_dir>/train.log
- 若不存在，回退到 *.log 或 *.txt 中的第一个
- e2e_s 提取使用正则： r"e2e_s\s*=\s*([0-9]+(?:\.[0-9]+)?)"
- 兼容 tqdm 的 ^M 控制符（不需要额外清洗）
"""

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 正则模式
PATTERN_E2E = re.compile(r"e2e_s\s*=\s*([0-9]+(?:\.[0-9]+)?)")
PATTERN_OOM = re.compile(r"(out of memory|CUDA out of memory|HIP out of memory|OOM)", re.IGNORECASE)


def read_file_text(path: Path) -> str:
    try:
        with path.open("r", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return ""


def find_log_file(model_dir: Path) -> Optional[Path]:
    # 优先 train.log
    path = model_dir / "train.log"
    if path.exists():
        return path
    # 回退 *.log / *.txt
    candidates = list(model_dir.glob("*.log")) + list(model_dir.glob("*.txt"))
    return candidates[0] if candidates else None


def extract_e2e_points(text: str) -> List[float]:
    return [float(m.group(1)) for m in PATTERN_E2E.finditer(text)]


def is_oom(text: str) -> bool:
    return bool(PATTERN_OOM.search(text))


def collect_logs(log_root: Path, machines: List[str], last_n: int) -> List[Dict]:
    """
    返回每个日志文件提取结果的列表：
    [
      {
        "machine": "MI325X" or "H200",
        "model_name": "<dir name under logs/<machine>/>",
        "log_path": "/abs/path/to/train.log",
        "e2e_points": [ ...float... ],
        "median_e2e": 12.345 or None,
        "used_points": 5,
        "oom": True/False
      },
      ...
    ]
    """
    rows = []
    for m in machines:
        base = log_root / m
        if not base.exists():
            print(f"[WARN] {base} not found, skip.")
            continue

        for model_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
            log_file = find_log_file(model_dir)
            if not log_file:
                print(f"[WARN] no log file in {model_dir}")
                continue

            text = read_file_text(log_file)
            points = extract_e2e_points(text)
            oom_flag = is_oom(text)

            if points:
                used = points[-last_n:] if len(points) >= last_n else points
                median_val = statistics.median(used)
                used_count = len(used)
            else:
                median_val = None
                used_count = 0

            rows.append({
                "machine": m,
                "model_name": model_dir.name,
                "log_path": str(log_file),
                "e2e_points": points,
                "median_e2e": median_val,
                "used_points": used_count,
                "oom": oom_flag,
            })
    return rows


def aggregate_by_model(rows: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    将 rows 按 model_name 聚合：
    {
      "Wan2.1-T2V-1.3B-480P-81F": {
        "MI325X": {"median_e2e": 31.35, "oom": False, "log_path": "..."},
        "H200":   {"median_e2e": 43.21, "oom": False, "log_path": "..."},
      },
      ...
    }
    """
    agg: Dict[str, Dict[str, Dict]] = {}
    for r in rows:
        key = r["model_name"]
        agg.setdefault(key, {})
        agg[key][r["machine"]] = {
            "median_e2e": r["median_e2e"],
            "oom": r["oom"],
            "log_path": r["log_path"],
        }
    return agg


def compute_relative(mi_e2e: Optional[float], h200_e2e: Optional[float]) -> Optional[float]:
    """
    相对性能（%）= H200_e2e / MI325X_e2e * 100
    e2e 是耗时（越小越好）；该值 >100% 表示 MI325X 更快（满足你的口径）
    """
    if mi_e2e is None or h200_e2e is None:
        return None
    if mi_e2e <= 0:
        return None
    return (h200_e2e / mi_e2e) * 100.0


def fnum(x: Optional[float], digits: int = 3, nan: str = "NaN") -> str:
    return f"{x:.{digits}f}" if (x is not None) else nan


def write_raw_points_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["machine", "model_name", "log_path", "oom", "num_points", "used_points", "median_e2e", "points"])
        for r in rows:
            writer.writerow([
                r["machine"],
                r["model_name"],
                r["log_path"],
                "Yes" if r["oom"] else "No",
                len(r["e2e_points"]),
                r["used_points"],
                fnum(r["median_e2e"], digits=6),
                ";".join(map(str, r["e2e_points"]))
            ])


def write_summary_csv(agg: Dict[str, Dict[str, Dict]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "MI325X e2e_s (s/iter)",
            "H200 e2e_s (s/iter)",
            "MI325X vs. H200 (%)",
            "MI325X OOM",
            "H200 OOM",
            "MI325X log",
            "H200 log",
        ])
        for model in sorted(agg.keys()):
            mi = agg[model].get("MI325X", {})
            h2 = agg[model].get("H200", {})
            mi_e2e = mi.get("median_e2e")
            h2_e2e = h2.get("median_e2e")
            rel = compute_relative(mi_e2e, h2_e2e)

            writer.writerow([
                model,
                fnum(mi_e2e),
                fnum(h2_e2e),
                f"{rel:.0f}%" if rel is not None else "NaN",
                "Yes" if (mi.get("oom") is True) else ("No" if mi else "NaN"),
                "Yes" if (h2.get("oom") is True) else ("No" if h2 else "NaN"),
                mi.get("log_path", "NaN") if mi else "NaN",
                h2.get("log_path", "NaN") if h2 else "NaN",
            ])


def write_summary_markdown(agg: Dict[str, Dict[str, Dict]], path: Path) -> None:
    lines: List[str] = []
    lines.append("| Model | MI325X vs. H200 | MI325X e2e_s (s/iter) | H200 e2e_s (s/iter) | Notes |")
    lines.append("|---|---:|---:|---:|---|")
    for model in sorted(agg.keys()):
        mi = agg[model].get("MI325X", {})
        h2 = agg[model].get("H200", {})
        mi_e2e = mi.get("median_e2e")
        h2_e2e = h2.get("median_e2e")
        rel = compute_relative(mi_e2e, h2_e2e)

        rel_str = f"{rel:.0f}%" if rel is not None else "NaN"
        mi_str = fnum(mi_e2e)
        h2_str = fnum(h2_e2e)

        notes: List[str] = []
        if not mi:
            notes.append("MI325X missing")
        elif mi.get("oom"):
            notes.append("MI325X OOM")
        if not h2:
            notes.append("H200 missing")
        elif h2.get("oom"):
            notes.append("H200 OOM")

        lines.append(f"| {model} | {rel_str} | {mi_str} | {h2_str} | {', '.join(notes)} |")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Parse e2e_s from logs and generate MI325X vs H200 comparison tables."
    )
    parser.add_argument("--log-root", type=str, default="logs", help="Root dir containing logs/<machine>/<model>/train.log")
    parser.add_argument("--machines", type=str, nargs="+", default=["MI325X", "H200"], help="Machines to scan, order matters.")
    parser.add_argument("--last-n", type=int, default=5, help="Use median of last N e2e_s values (if fewer, use all).")
    parser.add_argument("--raw-csv", type=str, default="e2e_raw_points.csv", help="Output CSV file for raw points.")
    parser.add_argument("--summary-csv", type=str, default="e2e_summary.csv", help="Output CSV file for summary.")
    parser.add_argument("--summary-md", type=str, default="e2e_summary.md", help="Output Markdown file for summary.")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    machines = args.machines
    last_n = args.last_n

    rows = collect_logs(log_root, machines, last_n)
    if not rows:
        print("[WARN] No logs found. Please ensure logs/MI325X/... and/or logs/H200/... exist.")
        return

    # 写原始点
    write_raw_points_csv(rows, Path(args.raw_csv))

    # 聚合 + 汇总
    agg = aggregate_by_model(rows)
    write_summary_csv(agg, Path(args.summary_csv))
    write_summary_markdown(agg, Path(args.summary_md))

    print("Done. Generated:")
    print(f"  - {args.raw_csv}")
    print(f"  - {args.summary_csv}")
    print(f"  - {args.summary_md}")


if __name__ == "__main__":
    main()
