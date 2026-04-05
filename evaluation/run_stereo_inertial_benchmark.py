#!/usr/bin/env python3
"""
ORB-SLAM3 Stereo-Inertial Benchmark Script
===========================================
依序執行以下步驟：
  1. 執行 stereo_inertial_euroc  (EuRoC 資料集: V1_01, V1_02, V1_03)
  2. 執行 stereo_inertial_tum_vi (TUM-VI 資料集: room1-6, corridor1-5, slides1-3)
  3. 對每個輸出軌跡用 fix_time.py 做時間戳轉換 (ns → s)
  4. 用 evo_ape 計算 APE (RMSE)
  5. 將所有結果存進 evaluation/stereo_inertial_results_<timestamp>.csv

使用方式：
  cd /home/lab405/henry/for_git/ORB_SLAM3/evaluation
  python3 run_stereo_inertial_benchmark.py

  # 或只跑特定資料集：
  python3 run_stereo_inertial_benchmark.py --dataset euroc
  python3 run_stereo_inertial_benchmark.py --dataset tum
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
#  路徑設定
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent

VOCAB       = ROOT_DIR / "Vocabulary/ORBvoc.txt"
SI_DIR      = ROOT_DIR / "Examples/Stereo-Inertial"

EUROC_BIN   = SI_DIR / "stereo_inertial_euroc"
TUM_BIN     = SI_DIR / "stereo_inertial_tum_vi"
EUROC_YAML  = SI_DIR / "EuRoC.yaml"
TUM_YAML    = SI_DIR / "TUM-VI.yaml"

TUM_TS_DIR   = SI_DIR / "TUM_TimeStamps"
EUROC_TS_DIR = SI_DIR / "EuRoC_TimeStamps"

TUM_DATA   = ROOT_DIR / "data/TUM"
EUROC_DATA = ROOT_DIR / "data/Euroc"

FIX_TIME_PY = SCRIPT_DIR / "fix_time.py"

# ─────────────────────────────────────────────
#  資料集清單
# ─────────────────────────────────────────────

# (資料集大類目錄, 序列名稱, 輸出識別名)
TUM_DATASETS = [
    # room
    ("room",     "room1",     "stereo_tum_room1"),
    ("room",     "room2",     "stereo_tum_room2"),
    ("room",     "room3",     "stereo_tum_room3"),
    ("room",     "room4",     "stereo_tum_room4"),
    ("room",     "room5",     "stereo_tum_room5"),
    ("room",     "room6",     "stereo_tum_room6"),
    # corridor
    ("corridor", "corridor1", "stereo_tum_corr1"),
    ("corridor", "corridor2", "stereo_tum_corr2"),
    ("corridor", "corridor3", "stereo_tum_corr3"),
    ("corridor", "corridor4", "stereo_tum_corr4"),
    ("corridor", "corridor5", "stereo_tum_corr5"),
    # slides
    ("slides",   "slides1",   "stereo_tum_slides1"),
    ("slides",   "slides2",   "stereo_tum_slides2"),
    ("slides",   "slides3",   "stereo_tum_slides3"),
]

# (序列資料夾名, EuRoC 時間戳檔案前綴, 輸出識別名)
EUROC_DATASETS = [
    ("V1_01_easy",      "V101", "stereo_euroc_v101"),
    ("V1_02_medium",    "V102", "stereo_euroc_v102"),
    ("V1_03_difficult", "V103", "stereo_euroc_v103"),
]


# ─────────────────────────────────────────────
#  輔助函式
# ─────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)


def run_cmd(cmd: list, cwd=None, timeout=7200):
    """執行指令，回傳 (stdout, stderr, returncode)"""
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"  >> {cmd_str}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def parse_evo_rmse(stdout: str, stderr: str):
    """從 evo_ape 輸出中解析 RMSE"""
    combined = stdout + stderr
    match = re.search(r'rmse\s+([\d.eE+\-]+)', combined)
    if match:
        return float(match.group(1))
    return None


def resolve_euroc_seq_path(seq_folder: str) -> Path | None:
    """
    解析 EuRoC 序列實際路徑（處理 V1_03_difficult 有時有巢狀資料夾的情況）
    """
    base = EUROC_DATA / seq_folder
    if (base / "mav0").exists():
        return base
    # 嘗試巢狀結構
    nested = base / seq_folder
    if (nested / "mav0").exists():
        return nested
    return None


def fix_time(raw_traj: Path, fixed_traj: Path) -> bool:
    """呼叫 fix_time.py 轉換時間戳"""
    stdout, stderr, rc = run_cmd(
        [sys.executable, FIX_TIME_PY, raw_traj, fixed_traj]
    )
    if rc != 0:
        log(f"  [ERROR] fix_time.py 失敗:\n{stderr[:500]}")
        return False
    return True


def eval_with_evo(gt_csv: str, fixed_traj: Path) -> float | None:
    """用 evo_ape euroc 評估，回傳 RMSE"""
    stdout, stderr, rc = run_cmd(
        ["evo_ape", "euroc", gt_csv, str(fixed_traj), "-a"],
        cwd=SCRIPT_DIR,
    )
    if rc != 0:
        log(f"  [ERROR] evo_ape 失敗:\n{stderr[:500]}")
        return None
    rmse = parse_evo_rmse(stdout, stderr)
    if rmse is None:
        log(f"  [WARN] 無法從 evo_ape 輸出解析 RMSE\n{stdout[:300]}")
    return rmse


# ─────────────────────────────────────────────
#  EuRoC 執行函式
# ─────────────────────────────────────────────

def run_euroc(seq_folder: str, ts_name: str, output_name: str) -> dict:
    """對單一 EuRoC 序列執行完整流程，回傳結果 dict"""
    entry = {
        "dataset":     "EuRoC",
        "sequence":    seq_folder,
        "output_name": output_name,
        "rmse":        "FAILED",
        "note":        "",
    }

    # 找序列路徑
    seq_path = resolve_euroc_seq_path(seq_folder)
    if seq_path is None:
        entry["note"] = "序列資料夾不存在"
        log(f"  [SKIP] {EUROC_DATA / seq_folder}/mav0 不存在")
        return entry

    times_file = EUROC_TS_DIR / f"{ts_name}.txt"
    gt_csv     = str(seq_path / "mav0/state_groundtruth_estimate0/data.csv")

    if not times_file.exists():
        entry["note"] = f"時間戳檔案不存在: {times_file}"
        log(f"  [SKIP] {times_file}")
        return entry

    if not os.path.exists(gt_csv):
        entry["note"] = f"Ground truth 不存在: {gt_csv}"
        log(f"  [SKIP] {gt_csv}")
        return entry

    # 1. 執行 stereo_inertial_euroc
    cmd = [EUROC_BIN, VOCAB, EUROC_YAML, seq_path, times_file, output_name]
    stdout, stderr, rc = run_cmd(cmd, cwd=ROOT_DIR)
    if rc != 0:
        entry["note"] = f"stereo_inertial_euroc 回傳 {rc}"
        log(f"  [ERROR] 執行失敗:\n{stderr[-800:]}")
        return entry

    # 2. fix_time.py
    raw_traj   = ROOT_DIR / f"f_{output_name}.txt"
    fixed_traj = SCRIPT_DIR / f"fix_{output_name}.txt"

    if not raw_traj.exists():
        entry["note"] = f"找不到輸出軌跡: {raw_traj}"
        log(f"  [ERROR] {raw_traj} 不存在")
        return entry

    if not fix_time(raw_traj, fixed_traj):
        entry["note"] = "fix_time.py 失敗"
        return entry

    # 3. evo_ape
    rmse = eval_with_evo(gt_csv, fixed_traj)
    if rmse is not None:
        entry["rmse"] = rmse
        entry["note"] = "OK"
    else:
        entry["note"] = "evo_ape 無法解析 RMSE"

    return entry


# ─────────────────────────────────────────────
#  TUM-VI 執行函式
# ─────────────────────────────────────────────

def run_tum(category: str, seq_name: str, output_name: str) -> dict:
    """對單一 TUM-VI 序列執行完整流程，回傳結果 dict"""
    entry = {
        "dataset":     "TUM-VI",
        "sequence":    seq_name,
        "output_name": output_name,
        "rmse":        "FAILED",
        "note":        "",
    }

    dataset_path = TUM_DATA / category / f"dataset-{seq_name}_512_16"
    mav0 = dataset_path / "mav0"

    if not mav0.exists():
        entry["note"] = f"資料集目錄不存在: {mav0}"
        log(f"  [SKIP] {mav0}")
        return entry

    cam0       = mav0 / "cam0/data"
    cam1       = mav0 / "cam1/data"
    times_file = TUM_TS_DIR / f"dataset-{seq_name}_512.txt"
    imu_csv    = mav0 / "imu0/data.csv"
    gt_csv     = str(mav0 / "mocap0/data.csv")

    for p, label in [(cam0, "cam0"), (cam1, "cam1"),
                     (times_file, "timestamps"), (imu_csv, "IMU")]:
        if not Path(p).exists():
            entry["note"] = f"{label} 不存在: {p}"
            log(f"  [SKIP] {p}")
            return entry

    if not os.path.exists(gt_csv):
        entry["note"] = f"Ground truth 不存在: {gt_csv}"
        log(f"  [SKIP] {gt_csv}")
        return entry

    # 1. 執行 stereo_inertial_tum_vi
    cmd = [TUM_BIN, VOCAB, TUM_YAML, cam0, cam1, times_file, imu_csv, output_name]
    stdout, stderr, rc = run_cmd(cmd, cwd=ROOT_DIR)
    if rc != 0:
        entry["note"] = f"stereo_inertial_tum_vi 回傳 {rc}"
        log(f"  [ERROR] 執行失敗:\n{stderr[-800:]}")
        return entry

    # 2. fix_time.py
    raw_traj   = ROOT_DIR / f"f_{output_name}.txt"
    fixed_traj = SCRIPT_DIR / f"fix_{output_name}.txt"

    if not raw_traj.exists():
        entry["note"] = f"找不到輸出軌跡: {raw_traj}"
        log(f"  [ERROR] {raw_traj} 不存在")
        return entry

    if not fix_time(raw_traj, fixed_traj):
        entry["note"] = "fix_time.py 失敗"
        return entry

    # 3. evo_ape
    rmse = eval_with_evo(gt_csv, fixed_traj)
    if rmse is not None:
        entry["rmse"] = rmse
        entry["note"] = "OK"
    else:
        entry["note"] = "evo_ape 無法解析 RMSE"

    return entry


# ─────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Stereo-Inertial Benchmark")
    parser.add_argument(
        "--dataset",
        choices=["euroc", "tum", "all"],
        default="all",
        help="要測試的資料集 (預設: all)",
    )
    args = parser.parse_args()

    # 確認執行檔存在
    for bin_path, name in [(EUROC_BIN, "stereo_inertial_euroc"),
                            (TUM_BIN,   "stereo_inertial_tum_vi")]:
        if not bin_path.exists():
            log(f"[WARN] 找不到執行檔: {bin_path}，請先編譯專案")

    results = []

    # ── EuRoC ──
    if args.dataset in ("euroc", "all"):
        log("\n" + "=" * 60)
        log("EuRoC Stereo-Inertial 測試")
        log("=" * 60)
        for seq_folder, ts_name, output_name in EUROC_DATASETS:
            log(f"\n[EuRoC] {seq_folder}")
            entry = run_euroc(seq_folder, ts_name, output_name)
            results.append(entry)
            log(f"  → RMSE = {entry['rmse']}  ({entry['note']})")

    # ── TUM-VI ──
    if args.dataset in ("tum", "all"):
        log("\n" + "=" * 60)
        log("TUM-VI Stereo-Inertial 測試")
        log("=" * 60)
        for category, seq_name, output_name in TUM_DATASETS:
            log(f"\n[TUM-VI] {seq_name}")
            entry = run_tum(category, seq_name, output_name)
            results.append(entry)
            log(f"  → RMSE = {entry['rmse']}  ({entry['note']})")

    # ── 儲存 CSV ──
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = SCRIPT_DIR / f"stereo_inertial_results_{ts_str}.csv"

    fieldnames = ["dataset", "sequence", "output_name", "rmse", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── 摘要輸出 ──
    log("\n" + "=" * 60)
    log(f"結果已儲存至: {csv_path}")
    log("=" * 60)
    log(f"{'Dataset':<10} {'Sequence':<22} {'RMSE':<14} {'Note'}")
    log("-" * 65)
    for r in results:
        rmse_str = f"{r['rmse']:.6f}" if isinstance(r['rmse'], float) else str(r['rmse'])
        log(f"{r['dataset']:<10} {r['sequence']:<22} {rmse_str:<14} {r['note']}")


if __name__ == "__main__":
    main()
