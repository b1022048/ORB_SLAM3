#!/usr/bin/env python3
"""
ORB-SLAM3 Unified Benchmark Script
====================================
支援以下演算法：
  mono            – mono_euroc / mono_tum_vi
  mono_inertial   – mono_inertial_euroc / mono_inertial_tum_vi
  stereo          – stereo_euroc / stereo_tum_vi
  stereo_inertial – stereo_inertial_euroc / stereo_inertial_tum_vi

依序執行以下步驟：
  1. 執行對應二進位程式 (EuRoC: V1_01~V1_03 / TUM-VI: room1-6, corridor1-5, slides1-3)
  2. 以 fix_time.py 將時間戳 ns → s
  3. 以 evo_ape 計算 APE RMSE
  4. 結果存入 evaluation/benchmark_results_<timestamp>.csv

使用方式：
  cd /home/lab405/henry/for_git/ORB_SLAM3/evaluation

  # 執行所有演算法與資料集 (預設)
  python3 run_stereo_inertial_benchmark.py

  # 只跑特定演算法
  python3 run_stereo_inertial_benchmark.py --algo mono
  python3 run_stereo_inertial_benchmark.py --algo mono_inertial
  python3 run_stereo_inertial_benchmark.py --algo stereo
  python3 run_stereo_inertial_benchmark.py --algo stereo_inertial

  # 只跑特定資料集
  python3 run_stereo_inertial_benchmark.py --dataset euroc
  python3 run_stereo_inertial_benchmark.py --dataset tum

  # 組合篩選
  python3 run_stereo_inertial_benchmark.py --algo stereo --dataset euroc
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
#  基礎路徑
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent

VOCAB       = ROOT_DIR / "Vocabulary/ORBvoc.txt"
EXAMPLES    = ROOT_DIR / "Examples"
FIX_TIME_PY = SCRIPT_DIR / "fix_time.py"

TUM_DATA   = ROOT_DIR / "data/TUM"
EUROC_DATA = ROOT_DIR / "data/Euroc"

# ─────────────────────────────────────────────
#  演算法設定
#
#  tum_extra(mav0_path) → 插入在 times_file 之前的額外引數列表
#    mono              : []
#    mono_inertial     : [imu_file]
#    stereo            : [cam1]
#    stereo_inertial   : [cam1, imu_file]
# ─────────────────────────────────────────────

# 插在 times_file 之前的額外引數（僅 cam1）
def _tum_pre_mono(mav0: Path) -> list:
    return []

def _tum_pre_mono_inertial(mav0: Path) -> list:
    return []

def _tum_pre_stereo(mav0: Path) -> list:
    return [str(mav0 / "cam1/data")]

def _tum_pre_stereo_inertial(mav0: Path) -> list:
    return [str(mav0 / "cam1/data")]

# 插在 times_file 之後的額外引數（imu CSV）
def _tum_post_mono(mav0: Path) -> list:
    return []

def _tum_post_mono_inertial(mav0: Path) -> list:
    return [str(mav0 / "imu0/data.csv")]

def _tum_post_stereo(mav0: Path) -> list:
    return []

def _tum_post_stereo_inertial(mav0: Path) -> list:
    return [str(mav0 / "imu0/data.csv")]


ALGO_CONFIGS: dict = {
    "mono": {
        "label":        "Mono",
        "bin_dir":      EXAMPLES / "Monocular",
        "bin_euroc":    "mono_euroc",
        "bin_tum":      "mono_tum_vi",
        "yaml_euroc":   EXAMPLES / "Monocular/EuRoC.yaml",
        "yaml_tum":     EXAMPLES / "Monocular/TUM-VI.yaml",
        "euroc_ts_dir": EXAMPLES / "Monocular/EuRoC_TimeStamps",
        "tum_ts_dir":   EXAMPLES / "Monocular/TUM_TimeStamps",
        "tum_pre":      _tum_pre_mono,             # 插在 times 前
        "tum_post":     _tum_post_mono,            # 插在 times 後
        "out_prefix_e": "mono_euroc",
        "out_prefix_t": "mono_tum",
        "evo_align":    "-as",                     # monocular: align + scale
    },
    "mono_inertial": {
        "label":        "Mono-Inertial",
        "bin_dir":      EXAMPLES / "Monocular-Inertial",
        "bin_euroc":    "mono_inertial_euroc",
        "bin_tum":      "mono_inertial_tum_vi",
        "yaml_euroc":   EXAMPLES / "Monocular-Inertial/EuRoC.yaml",
        "yaml_tum":     EXAMPLES / "Monocular-Inertial/TUM-VI.yaml",
        "euroc_ts_dir": EXAMPLES / "Monocular-Inertial/EuRoC_TimeStamps",
        "tum_ts_dir":   EXAMPLES / "Monocular-Inertial/TUM_TimeStamps",
        "tum_pre":      _tum_pre_mono_inertial,    # 插在 times 前（無）
        "tum_post":     _tum_post_mono_inertial,   # 插在 times 後（imu）
        "out_prefix_e": "mimu_euroc",
        "out_prefix_t": "mimu_tum",
        "evo_align":    "-a",                      # inertial: scale known, align only
    },
    "stereo": {
        "label":        "Stereo",
        "bin_dir":      EXAMPLES / "Stereo",
        "bin_euroc":    "stereo_euroc",
        "bin_tum":      "stereo_tum_vi",
        "yaml_euroc":   EXAMPLES / "Stereo/EuRoC.yaml",
        "yaml_tum":     EXAMPLES / "Stereo/TUM-VI.yaml",
        "euroc_ts_dir": EXAMPLES / "Stereo/EuRoC_TimeStamps",
        "tum_ts_dir":   EXAMPLES / "Stereo/TUM_TimeStamps",
        "tum_pre":      _tum_pre_stereo,           # 插在 times 前（cam1）
        "tum_post":     _tum_post_stereo,          # 插在 times 後（無）
        "out_prefix_e": "se_euroc",
        "out_prefix_t": "se_tum",
        "evo_align":    "-a",                      # stereo: align only
    },
    "stereo_inertial": {
        "label":        "Stereo-Inertial",
        "bin_dir":      EXAMPLES / "Stereo-Inertial",
        "bin_euroc":    "stereo_inertial_euroc",
        "bin_tum":      "stereo_inertial_tum_vi",
        "yaml_euroc":   EXAMPLES / "Stereo-Inertial/EuRoC.yaml",
        "yaml_tum":     EXAMPLES / "Stereo-Inertial/TUM-VI.yaml",
        "euroc_ts_dir": EXAMPLES / "Stereo-Inertial/EuRoC_TimeStamps",
        "tum_ts_dir":   EXAMPLES / "Stereo-Inertial/TUM_TimeStamps",
        "tum_pre":      _tum_pre_stereo_inertial,  # 插在 times 前（cam1）
        "tum_post":     _tum_post_stereo_inertial, # 插在 times 後（imu）
        "out_prefix_e": "stereo_euroc",
        "out_prefix_t": "stereo_tum",
        "evo_align":    "-a",                      # stereo: align only
    },
}

# ─────────────────────────────────────────────
#  資料集清單
#  EuRoC : (seq_folder, ts_stem, output_suffix)
#  TUM-VI: (category,   seq_name, output_suffix)
# ─────────────────────────────────────────────
EUROC_DATASETS = [
    ("V1_01_easy",      "V101", "v101"),
    ("V1_02_medium",    "V102", "v102"),
    ("V1_03_difficult", "V103", "v103"),
]

TUM_DATASETS = [
    # room
    ("room",     "room1",     "room1"),
    ("room",     "room2",     "room2"),
    ("room",     "room3",     "room3"),
    ("room",     "room4",     "room4"),
    ("room",     "room5",     "room5"),
    ("room",     "room6",     "room6"),
    # corridor
    ("corridor", "corridor1", "corr1"),
    ("corridor", "corridor2", "corr2"),
    ("corridor", "corridor3", "corr3"),
    ("corridor", "corridor4", "corr4"),
    ("corridor", "corridor5", "corr5"),
    # slides
    ("slides",   "slides1",   "slides1"),
    ("slides",   "slides2",   "slides2"),
    ("slides",   "slides3",   "slides3"),
    # magistrale
    ("magistrale", "magistrale1", "mag1"),
    ("magistrale", "magistrale2", "mag2"),
    ("magistrale", "magistrale3", "mag3"),
    ("magistrale", "magistrale4", "mag4"),
    ("magistrale", "magistrale5", "mag5"),
]

# ─────────────────────────────────────────────
#  輔助函式
# ─────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: list, cwd=None, timeout: int = 7200):
    """執行外部指令，回傳 (stdout, stderr, returncode)"""
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


def parse_evo_rmse(stdout: str, stderr: str) -> float | None:
    """從 evo_ape 輸出解析 RMSE 數值"""
    combined = stdout + stderr
    match = re.search(r'rmse\s+([\d.eE+\-]+)', combined)
    if match:
        return float(match.group(1))
    return None


def resolve_euroc_seq_path(seq_folder: str) -> Path | None:
    """解析 EuRoC 序列實際路徑（處理巢狀資料夾情況）"""
    base = EUROC_DATA / seq_folder
    if (base / "mav0").exists():
        return base
    nested = base / seq_folder
    if (nested / "mav0").exists():
        return nested
    return None


def fix_time(raw_traj: Path, fixed_traj: Path) -> bool:
    """呼叫 fix_time.py 轉換時間戳 ns → s"""
    stdout, stderr, rc = run_cmd(
        [sys.executable, FIX_TIME_PY, raw_traj, fixed_traj]
    )
    if rc != 0:
        log(f"  [ERROR] fix_time.py 失敗:\n{stderr[:500]}")
        return False
    return True


def eval_with_evo(gt_csv: str, fixed_traj: Path, align_flag: str = "-a") -> float | None:
    """以 evo_ape euroc 計算 RMSE（align_flag: '-a' 或 '-as'）"""
    stdout, stderr, rc = run_cmd(
        ["evo_ape", "euroc", gt_csv, str(fixed_traj), align_flag],
        cwd=SCRIPT_DIR,
    )
    if rc != 0:
        log(f"  [ERROR] evo_ape 失敗:\n{stderr[:500]}")
        return None
    rmse = parse_evo_rmse(stdout, stderr)
    if rmse is None:
        log(f"  [WARN] 無法從 evo_ape 輸出解析 RMSE\n{stdout[:300]}")
    return rmse


def _postprocess(raw_traj: Path, output_name: str, gt_csv: str,
                 align_flag: str = "-a") -> tuple:
    """fix_time + evo_ape，回傳 (rmse_or_'FAILED', note)"""
    if not raw_traj.exists():
        log(f"  [ERROR] {raw_traj} 不存在")
        return "FAILED", f"找不到輸出軌跡: {raw_traj}"

    fixed_traj = SCRIPT_DIR / f"fix_{output_name}.txt"

    if not fix_time(raw_traj, fixed_traj):
        return "FAILED", "fix_time.py 失敗"

    rmse = eval_with_evo(gt_csv, fixed_traj, align_flag)
    if rmse is not None:
        return rmse, "OK"
    return "FAILED", "evo_ape 無法解析 RMSE"


# ─────────────────────────────────────────────
#  EuRoC 執行函式
#  所有 EuRoC 二進位呼叫格式相同：
#    bin  vocab  yaml  seq_path  times_file  output_name
#  （IMU 由二進位內部從 mav0/imu0 載入）
# ─────────────────────────────────────────────

def run_euroc(cfg: dict, seq_folder: str, ts_stem: str, seq_suffix: str) -> dict:
    output_name = f"{cfg['out_prefix_e']}_{seq_suffix}"
    entry = {
        "algo":        cfg["label"],
        "dataset":     "EuRoC",
        "sequence":    seq_folder,
        "output_name": output_name,
        "rmse":        "FAILED",
        "note":        "",
    }

    seq_path = resolve_euroc_seq_path(seq_folder)
    if seq_path is None:
        entry["note"] = "序列資料夾不存在"
        log(f"  [SKIP] {EUROC_DATA / seq_folder}/mav0 不存在")
        return entry

    times_file = cfg["euroc_ts_dir"] / f"{ts_stem}.txt"
    gt_csv     = str(seq_path / "mav0/state_groundtruth_estimate0/data.csv")
    binary     = cfg["bin_dir"] / cfg["bin_euroc"]

    for p, label in [(binary, "執行檔"), (times_file, "時間戳檔案"),
                     (Path(gt_csv), "Ground truth")]:
        if not Path(p).exists():
            entry["note"] = f"{label} 不存在: {p}"
            log(f"  [SKIP] {p}")
            return entry

    cmd = [binary, VOCAB, cfg["yaml_euroc"], seq_path, times_file, output_name]
    stdout, stderr, rc = run_cmd(cmd, cwd=ROOT_DIR)
    if rc != 0:
        entry["note"] = f"{cfg['bin_euroc']} 回傳 {rc}"
        log(f"  [ERROR] 執行失敗:\n{stderr[-800:]}")
        return entry

    rmse, note = _postprocess(ROOT_DIR / f"f_{output_name}.txt", output_name, gt_csv,
                               cfg["evo_align"])
    entry["rmse"] = rmse
    entry["note"] = note
    return entry


# ─────────────────────────────────────────────
#  TUM-VI 執行函式
#  呼叫格式：bin  vocab  yaml  cam0  [extra...]  times_file  output_name
#  extra 依演算法而異（無 / imu / cam1 / cam1+imu）
# ─────────────────────────────────────────────

def run_tum(cfg: dict, category: str, seq_name: str, seq_suffix: str) -> dict:
    output_name = f"{cfg['out_prefix_t']}_{seq_suffix}"
    entry = {
        "algo":        cfg["label"],
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
    times_file = cfg["tum_ts_dir"] / f"dataset-{seq_name}_512.txt"
    gt_csv     = str(mav0 / "mocap0/data.csv")
    binary     = cfg["bin_dir"] / cfg["bin_tum"]
    pre_args  = cfg["tum_pre"](mav0)   # times_file 之前（cam1）
    post_args = cfg["tum_post"](mav0)  # times_file 之後（imu）

    for p, label in [(binary, "執行檔"), (cam0, "cam0"),
                     (times_file, "timestamps"), (Path(gt_csv), "Ground truth")]:
        if not Path(p).exists():
            entry["note"] = f"{label} 不存在: {p}"
            log(f"  [SKIP] {p}")
            return entry

    for extra_p in pre_args + post_args:
        if not Path(extra_p).exists():
            entry["note"] = f"額外引數路徑不存在: {extra_p}"
            log(f"  [SKIP] {extra_p}")
            return entry

    # 正確順序：cam0  [cam1]  times_file  [imu]  output_name
    cmd = [binary, VOCAB, cfg["yaml_tum"], cam0] + pre_args + [times_file] + post_args + [output_name]
    stdout, stderr, rc = run_cmd(cmd, cwd=ROOT_DIR)
    if rc != 0:
        entry["note"] = f"{cfg['bin_tum']} 回傳 {rc}"
        log(f"  [ERROR] 執行失敗:\n{stderr[-800:]}")
        return entry

    rmse, note = _postprocess(ROOT_DIR / f"f_{output_name}.txt", output_name, gt_csv,
                               cfg["evo_align"])
    entry["rmse"] = rmse
    entry["note"] = note
    return entry


# ─────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Unified Benchmark")
    parser.add_argument(
        "--algo",
        choices=["mono", "mono_inertial", "stereo", "stereo_inertial", "all"],
        default="all",
        help="要測試的演算法 (預設: all)",
    )
    parser.add_argument(
        "--dataset",
        choices=["euroc", "tum", "all"],
        default="all",
        help="要測試的資料集 (預設: all)",
    )
    args = parser.parse_args()

    algo_keys = list(ALGO_CONFIGS.keys()) if args.algo == "all" else [args.algo]

    # 預先警告缺少執行檔
    for key in algo_keys:
        cfg = ALGO_CONFIGS[key]
        for bin_name in (cfg["bin_euroc"], cfg["bin_tum"]):
            bin_path = cfg["bin_dir"] / bin_name
            if not bin_path.exists():
                log(f"[WARN] 找不到執行檔: {bin_path}，請先編譯專案")

    results = []

    for algo_key in algo_keys:
        cfg = ALGO_CONFIGS[algo_key]

        # ── EuRoC ──
        if args.dataset in ("euroc", "all"):
            log("\n" + "=" * 65)
            log(f"{cfg['label']} × EuRoC 測試")
            log("=" * 65)
            for seq_folder, ts_stem, seq_suffix in EUROC_DATASETS:
                log(f"\n[{cfg['label']}][EuRoC] {seq_folder}")
                entry = run_euroc(cfg, seq_folder, ts_stem, seq_suffix)
                results.append(entry)
                log(f"  → RMSE = {entry['rmse']}  ({entry['note']})")

        # ── TUM-VI ──
        if args.dataset in ("tum", "all"):
            log("\n" + "=" * 65)
            log(f"{cfg['label']} × TUM-VI 測試")
            log("=" * 65)
            for category, seq_name, seq_suffix in TUM_DATASETS:
                log(f"\n[{cfg['label']}][TUM-VI] {seq_name}")
                entry = run_tum(cfg, category, seq_name, seq_suffix)
                results.append(entry)
                log(f"  → RMSE = {entry['rmse']}  ({entry['note']})")

    # ── 儲存 CSV ──
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = SCRIPT_DIR / f"benchmark_results_{ts_str}.csv"

    fieldnames = ["algo", "dataset", "sequence", "output_name", "rmse", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── 摘要輸出 ──
    log("\n" + "=" * 75)
    log(f"結果已儲存至: {csv_path}")
    log("=" * 75)
    log(f"{'Algo':<18} {'Dataset':<8} {'Sequence':<22} {'RMSE':<14} {'Note'}")
    log("-" * 75)
    for r in results:
        rmse_str = f"{r['rmse']:.6f}" if isinstance(r['rmse'], float) else str(r['rmse'])
        log(f"{r['algo']:<18} {r['dataset']:<8} {r['sequence']:<22} {rmse_str:<14} {r['note']}")


if __name__ == "__main__":
    main()
