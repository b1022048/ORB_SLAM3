#!/usr/bin/env python3
"""
軌跡診斷腳本
=============
將每幀位置誤差（ATE）與 tracking_log.csv 的追蹤指標對齊，
產生多層時序診斷圖。

使用方式：
  python3 analyze_trajectory.py --name stereo_imu_euroc_v102 \
      --gt /media/lab405/Windows1/data/Euroc/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv \
      --gt-format euroc

  python3 analyze_trajectory.py --name stereo_imu_tum_out6 \
      --gt /media/lab405/Windows1/data/TUM/outdoors/dataset-outdoors6_512_16/mav0/mocap0/data.csv \
      --gt-format tum

引數：
  --name        output_name（對應 fix_{name}.txt 和 {name}_tracking_log.csv）
  --gt          GT 檔案路徑
  --gt-format   euroc 或 tum（決定 GT 的時間戳和四元數格式）
  --traj        直接指定估計軌跡路徑（可選，預設用 fix_{name}.txt）
  --log         直接指定 tracking log 路徑（可選，預設用 {name}_tracking_log.csv）
  --output      輸出圖片檔名（預設 diagnosis_{name}.pdf）
  --no-align    關閉 SE3 對齊（debug 用）
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────
#  GT 載入
# ─────────────────────────────────────────────

def load_gt_euroc(gt_path: Path) -> pd.DataFrame:
    """EuRoC GT：timestamp_ns, px, py, pz, qw, qx, qy, qz"""
    rows = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            ts = float(parts[0]) / 1e9
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            rows.append({"timestamp": ts, "px": px, "py": py, "pz": pz})
    return pd.DataFrame(rows)


def load_gt_tum(gt_path: Path) -> pd.DataFrame:
    """TUM-VI GT：同 EuRoC 格式（mocap0/data.csv）"""
    return load_gt_euroc(gt_path)


# ─────────────────────────────────────────────
#  估計軌跡載入（TUM 格式）
# ─────────────────────────────────────────────

def load_traj(traj_path: Path) -> pd.DataFrame:
    """TUM 格式：timestamp tx ty tz qx qy qz qw"""
    rows = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts = float(parts[0])
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            rows.append({"timestamp": ts, "px": px, "py": py, "pz": pz})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  時間戳對齊（最近鄰）
# ─────────────────────────────────────────────

def match_timestamps(est: pd.DataFrame, gt: pd.DataFrame,
                     max_diff: float = 0.05) -> tuple:
    """對每個估計幀找最近的 GT 幀，回傳對齊後的兩個 numpy array (N,3)"""
    est_ts = est["timestamp"].values
    gt_ts  = gt["timestamp"].values
    gt_pos = gt[["px", "py", "pz"]].values

    est_matched, gt_matched, timestamps = [], [], []
    for i, ts in enumerate(est_ts):
        idx = np.argmin(np.abs(gt_ts - ts))
        if np.abs(gt_ts[idx] - ts) < max_diff:
            est_matched.append(est[["px", "py", "pz"]].values[i])
            gt_matched.append(gt_pos[idx])
            timestamps.append(ts)

    return (np.array(timestamps),
            np.array(est_matched),
            np.array(gt_matched))


# ─────────────────────────────────────────────
#  SE3 對齊（Umeyama，無尺度）
# ─────────────────────────────────────────────

def align_se3(est_pos: np.ndarray, gt_pos: np.ndarray):
    """回傳 R (3x3), t (3,)，使 R @ est + t ≈ gt"""
    mu_e = est_pos.mean(axis=0)
    mu_g = gt_pos.mean(axis=0)
    ec = est_pos - mu_e
    gc = gt_pos  - mu_g
    H  = ec.T @ gc
    U, _, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    D  = np.diag([1.0, 1.0, d])
    R  = Vt.T @ D @ U.T
    t  = mu_g - R @ mu_e
    return R, t


# ─────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="軌跡診斷腳本")
    parser.add_argument("--name",      required=True, help="output_name")
    parser.add_argument("--gt",        required=True, help="GT 檔案路徑")
    parser.add_argument("--gt-format", choices=["euroc", "tum"], default="euroc")
    parser.add_argument("--traj",      default=None, help="估計軌跡路徑（可選）")
    parser.add_argument("--log",       default=None, help="tracking log 路徑（可選）")
    parser.add_argument("--output",    default=None, help="輸出圖片檔名")
    parser.add_argument("--no-align",  action="store_true", help="關閉 SE3 對齊")
    args = parser.parse_args()

    name      = args.name
    traj_path = Path(args.traj) if args.traj else SCRIPT_DIR / f"fix_{name}.txt"
    log_path  = Path(args.log)  if args.log  else SCRIPT_DIR / f"{name}_tracking_log.csv"
    out_path  = Path(args.output) if args.output else SCRIPT_DIR / f"diagnosis_{name}.pdf"
    gt_path   = Path(args.gt)

    # ── 載入資料 ──
    for p, label in [(traj_path, "估計軌跡"), (gt_path, "GT"), (log_path, "Tracking log")]:
        if not p.exists():
            print(f"[ERROR] 找不到 {label}: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading trajectory: {traj_path}")
    est = load_traj(traj_path)

    print(f"Loading GT: {gt_path}")
    gt = load_gt_euroc(gt_path) if args.gt_format == "euroc" else load_gt_tum(gt_path)

    print(f"Loading tracking log: {log_path}")
    log_df = pd.read_csv(log_path)

    print(f"Est frames: {len(est)}, GT frames: {len(gt)}, Log frames: {len(log_df)}")

    # ── 時間戳對齊：估計軌跡 ↔ GT ──
    ts_arr, est_pos, gt_pos = match_timestamps(est, gt)
    if len(ts_arr) == 0:
        print("[ERROR] 沒有找到任何對齊的幀，請檢查時間戳格式", file=sys.stderr)
        sys.exit(1)
    print(f"Matched frames: {len(ts_arr)}")

    # ── SE3 對齊 ──
    if not args.no_align:
        R, t = align_se3(est_pos, gt_pos)
        est_pos_aligned = (R @ est_pos.T).T + t
    else:
        est_pos_aligned = est_pos

    # ── 每幀位置誤差 ──
    errors = np.linalg.norm(est_pos_aligned - gt_pos, axis=1)
    rel_ts = ts_arr - ts_arr[0]

    # ── 對齊 tracking log（用軌跡的 t0 當共同參考點）──
    log_ts  = log_df["timestamp"].values
    common_t0 = ts_arr[0]          # 軌跡第一幀的絕對時間戳
    log_rel = log_ts - common_t0   # log 也用同一個 t0，負值表示初始化前的幀

    # ── 軌跡斷點：tracking log 中狀態標記 ──
    state_col_name = "當前幀狀態" if "當前幀狀態" in log_df.columns else \
                     ("当前帧状态" if "当前帧状态" in log_df.columns else None)
    if state_col_name:
        lost_mask = log_df[state_col_name].isin(["LOST", "RECENTLY_LOST"]).values
    else:
        lost_mask = np.zeros(len(log_df), dtype=bool)
    reloc_mask = log_df["reloc_attempted"].astype(str).str.lower().isin(["yes", "true", "1"]).values

    # tracking_method 顏色對應
    method_map   = {"none": 0, "RefKF": 1, "MM": 2, "IMU_MM": 3, "Reloc": 4}
    method_vals  = log_df["tracking_method"].map(lambda x: method_map.get(str(x), -1)).values
    method_cmap  = ["gray", "steelblue", "forestgreen", "darkorange", "red"]

    # ── 繪圖 ──
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
    fig.suptitle(f"Trajectory Diagnosis: {name}", fontsize=13, fontweight="bold")

    # 子圖 1：每幀位置誤差
    ax1 = axes[0]
    ax1.plot(rel_ts, errors, color="royalblue", linewidth=0.8, label="Position Error (m)")
    ax1.set_ylabel("ATE (m)")
    ax1.set_title("Per-frame Position Error (after alignment)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    # 標記 RMSE
    rmse = float(np.sqrt(np.mean(errors**2)))
    ax1.axhline(rmse, color="red", linestyle="--", linewidth=0.8,
                label=f"RMSE={rmse:.4f}m")
    ax1.legend(loc="upper right", fontsize=8)

    # 子圖 2：final_inliers vs matches_before_tlm_opt
    ax2 = axes[1]
    ax2.plot(log_rel, log_df["matches_before_tlm_opt"].values,
             color="orange", linewidth=0.8, linestyle="--", label="before_opt")
    ax2.plot(log_rel, log_df["final_inliers"].values,
             color="darkgreen", linewidth=0.8, label="final_inliers")
    ax2.set_ylabel("Inliers")
    ax2.set_title("Inliers: before opt (orange--) vs after opt (green)  |  large gap = unstable pose opt")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # LOST 區塊標記（子圖 1 & 2）
    in_lost = False
    lost_start = None
    for i, (t_i, is_lost) in enumerate(zip(log_rel, lost_mask)):
        if is_lost and not in_lost:
            lost_start = t_i
            in_lost = True
        elif not is_lost and in_lost:
            for ax in [ax1, ax2]:
                ax.axvspan(lost_start, t_i, alpha=0.15, color="gray", label="_nolegend_")
            in_lost = False
    if in_lost:
        for ax in [ax1, ax2]:
            ax.axvspan(lost_start, log_rel[-1], alpha=0.15, color="gray")

    # Reloc 事件（紅色垂直線）
    for t_i in log_rel[reloc_mask]:
        ax2.axvline(t_i, color="red", linewidth=0.5, alpha=0.6)

    # 子圖 3：tracking_method — 用 step 折線顯示，非 MM 一目了然
    ax3 = axes[2]
    ax3.step(log_rel, method_vals, where="mid", color="royalblue", linewidth=0.8)
    ax3.set_yticks(list(method_map.values()))
    ax3.set_yticklabels(list(method_map.keys()), fontsize=8)
    ax3.set_ylim(-0.5, max(method_map.values()) + 0.5)

    # RECENTLY_LOST / LOST 狀態背景
    if state_col_name:
        rl_mask = log_df[state_col_name].astype(str) == "RECENTLY_LOST"
        lo_mask = log_df[state_col_name].astype(str) == "LOST"
        in_rl = False; in_lo = False; rl_s = lo_s = None
        for t_i, is_rl, is_lo in zip(log_rel, rl_mask.values, lo_mask.values):
            if is_rl and not in_rl:
                rl_s = t_i; in_rl = True
            elif not is_rl and in_rl:
                ax3.axvspan(rl_s, t_i, alpha=0.35, color="orange"); in_rl = False
            if is_lo and not in_lo:
                lo_s = t_i; in_lo = True
            elif not is_lo and in_lo:
                ax3.axvspan(lo_s, t_i, alpha=0.5, color="red"); in_lo = False
        if in_rl and rl_s is not None:
            ax3.axvspan(rl_s, log_rel[-1], alpha=0.35, color="orange")
        if in_lo and lo_s is not None:
            ax3.axvspan(lo_s, log_rel[-1], alpha=0.5, color="red")

    ax3.set_ylabel("Tracking Method")
    ax3.set_title("Tracking Method (step)  |  orange=RECENTLY_LOST  red=LOST")
    ax3.legend(handles=[
        mpatches.Patch(color="orange", alpha=0.5, label="RECENTLY_LOST"),
        mpatches.Patch(color="red",    alpha=0.5, label="LOST"),
    ], loc="upper right", fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 子圖 4：bias norm（VI 才有意義，純視覺全為 0）
    ax4 = axes[3]
    ax4.plot(log_rel, log_df["bias_acc_norm"].values,
             color="tomato", linewidth=0.8, label="bias_acc_norm")
    ax4.plot(log_rel, log_df["bias_gyro_norm"].values,
             color="steelblue", linewidth=0.8, label="bias_gyro_norm")
    ax4.set_ylabel("Bias norm")
    ax4.set_xlabel("Relative time (s)")
    ax4.set_title("IMU Bias Norm (always 0 in pure visual mode)")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 對齊 x 軸範圍 ──
    t_max_log = float(log_rel[-1])
    t_max_err = float(rel_ts[-1])
    for ax in axes:
        ax.set_xlim(0, max(t_max_log, t_max_err))

    # ── 圖例補充 ──
    lost_patch = mpatches.Patch(color="gray", alpha=0.3, label="LOST / RECENTLY_LOST")
    reloc_line = mpatches.Patch(color="red",  alpha=0.6, label="Reloc attempted")
    fig.legend(handles=[lost_patch, reloc_line],
               loc="lower center", ncol=2, fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nDiagnosis plot saved to: {out_path}")
    print(f"RMSE = {rmse:.6f} m  (matched frames: {len(ts_arr)})")


if __name__ == "__main__":
    main()
