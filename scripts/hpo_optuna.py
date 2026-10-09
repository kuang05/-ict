# -*- coding: utf-8 -*-
"""禾医 · 自动超参数优化（Optuna TPE 贝叶斯搜索 + 中位数剪枝）

机器自动学习最优训练参数：
  搜索空间由 12 轮实验证据界定；每个 trial 训 12 epochs 并按轮上报 PDval；
  落后于同轮中位数的 trial 自动终止（节约算力）；TPE 从已完成试验学习更优方向。
最优参数自动触发 40 轮完整训练。

用法: nohup python scripts/hpo_optuna.py > logs/hpo.log 2>&1 &
恢复: 同命令重跑（SQLite 断点续跑）

== 2026-10-08 修复 ==
bug1（致命）: 原版调用 train_field_v3.py 时未传 --pv-data/--pd-root/--extra-data,
  训练进程落到默认路径 /root/data/... 立刻 PermissionError 退出; 日志里没有任何
  PDval, 于是每个 trial 都返回 0.0 —— 12 个 trial 全部无效, 且 0.0 被 TPE 当成
  有效结果, 最终拿"0 分最优参数"启动了正式训练。现补全三个路径参数。
bug2（隐蔽）: 进程异常退出时不应静默返回 0, 否则坏配置会被误判为"可行但差"。
  现改为抛 RuntimeError, 让该 trial 状态置为 FAIL 且不参与 best 选择。
bug3: 去掉 --extend-classes。v13 实测 30 类扩展 + pd_repeat=1 反而使 test top1
  从 67.9%(v11) 掉到 64.4%, 属负向因子, 不应进入搜索空间。
"""
import json
import os
import re
import subprocess
import sys
import time

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

# 数据路径必须显式传入: 默认值 /root/data/... 在本机不可读
PV_DATA = os.environ.get(
    "HEYI_PV", "/home/ma-user/data/pv/Plant_leave_diseases_dataset_without_augmentation")
PD_ROOT = os.environ.get("HEYI_PD", "/home/ma-user/data/pd_v2")
EXTRA_DATA = os.environ.get("HEYI_EXTRA", "/home/ma-user/data/pv_extra")

STORAGE = "sqlite:///results/hpo_study.db"
N_TRIALS = int(os.environ.get("HPO_TRIALS", "12"))
TRIAL_EPOCHS = int(os.environ.get("HPO_TRIAL_EPOCHS", "12"))
FINAL_EPOCHS = int(os.environ.get("HPO_FINAL_EPOCHS", "40"))

DATA_ARGS = ["--pv-data", PV_DATA, "--pd-root", PD_ROOT, "--extra-data", EXTRA_DATA]


def _pdvals(log):
    txt = open(log, encoding="utf-8", errors="ignore").read()
    return [float(m) for m in re.findall(r"PDval=([0-9.]+)", txt)], txt


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1.2e-3, log=True)
    mix_p = trial.suggest_float("mix_p", 0.0, 0.7)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.4)
    mixup_p = trial.suggest_float("mixup_p", 0.0, 0.5)
    pd_repeat = trial.suggest_categorical("pd_repeat", [1, 2])
    use_ema = trial.suggest_categorical("ema", [0, 1])
    pv_per_class = trial.suggest_categorical("pv_per_class", [400, 600, 800])

    tag = f"hpo{trial.number}"
    log = f"logs/hpo_{trial.number}.log"
    out = f"model/checkpoints/hpo_{trial.number}.pth"
    cmd = [
        sys.executable, "model/train_field_v3.py",
        "--backbone", "resnet50", "--img-size", "224",
        "--pd-repeat", str(pd_repeat), "--lr", f"{lr:.6f}", "--mix-p", f"{mix_p:.3f}",
        "--mixup", f"{mixup_alpha:.3f}", "--mixup-p", f"{mixup_p:.3f}",
        "--ema-decay", "0.999" if use_ema else "0",
        "--pv-per-class", str(pv_per_class),
        "--epochs", str(TRIAL_EPOCHS), "--batch", "64", "--workers", "48",
        "--out", out, "--tag", tag,
    ] + DATA_ARGS
    env = {**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        best_seen = 0.0
        while proc.poll() is None:
            time.sleep(20)
            try:
                pdvals, _ = _pdvals(log)
                if pdvals:
                    ep = len(pdvals)
                    best_seen = max(best_seen, max(pdvals))
                    trial.report(best_seen, ep)
                    if trial.should_prune():
                        proc.kill()
                        raise optuna.TrialPruned(f"ep{ep} pdval={pdvals[-1]:.4f}")
            except optuna.TrialPruned:
                raise
            except Exception:
                pass
    pdvals, txt = _pdvals(log)
    if not pdvals:
        tail = "\n".join([l for l in txt.strip().splitlines()
                          if "Warning" not in l and "is not 1" not in l][-8:])
        raise RuntimeError(
            f"trial {trial.number} 训练进程未产出任何 PDval（疑似崩溃/参数错误）。"
            f"日志尾部:\n{tail}")
    best = max(pdvals)
    print(f"[trial {trial.number}] best PDval={best:.4f} params={trial.params}", flush=True)
    return best


def final_run(best_params):
    """最优参数自动触发 40 轮完整训练"""
    tag = "v17_hpo_best"
    cmd = [
        sys.executable, "model/train_field_v3.py",
        "--backbone", "resnet50", "--img-size", "224",
        "--pd-repeat", str(best_params["pd_repeat"]),
        "--lr", f"{best_params['lr']:.6f}", "--mix-p", f"{best_params['mix_p']:.3f}",
        "--mixup", f"{best_params['mixup_alpha']:.3f}", "--mixup-p", f"{best_params['mixup_p']:.3f}",
        "--ema-decay", "0.999" if best_params["ema"] else "0",
        "--pv-per-class", str(best_params["pv_per_class"]),
        "--epochs", str(FINAL_EPOCHS), "--batch", "64", "--workers", "48",
        "--out", "model/checkpoints/best_hpo_final.pth", "--tag", tag,
    ] + DATA_ARGS
    env = {**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
    with open("logs/hpo_final.log", "w") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    print("最优参数 40 轮完整训练已自动启动: logs/hpo_final.log", flush=True)


def main():
    study = optuna.create_study(
        study_name="heyi_hpo", storage=STORAGE, load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=4),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    try:
        best = study.best_trial
    except ValueError:
        print("\n===== HPO 失败: 没有任何 trial 产出有效 PDval =====", flush=True)
        print("请检查 logs/hpo_*.log 是否全部为训练崩溃。", flush=True)
        return
    print("\n===== HPO 完成 =====", flush=True)
    print("最优 PDval:", f"{best.value:.4f}")
    print("最优参数:", json.dumps(best.params, ensure_ascii=False), flush=True)
    with open("results/hpo_best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_pdval": best.value, "params": best.params,
                   "n_trials": len(study.trials)}, f, ensure_ascii=False, indent=2)
    final_run(best.params)


if __name__ == "__main__":
    main()
