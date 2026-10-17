# -*- coding: utf-8 -*-
from __future__ import annotations
"""禾医 · 复诊时序对比（疗效验证，2026-10-17 新增，原创机制）
==================================================================
设计动机：现在的"复诊"只是一个反馈表单——农户拍第二张照片，系统并不会真的看。
          我们把复诊变成**可计算的疗效验证**：
          同一株/同一叶片的两次拍照 → 病害信号变化率 → 改善/恶化/稳定 → 处置建议。

机制（全确定性，不经大模型）：
  病害信号 = dark_pct + rust_pct + mildew_pct + yellow_pct（视觉特征提取器输出）
  变化率   = (after - before) / max(before, ε)
  · ≤ -20%  → improved（改善）
  · ≥ +20%  → worsened（恶化）
  · 其余    → stable（稳定）
叠加规则：若两次识别类别不同（A→B），无论信号如何都提示"重新确认/人工复核"。

为什么是真创新：农业 App 的"复诊"普遍是提醒用户回访；**把两次照片变成疗效判定**，
才让"进化闭环"真正落地——农户知道药有没有用，系统也拿到"处置→效果"的监督信号。
"""
DISEASE_KEYS = ("dark_pct", "rust_pct", "mildew_pct", "yellow_pct")
THRESH = 0.20   # ±20% 判定阈值


def _signal(features: dict) -> float:
    f = features or {}
    return sum(float(f.get(k, 0) or 0) for k in DISEASE_KEYS)


def compare(before: dict, after: dict) -> dict:
    """before/after 均为 model_infer.predict() 的返回。返回对比结论 + 建议。"""
    fb = (before or {}).get("features") or {}
    fa = (after or {}).get("features") or {}
    pb = ((before or {}).get("predictions") or [{}])[0]
    pa = ((after or {}).get("predictions") or [{}])[0]
    sb, sa = _signal(fb), _signal(fa)
    if sb <= 1e-6:
        rel = 0.0 if sa <= 1e-6 else 1.0
    else:
        rel = (sa - sb) / sb
    class_changed = (pb.get("class_name") and pa.get("class_name")
                     and pb["class_name"] != pa["class_name"])

    if rel <= -THRESH:
        verdict, label = "improved", "改善"
    elif rel >= THRESH:
        verdict, label = "worsened", "恶化"
    else:
        verdict, label = "stable", "稳定"

    if class_changed:
        advice = (f"⚠ 两次识别结果不同（{pb.get('class_name')} → {pa.get('class_name')}），"
                  "可能是拍摄部位/角度不同或病情变化。建议重新拍摄同一叶片的正反面，"
                  "或联系当地植保站复核后再决定用药。")
    elif verdict == "improved":
        advice = (f"病斑信号下降 {abs(rel)*100:.0f}%，处置有效。继续按当前方案，"
                  "保持通风降湿；2 周后再复诊一次确认巩固。")
    elif verdict == "worsened":
        advice = (f"病斑信号上升 {rel*100:.0f}%，处置未奏效或在扩散。"
                  "建议立即联系当地植保站，必要时隔离病株；若已连续用药 2 次无改善，考虑换药。")
    else:
        advice = ("病斑信号无明显变化（±20% 内）。若已用药 5~7 天仍无改善，"
                  "建议联系植保站评估是否换药或调整浓度。")

    return {
        "verdict": verdict, "verdict_cn": label,
        "signal_before": round(sb, 4), "signal_after": round(sa, 4),
        "change_pct": round(rel * 100, 1),
        "class_before": pb.get("class_name"), "class_after": pa.get("class_name"),
        "class_changed": bool(class_changed),
        "confidence_before": pb.get("confidence"), "confidence_after": pa.get("confidence"),
        "features_before": fb, "features_after": fa,
        "advice": advice,
        "note": "信号=暗斑+锈斑+霉层+黄化占比之和；判定阈值 ±20%，确定性计算、不经大模型",
    }
