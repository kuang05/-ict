# -*- coding: utf-8 -*-
"""禾医 · VLM 第二意见仲裁（2026-10-17 新增，原创机制）
==================================================================
设计动机：CNN 低置信时，"硬猜"与"直接转人工"都不理想——
          前者可能把病叶判成健康（最危险错误），后者让守门拒绝率过高。
          我们引入视觉大模型（VLM）作为**复核员而非决策者**：
          · 只让它描述可见症状 + 从候选里选一个（或说无法判断）
          · **不给用药建议**——事实仍只来自知识库（防幻觉边界不破）
          · 三态仲裁：一致 / 不一致 / 无法判断
成本控制：只在 confidence < τ 时调用（默认 τ=0.6，与守门曲线同口径）。

三态语义：
  agree    → 双通道一致：可采信（报告附"VLM 复核支持，建议 3 天后复诊"）
  disagree → 双通道不一致：**转人工复核，不自动给药方**（宁可慢，不可错）
  uncertain→ VLM 亦无法判断：转人工

分歧样本写入 data/second_opinion_log.jsonl —— 进化层的训练素材候选
（与复诊反馈、误诊回流同一数据通道）。

用法（模块内）:
    from agent import second_opinion
    if second_opinion.should_review(conf, tau=0.6):
        so = second_opinion.arbitrate(image_b64=b64, cnn_top3=preds, llm=llm)
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "..", "data", "second_opinion_log.jsonl")
TAU_DEFAULT = 0.6

PROMPT_TMPL = """你是植保复核员。请只做两件事，不要给出任何用药建议：
1) 用一句话描述照片中可见的症状（颜色/形状/病斑分布；若无病斑请明说）；
2) 从下面候选里选出最接近的一个，无法判断就选 0。
候选：
{cands}
最后一行必须严格是：结论: <序号>
"""


def should_review(confidence: float, tau: float = TAU_DEFAULT) -> bool:
    """是否需要 VLM 复核（低置信才调用，成本控制）"""
    try:
        return float(confidence) < float(tau)
    except (TypeError, ValueError):
        return False


def _cands_text(names: list) -> str:
    return "\n".join(f"{i+1}) {n}" for i, n in enumerate(names))


def arbitrate(image_path=None, image_b64=None, cnn_top3=None, names_cn=None,
              llm=None, tau: float = TAU_DEFAULT, confidence: float | None = None,
              timeout: int = 60) -> dict:
    """三态仲裁。返回 dict（永远不抛异常，失败降级为 uncertain）。"""
    preds = cnn_top3 or []
    names = names_cn or [p.get("name_cn") or p.get("class_name", "?") for p in preds[:3]]
    out = {"called": False, "verdict": "uncertain", "agree": None,
           "vlm_choice": None, "vlm_text": "", "reason": ""}
    if not llm or not getattr(llm, "enabled", False):
        out["reason"] = "llm_disabled"
        return out
    if not preds or len(names) < 2:
        out["reason"] = "no_candidates"
        return out
    out["called"] = True
    try:
        text = llm.chat_vision(PROMPT_TMPL.format(cands=_cands_text(names)),
                               image_path=image_path, image_b64=image_b64,
                               temperature=0.1, timeout=timeout)
    except Exception as e:
        out["reason"] = f"vlm_error:{type(e).__name__}"
        return out
    if not text:
        out["reason"] = "vlm_empty"
        return out
    out["vlm_text"] = text.strip()[:600]
    m = re.search(r"结论\s*[:：]\s*(\d)", text)
    choice = int(m.group(1)) if m else None
    if choice is None or choice == 0 or choice > len(names):
        out.update(verdict="uncertain", reason="vlm_undecided")
    elif choice == 1:
        out.update(verdict="agree", agree=True, vlm_choice=names[0])
    else:
        out.update(verdict="disagree", agree=False, vlm_choice=names[choice - 1])
    # 分歧/不确定样本留痕（进化层素材）
    try:
        os.makedirs(os.path.dirname(os.path.abspath(LOG_PATH)), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "confidence": confidence, "tau": tau,
                                "cnn_top3": names, "verdict": out["verdict"],
                                "vlm_choice": out["vlm_choice"],
                                "vlm_text": out["vlm_text"][:300]},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass
    return out


def arbitrate_note(so: dict) -> str:
    """把仲裁结果翻译成给农户的一句话（附在报告前/后）"""
    if not so or not so.get("called"):
        return ""
    v = so.get("verdict")
    if v == "agree":
        return "✅ 视觉复核：VLM 复核支持上述诊断（双通道一致），建议 3 天后复诊对比。"
    if v == "disagree":
        return ("⚠ 视觉复核：VLM 判断更接近「" + str(so.get("vlm_choice")) +
                "」，与识别结果不一致——**建议先联系当地植保站人工复核，暂不按上方处方施药**。")
    return "⚠ 视觉复核：VLM 亦无法确定（可能光照/角度/早期病斑），建议人工复核或 3 天后复拍。"
