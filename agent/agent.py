# -*- coding: utf-8 -*-
"""
禾医·农事智能体（Agent）
========================
设计对齐第十一届创新赛"智能体开发"考察点——清晰的 工具(Tool) + 编排(Orchestration) 结构:

  工具:
    T1 query_kb(类别名)          —— 知识库精确查询(症状/病原/条件/防治/风险)
    T2 kb_search(自由文本)       —— 知识库语义检索(轻量RAG)
    T3 treatment_plan(诊断结果)  —— 生成结构化处置方案(用药/农事/监测三段)
    T4 escalation_check(诊断结果)—— 高危/暴发性病害升级提醒(如晚疫病、黄龙病、病毒病)

  编排:
    explain_diagnosis()  : 图像诊断结果 -> 知识库事实 -> (可选)LLM润色成农技员口吻报告
    chat()               : 用户自然语言提问 -> kb_search(RAG) -> LLM生成(未配置LLM时模板拼接)

  LLM 仅做"语言组织"，不做事实生成 —— 事实全部来自知识库，规避大模型幻觉，
  这一点写入答辩要点（可信AI设计）。
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.knowledge_base import KnowledgeBase
from agent.llm_client import LLMClient

kb = KnowledgeBase()
llm = LLMClient()

# ------------------------------------------------------------- T1/T2
def query_kb(class_name: str) -> dict:
    return kb.get(class_name)

def kb_search(query: str, topk: int = 3):
    return kb.search(query, topk=topk)

# ------------------------------------------------------------- T3
def treatment_plan(pred_class: str, confidence: float) -> dict:
    info = kb.get(pred_class)
    urgent = info.get("severity") in ("极高（暴发性）", "高") or "病毒" in str(info.get("pathogen", ""))
    plan = {
        "diagnosis": {"class": pred_class, "name_cn": info.get("name_cn", pred_class),
                      "confidence": confidence, "pathogen": info.get("pathogen", "?")},
        "immediate": info.get("treatment", "按常规管理"),
        "agronomy": info.get("prevention", "-"),
        "monitor": "3天后复诊拍照对比；全田抽查10%植株确认扩散范围",
        "escalation": None,
    }
    if urgent:
        plan["escalation"] = ("⚠ 该病害为高危/暴发型或无特效药（病毒类）——" + info.get("risk_if_untreated", "") +
                              " 建议立即联系当地植保站并隔离病株/病区。")
    return plan

# ------------------------------------------------------------- T4 已并入T3 escalation

def _template_report(plan: dict, features: dict | None) -> str:
    d = plan["diagnosis"]
    lines = [
        f"【诊断】{d['name_cn']}（{d['class']}），模型置信度 {d['confidence']*100:.1f}%，病原：{d['pathogen']}。",
    ]
    if features:
        lines.append(f"【影像依据】叶面占比 {features.get('leaf_pct', 0)}%，"
                     f"黄化 {features.get('yellow_pct', 0)}%，锈色孢子堆 {features.get('rust_pct', 0)}%，"
                     f"坏死黑斑 {features.get('dark_pct', 0)}%（簇状密度 {features.get('cluster', 0)}）。")
    kb_info = query_kb(d["class"])
    lines.append(f"【典型症状】{kb_info.get('symptoms', '-')}")
    lines.append(f"【发病条件】{kb_info.get('conditions', '-')}")
    lines.append(f"【立即处置】{plan['immediate']}")
    lines.append(f"【农事预防】{plan['agronomy']}")
    lines.append(f"【复诊监测】{plan['monitor']}")
    if plan["escalation"]:
        lines.append(plan["escalation"])
    return "\n".join(lines)

# ------------------------------------------------------------- 编排入口
def explain_diagnosis(pred_class: str, confidence: float, features: dict | None = None) -> dict:
    """图像诊断 -> 结构化方案 + 自然语言报告(LLM可用则润色, 否则模板)。"""
    plan = treatment_plan(pred_class, confidence)
    report = _template_report(plan, features)
    if llm.enabled:
        prompt = (
            "你是植保农技员。严格基于以下事实(不得编造新事实)改写成一段面向农户的通俗诊断报告，"
            "保留所有用药名称与数字，200字以内:\n" + report)
        refined = llm.chat([{"role": "user", "content": prompt}])
        if refined:
            report = refined + "\n\n——以上由禾医智能体基于知识库事实生成"
    return {"plan": plan, "report": report, "llm_used": llm.enabled}

def chat(question: str) -> dict:
    """RAG问答: 知识库检索 -> 上下文注入 -> LLM/模板回答。"""
    hits = kb_search(question, topk=2)
    ctx = "\n".join(
        f"[{i+1}] {info.get('name_cn','')}({cid}): 症状-{info.get('symptoms','-')}；"
        f"防治-{info.get('treatment', info.get('action','-'))}"
        for i, (cid, s, info) in enumerate(hits))
    if llm.enabled:
        messages = [
            {"role": "system", "content":
             "你是植保助手'禾医'。只依据给出的知识库片段回答，片段不足时明确说明需咨询当地植保站。"},
            {"role": "user", "content": f"知识库片段:\n{ctx}\n\n问题: {question}"},
        ]
        ans = llm.chat(messages)
        if ans:
            return {"answer": ans, "refs": [cid for cid, _, _ in hits], "llm_used": True}
    if hits:
        best = hits[0][2]
        return {"answer": (f"根据知识库，您的问题最接近『{best.get('name_cn','未知')}』:\n"
                           f"· 症状: {best.get('symptoms','-')}\n"
                           f"· 防治: {best.get('treatment', best.get('action','-'))}"),
                "refs": [cid for cid, _, _ in hits], "llm_used": False}
    return {"answer": "知识库暂无匹配内容，建议联系当地植保站。", "refs": [], "llm_used": False}

if __name__ == "__main__":
    print(explain_diagnosis("Tomato___Late_blight", 0.93,
                            {"green": 0.41, "yellow": 0.18, "rust": 0.03,
                             "dark": 0.21, "light_fungus": 0.01, "dark_cluster": 0.31})["report"])
    print("\n---\n")
    print(chat("玉米叶子上锈色的粉是什么病？"))
