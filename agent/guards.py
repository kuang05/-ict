# -*- coding: utf-8 -*-
"""禾医 · 防幻觉护栏（本地 Demo 版，与 new/app/agent_core.py 同源逻辑）
==================================================================
四层护栏：
  ① 知识库唯一事实源（agent.py 的模板报告本身）
  ② prompt 铁律（"即使农业上对也不许出现"）
  ③ 输出侧确定性校验 hallucination_check：数字/药名必须落地于给定事实 + 退化输出检测
  ④ 输入侧假药拒绝 unknown_drug_check：问题提及的药不在知识库词表 -> 拒绝给用法

2026-10-16 新增。与 new/ 仓库（赛题二交付）的护栏保持行为一致，
保证本地 Demo 与平台侧演示的防幻觉表现相同。
"""
import re

_NUM_RE = re.compile(r"\d+(?:[.~～\-–]\d+)?\s*(?:天|次|℃|%|克|毫升|ml|倍|小时|年|月|亩|ppm|波美度|倍液)")
_DRUG_SUFFIX = "唑菌酯胺嗪啉锰锌铜素灵威醚丹酚腈烷肟醇霜虱螨"
_MANUAL_DRUGS = {
    "氟吡菌胺", "噻虫嗪", "吡虫啉", "双丙环虫酯", "螺虫乙酯", "磷酸三钠", "防虫网",
    "次氯酸钠", "嘧菌酯", "苯醚甲环唑", "代森锰锌", "波尔多液", "石硫合剂",
    "农用链霉素", "噻唑锌", "克菌丹", "醚菌酯", "联苯肼酯", "螺螨酯", "乙螨唑",
    "唑醚·代森联", "氟菌·肟菌酯", "霜脲·锰锌", "精甲霜·锰锌", "氟吡菌胺·霜霉威",
    "烯酰吗啉", "腈菌唑", "枯草芽孢杆菌", "木霉菌", "苦参碱", "印楝素",
}
_DOSAGE_FORMS = ("乳油", "可湿性粉剂", "悬浮剂", "水剂", "粉剂", "颗粒剂", "水乳剂", "微乳剂")
_ASK_VERBS = ("治", "防治", "喷", "打")
_LEX = None


def _drug_lexicon(kb=None) -> set:
    """知识库药名词表（treatment/action 字段切词）+ 人工补充表"""
    global _LEX
    if _LEX is not None:
        return _LEX
    lex = set(_MANUAL_DRUGS)
    try:
        diseases = getattr(kb, "diseases", {}) if kb is not None else {}
        for cid, e in diseases.items():
            if cid.startswith("_") or not isinstance(e, dict):
                continue
            for f in ("treatment", "action", "prevention"):
                t = e.get(f) or ""
                for w in re.split(r"[、，,；;。（）()\s/]+", t):
                    w = w.strip()
                    if 2 <= len(w) <= 12 and any(c in _DRUG_SUFFIX for c in w):
                        lex.add(w)
    except Exception:
        pass
    _LEX = lex
    return lex


def _num_tokens(text: str) -> set:
    return set(m.group(0).replace(" ", "") for m in _NUM_RE.finditer(text or ""))


def hallucination_check(out: str, facts: str, kb=None) -> list:
    """输出护栏：返回违规项（数字未落地 / 药名不在事实中 / 退化重复）。空=通过。"""
    if not out:
        return []
    bad = []
    for t in sorted(_num_tokens(out) - _num_tokens(facts)):
        bad.append("数字:" + t)
    lex = _drug_lexicon(kb)
    fd = {d for d in lex if d in (facts or "")}
    for d in sorted(d for d in lex if d in out and d not in fd):
        bad.append("药名:" + d)
    grams = re.findall(r"[一-鿿]{4}", out)
    if grams:
        from collections import Counter
        top = Counter(grams).most_common(1)[0]
        if top[1] >= 3:
            bad.append(f"退化输出:{top[0]}x{top[1]}")
    return bad


def _clean_drug_cand(cand: str) -> str:
    for v in _ASK_VERBS:
        i = cand.find(v)
        if i > 0:
            cand = cand[:i]
    for pre in ("听说用", "听说", "邻居推荐", "网上说", "有人推荐", "用"):
        if cand.startswith(pre):
            cand = cand[len(pre):]
    for suf in _DOSAGE_FORMS:
        if cand.endswith(suf):
            cand = cand[: -len(suf)]
    for w in ("怎么", "如何", "什么", "多少", "吗", "呢", "能", "可以", "行不行", "特效"):
        while cand.endswith(w):
            cand = cand[: -len(w)]
    return cand.strip("的了和与及,，、！？")


def unknown_drug_check(question: str, kb=None) -> list:
    """输入护栏：返回问题中提及的、知识库药名词表里没有的候选药名。"""
    if not question:
        return []
    lex = _drug_lexicon(kb)
    hits = set()

    def _keep(cand: str):
        if len(cand) < 2 or cand.endswith(("病", "毒", "虫", "螨")):
            return
        if cand in lex or any(cand in d or d in cand for d in lex):
            return
        hits.add(cand)

    pat = (r"([一-鿿A-Za-z0-9·\-]{2,20}?)(?:" + "|".join(_DOSAGE_FORMS) + r")?"
           r"(?:" + "|".join(_ASK_VERBS) + r")")
    for m in re.finditer(pat, question):
        _keep(_clean_drug_cand(m.group(1)))
    for m in re.finditer(r"([一-鿿A-Za-z0-9·\-]{2,20}?)(?:" + "|".join(_DOSAGE_FORMS) +
                         r")?(?:怎么配比|如何配比|剂量多少|用量多少)", question):
        _keep(_clean_drug_cand(m.group(1)))
    return sorted(hits)


def unknown_drug_reply(unknown: list) -> str:
    names = "、".join(f"「{u}」" for u in unknown[:3])
    return (f"⚠ 您提到的{names}不在禾医植保知识库中，无法确认其效果与用法。"
            "网络上的农药推荐请谨慎对待——虚假农药信息会造成真实损失。"
            "建议联系当地植保站核实登记信息后再用药，切勿盲目配比施用。")
