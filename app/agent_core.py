# -*- coding: utf-8 -*-
"""
禾医智能体核心(本地对照实现) —— 与 AgentArts 平台侧工作流逻辑一致:
  诊断流: 输入规整 → kb_search → get_prescription → escalation_check → 语言组织(可选LLM) → 组装(来源+升级卡)
  问答流: kb_search(top2) → 受限生成 → 输出+引用
防幻觉护栏: 事实仅来自知识库; 输出强制「来源: KB:<类ID>」; LLM缺席时模板模式(确定性)。
"""
import base64, json, os, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from knowledge_base import KB, dose_calc, remember, context_expand
from llm_client import LLMClient
from audit import audit_answer
from risk_policy import (HIGH_RISK_CLASSES, HIGH_RISK_HINTS, is_high_risk,
                         escalation_text, bridge_class_id, url_safe as _url_safe)
try:
    import vision as local_vision          # 真实视觉底座(CPU可跑)
    VISION_OK = True
except Exception as _e:
    local_vision, VISION_OK = None, False
    print("[HeYi] vision base unavailable:", _e)

kb = KB()
llm = LLMClient()
IMG_ENDPOINT = os.environ.get("HEYI_IMAGE_ENDPOINT", "")

def kb_search(query, topk=2):
    return kb.search(query, topk)

def get_prescription(class_id):
    return kb.get(class_id)

def escalation_card(info, class_id=None):
    """升级卡判定统一走 risk_policy（训练侧加权清单 = 决策侧升级口径）。"""
    if not is_high_risk(class_id, info):
        return None
    return escalation_text(info)

def template_report(info, confidence=None, features=None, lang="zh"):
    lines = [f"【诊断】{info['name_cn']}（{info['class_id']}）"
             + (f"，模型置信度 {confidence*100:.1f}%" if confidence else "")
             + f"，病原/病因：{info['pathogen']}。"]
    if features:
        lines.append(f"【影像依据】{features}")
    lines += [f"【典型症状】{info['symptoms']}",
              f"【发病条件】{info['conditions']}",
              f"【预防】{info['prevention']}",
              f"【治疗处置】{info['treatment']}",
              f"【不干预风险】{info['risk_if_untreated']}",
              f"来源: KB:{info['class_id']}",
              "安全提示：施药时佩戴防护用具并遵守安全间隔期，禁止超量使用。"]
    if lang == "en":
        lines.append("[EN] Diagnosis: " + info["name_en"] + " | Crop: " + info["crop"] +
                     ". The prescription below is in Chinese; an English summary is added in LLM mode.")
    return "\n".join(lines)

def llm_polish(info, confidence=None, features=None, lang="zh"):
    """护栏: 仅把事实JSON改写为通俗报告; system明确禁止新增事实。lang=en时附英文摘要。"""
    facts = template_report(info, confidence, features)
    extra = ("\n另外在报告末尾附上英文摘要(English summary, 80词内, 保留药名拼音/通用名)。"
             if lang == "en" else "")
    out = llm.chat([
        {"role": "system", "content":
         "你是植保助手禾医。把给定事实改写成面向农户的通俗报告(250字内)。"
         "严格约束: 保留全部药名与数字; 禁止出现事实之外的任何药名、剂量、天数; "
         "报告首行必须完整保留【诊断】行中的规范病害名称(如【诊断】番茄晚疫病, 一字不改); "
         "末尾原样保留「来源: KB:...」行。" + extra},
        {"role": "user", "content": facts}])
    if out and "来源: KB:" not in out:
        out += f"\n来源: KB:{info['class_id']}"          # 强制溯源行
    return out or facts


DOSE_HINTS = ("亩", "稀释", "兑水", "用量", "配比", "多少药", "喷多少")

def dose_intent_check(text: str):
    """确定性剂量计算意图: 按单位就近绑定槽位(亩→面积, 克/ml→制剂用量, 倍→稀释倍数)。
    计算外包原则: 数值由确定性工具产生, 不经模型。"""
    if not any(h in text for h in DOSE_HINTS):
        return None
    area = dose = dil = None
    for m in __import__("re").finditer(r"(\d+(?:\.\d+)?)\s*(亩|克|ml|毫升|L|升|倍)?", text):
        v, unit = float(m.group(1)), m.group(2) or ""
        if unit == "亩" and area is None:
            area = v
        elif unit in ("克", "ml", "毫升", "L", "升") and dose is None:
            dose = v
        elif unit == "倍" and dil is None:
            dil = int(v)
    if area is None or dose is None:
        return None
    return {"dose_calc": dose_calc(area, dose, dil or 800),
            "note": "数值由确定性工具计算，非模型生成"}

def diagnose(user_text="", image_result=None, lang="zh"):
    """诊断流。image_result: image_diagnose工具返回(dict)或None。"""
    query = (user_text or "").strip()
    unclear_note = None
    if image_result and image_result.get("verdict") == "unclear":
        c = (image_result.get("top") or {}).get("confidence", 0)
        # M5: 图像不明确但用户附有文字时，不丢弃文字线索——继续走文字诊断流并附注
        if not query:
            return {"mode": "kb-template",
                    "answer": ("未检出明确病斑特征（识别置信度 "
                               + f"{c*100:.1f}%，低于可信门槛）。" + chr(10)
                               + "可能原因：光照/角度/病斑早期。建议：①3天后复拍对比；"
                                 "②补拍病斑特写；③若田间大面积异常，联系当地植保站。"),
                    "refs": [], "escalation": None}
        unclear_note = (f"⚠ 图像未检出明确病斑特征（识别置信度 {c*100:.1f}%，低于可信门槛），"
                        "以下按文字描述判断。\n")
    if image_result and image_result.get("top"):
        top = image_result["top"]
        cls = top.get("class_id") or top.get("class_name")     # 兼容本地/端点两种键
        query = (query + " " + cls).strip()
        confidence = top.get("confidence")
    else:
        confidence = None
    hits = kb_search(query or user_text, topk=2)
    if not hits:
        return {"mode": "kb-template",
                "answer": (("⚠ 图像不明确，" if unclear_note else "")
                           + "知识库暂无匹配内容，建议联系当地植保站。")
                + (("\n" + unclear_note) if unclear_note else ""),
                "refs": [], "escalation": None}
    top = (image_result or {}).get("top") or {}
    class_id = top.get("class_id") or top.get("class_name") or hits[0]["class_id"]
    class_id = bridge_class_id(class_id)          # 模型命名空间 -> 知识库命名空间
    info = get_prescription(class_id)
    if not info:
        class_id = hits[0]["class_id"]            # 检索兜底, 两者保持一致(见 L1)
        info = get_prescription(class_id)
    features = image_result.get("features_text") if image_result else None
    report = (llm_polish(info, confidence, features, lang) if llm.enabled
             else template_report(info, confidence, features, lang))
    card = escalation_card(info, class_id)
    if card and card not in report:
        report += "\n" + card
    if unclear_note:
        report = unclear_note + report
    rec = record(info["class_id"], confidence, report)
    audit = audit_answer(report, info["class_id"], kb)
    return {"mode": "llm" if llm.enabled else "kb-template",
            "answer": report, "refs": [info["class_id"]] + [h["class_id"] for h in hits[1:]],
            "escalation": card, "record_id": rec["record_id"], "audit": audit}

def qa(question):
    hits = kb_search(question, topk=2)
    if llm.enabled and hits:
        ctx = "\n".join(json.dumps(h, ensure_ascii=False) for h in hits)
        ans = llm.chat([
            {"role": "system", "content":
             "你是植保助手禾医。只依据知识库片段回答; 片段不足时固定回复「知识库暂无匹配内容，建议联系当地植保站」。"
             "回答末尾列出引用类ID。"},
            {"role": "user", "content": f"知识库片段:\n{ctx}\n\n问题: {question}"}])
        if ans:
            return {"mode": "llm", "answer": ans, "refs": [h["class_id"] for h in hits]}
    if hits:
        best = kb.get(hits[0]["class_id"])
        return {"mode": "kb-template",
                "answer": (f"根据知识库，您的问题最接近『{best['name_cn']}』:\n"
                           f"· 症状: {best['symptoms']}\n"
                           f"· 防治: {best['treatment']}\n"
                           f"来源: KB:{best['class_id']}"),
                "refs": [h["class_id"] for h in hits]}
    return {"mode": "kb-template", "answer": "知识库暂无匹配内容，建议联系当地植保站。", "refs": []}

def stream_events(user_text="", image_result=None):
    """SSE事件流: route → source → escalation → answer(分块)。LLM模式answer单事件。"""
    if image_result and (image_result.get("top") or image_result.get("status") == "ok"):
        result, flow = diagnose(user_text, image_result), "diagnose"
    else:
        result, flow = route(user_text)
        if flow == "dose":
            yield json.dumps({"event": "answer", "text": result["answer"]}, ensure_ascii=False)
            return
    yield json.dumps({"event": "route", "flow": flow, "mode": result.get("mode")}, ensure_ascii=False)
    yield json.dumps({"event": "source", "refs": result.get("refs", [])}, ensure_ascii=False)
    if result.get("escalation"):
        yield json.dumps({"event": "escalation", "text": result["escalation"]}, ensure_ascii=False)
    ans = result.get("answer", "")
    step = max(1, len(ans) // 5)
    for i in range(0, len(ans), step):
        yield json.dumps({"event": "answer", "text": ans[i:i + step]}, ensure_ascii=False)
    if result.get("record_id"):
        yield json.dumps({"event": "record", "record_id": result["record_id"]}, ensure_ascii=False)

CONF_GATE = 0.78   # 置信度门槛: 校准于 AgML corn 4188张(健康误报100%→~22%, 病害召回93%+)

def _url_safe(u: str) -> bool:
    """SSRF防护: 仅允许公网 http/https, 拒绝内网/回环/保留地址。"""
    import ipaddress, socket
    from urllib.parse import urlparse
    try:
        p = urlparse(u)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        for inf in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(inf[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except Exception:
        return False

def image_diagnose(image_bytes=b"", image_url="", image_base64="", with_gradcam=False):
    """图像诊断工具(三级): ①本地视觉底座(v3_hr, 27类, CPU) ②远程端点 ③降级提示。
    字节流优先于URL/base64, 规避服务端SSRF面。"""
    data = image_bytes or (base64.b64decode(image_base64) if image_base64 else b"")
    if data and VISION_OK:
        try:
            r = local_vision.predict(data)
            top = r["predictions"][0]
            # 玉米健康/病害二分类门(99.91%): 健康玉米不再被强行判为病害
            if top["class_id"].startswith("Corn___"):
                g = local_vision.corn_gate(data)
                if g["healthy"] and g["confidence"] >= 0.90:
                    info_h = kb.get("Corn___healthy")
                    rec = record("Corn___healthy", g["confidence"],
                                 "二分类门判定: 健康玉米叶片, 无需处理")
                    return {"status": "ok", "engine": r["engine"] + "+corn_gate",
                            "latency_ms": r["latency_ms"], "verdict": "healthy",
                            "conf_gate": CONF_GATE, "gate": g,
                            "top": {"class_id": "Corn___healthy",
                                    "confidence": g["confidence"],
                                    "kb_source": "KB:Corn___healthy"},
                            "predictions": r["predictions"],
                            "coverage_note": "27类输出空间(PlantDoc协议)+玉米健康门"}
            verdict = "disease" if top["confidence"] >= CONF_GATE else "unclear"
            out = {"status": "ok", "engine": r["engine"], "latency_ms": r["latency_ms"],
                   "verdict": verdict, "conf_gate": CONF_GATE,
                   "top": {**top, "kb_source": "KB:" + top["class_id"]},
                   "predictions": r["predictions"],
                   "coverage_note": "27类输出空间(PlantDoc协议)"}
            if with_gradcam:
                g = local_vision.gradcam(data)
                out["heatmap_png_b64"] = g["heatmap_png_b64"]
            return out
        except Exception as e:
            print("[HeYi-vision] local failed:", e)
    if IMG_ENDPOINT and (image_url or image_base64):
        if image_url and not _url_safe(image_url):
            return {"status": "error", "detail": "image_url not allowed (SSRF guard)"}
        payload = json.dumps({"image_url": image_url, "image_base64": image_base64}).encode()
        req = urllib.request.Request(IMG_ENDPOINT, data=payload, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read().decode())
            if out.get("top"):
                out["features_text"] = "识别引擎: " + str(out.get("engine", "-")) + "，top1=" + str(out["top"].get("name_cn"))
            return out
        except Exception as e:
            pass
    return {"status": "endpoint_unavailable",
            "hint": "识别底座未加载且端点未配置。请改用文字描述症状。"}

RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records.jsonl")

def record(class_id, confidence, report):
    import time, uuid
    rid = uuid.uuid4().hex[:12]
    with open(RECORDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"record_id": rid, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "class_id": class_id, "confidence": confidence,
                            "report": report}, ensure_ascii=False) + chr(10))
    return {"record_id": rid}

def route(user_text="", image_result=None, lang="zh"):
    """意图路由: 图片→诊断流; 命中病害类→诊断流(处方+升级卡); 健康/背景/未命中→问答流。"""
    if image_result and (image_result.get("top") or image_result.get("status") == "ok"):
        return diagnose(user_text, image_result, lang), "diagnose"
    if image_result and image_result.get("status") in ("endpoint_unavailable", "error"):
        return {"mode": "degraded", "answer": image_result.get("hint", "图像诊断暂不可用，请用文字描述症状。"),
                "refs": [], "escalation": None}, "degraded"
    q = (user_text or "").strip()
    d = dose_intent_check(q)
    if d:
        return {"mode": "tool-dose", "answer": json.dumps(d, ensure_ascii=False), "refs": []}, "dose"
    hits = kb_search(q, topk=2) if q else []
    top = hits[0]["class_id"] if hits else ""
    info = get_prescription(top) if top else None
    is_disease = bool(info) and info.get("severity") not in ("无", "-", None)                  and "Background" not in top and "healthy" not in top
    if is_disease:
        return diagnose(q, lang=lang), "diagnose"
    return qa(q), "qa"
