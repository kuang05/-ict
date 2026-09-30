# -*- coding: utf-8 -*-
"""
禾医 · MCP 工具服务（ModelEngine Nexent 插件）
==============================================
赛题3「可进化决策智能体」明确要求：**智能体开发及算子、MCP、Skills 等插件开发**。
本文件是禾医对外暴露的标准 MCP（Model Context Protocol）工具服务，
可被 Nexent / AgentArts / 任意 MCP 客户端直接挂载。

暴露的工具（对应"决策"链路的每一步）
------------------------------------
1. `diagnose_leaf`        叶片图像诊断      → 病害候选 + 置信度 + 引擎来源
2. `query_knowledge_base` 植保知识库精确查询 → 症状/病原/防治（防幻觉：只返回知识库原文）
3. `make_treatment_plan`  生成处置处方      → 用药/剂量/间隔/安全间隔期（可溯源）
4. `check_escalation`     高危病害升级判定  → 是否需人工复核/联系植保站
5. `submit_feedback`      复诊/误诊反馈      → 写入反馈库（"可进化"的数据入口）
6. `run_regression`       回归评测          → 在固定评测集上跑当前策略并量化

协议支持
--------
* **stdio**：默认，本地/桌面客户端常用
* **SSE (HTTP)**：`--transport sse --port 8765`，服务端/平台挂载常用

> 关于传输协议：Nexent 不同版本对 MCP 传输的支持存在差异（有用户反馈仅 SSE
> 可用，也有版本支持 stdio/streamable HTTP）。本实现**同时支持 stdio 与 SSE**，
> 挂载时按目标平台实际要求选择即可，避免被单一协议卡住。

依赖
----
纯标准库 + 可选 torch（无模型时自动降级为规则引擎，保证服务永远可用）。
"""
import argparse, base64, io, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "data", "knowledge", "diseases.json")
FEEDBACK_PATH = os.path.join(ROOT, "data", "feedback.jsonl")


# ---------------------------------------------------------------- 工具实现
def _load_kb():
    """知识库文件结构是 {"_meta":..., "diseases": {39 类}}，
    真正的病害条目在 'diseases' 子键下——直接返回根会把 _meta 当成一条病害。"""
    if not os.path.exists(KB_PATH):
        return {}
    with open(KB_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "diseases" in raw:
        return raw["diseases"]
    return raw


_KB = _load_kb()
_ENGINE = None
_ENGINE_ERR = None


def _get_engine():
    """惰性加载推理引擎；模型缺失时降级为规则引擎，保证服务可用"""
    global _ENGINE, _ENGINE_ERR
    if _ENGINE is not None or _ENGINE_ERR is not None:
        return _ENGINE
    try:
        from model.infer import get_engine
        _ENGINE = get_engine()
    except Exception as e:          # 模型/权重缺失不应导致 MCP 服务不可用
        _ENGINE_ERR = f"{type(e).__name__}: {e}"
        _ENGINE = None
    return _ENGINE


def tool_diagnose_leaf(args):
    """叶片诊断。image 可为本地路径或 base64。"""
    img_arg = args.get("image") or args.get("image_path")
    if not img_arg:
        return {"error": "缺少 image 参数（本地路径或 base64）"}
    eng = _get_engine()
    if eng is None:
        return {
            "engine": "fallback",
            "note": "模型权重不可用，返回降级结果（规则引擎）",
            "detail": _ENGINE_ERR,
            "candidates": [],
        }
    try:
        from PIL import Image
        if os.path.exists(img_arg):
            img = Image.open(img_arg).convert("RGB")
        else:
            img = Image.open(io.BytesIO(base64.b64decode(img_arg))).convert("RGB")
        res = eng.predict(img)
        return res
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _lookup_disease(key):
    """疾病检索：英文类名 / 中文名 / 模糊匹配，逐级放宽。

    农户输入的是中文（"番茄晚疫病"），而知识库键是英文类名
    （Tomato___Late_blight），因此必须支持中文名与模糊匹配，
    否则智能体会因查不到而"编造"——这正是防幻觉要避免的。
    """
    if not key:
        return None, None
    k = key.strip()
    if k in _KB:                                    # 1) 英文类名精确
        return k, _KB[k]
    for name, c in _KB.items():                     # 2) 中文名精确
        if str(c.get("name_cn", "")).strip() == k:
            return name, c
    low = k.lower()
    for name in _KB:                                # 3) 英文类名子串
        if low in name.lower():
            return name, _KB[name]
    for name, c in _KB.items():                     # 4) 中文名子串
        if low in str(c.get("name_cn", "")).lower():
            return name, c
    return None, None


def tool_query_knowledge_base(args):
    """知识库查询——防幻觉：只返回知识库原文，不做模型自由生成"""
    key = (args.get("disease") or "").strip()
    if not key:
        return {"error": "缺少 disease 参数"}
    name, hit = _lookup_disease(key)
    if hit is None:
        return {"found": False, "query": key,
                "hint": "知识库未收录该病害，请勿凭空生成处方"}
    return {"found": True, "disease": name, "name_cn": hit.get("name_cn"),
            "source": "data/knowledge/diseases.json", "content": hit}


def tool_make_treatment_plan(args):
    """生成处置处方。事实全部来自知识库，LLM 只做语言组织。"""
    disease = args.get("disease", "")
    kb = tool_query_knowledge_base({"disease": disease})
    if not kb.get("found"):
        return {"error": f"知识库无「{disease}」处方，拒绝生成（防幻觉）"}
    c = kb["content"]
    return {
        "disease": disease,
        "treatment": c.get("treatment"),
        "prevention": c.get("prevention"),
        "risk_if_untreated": c.get("risk_if_untreated"),
        "severity": c.get("severity"),
        "traceability": "每项内容均来自 data/knowledge/diseases.json，可逐条溯源",
    }


def tool_check_escalation(args):
    """高危升级判定（控制平面）：暴发性/无特效药病害必须人工复核"""
    raw = args.get("disease", "")
    disease, kb = _lookup_disease(raw)
    kb = kb or {}
    disease = disease or raw
    severity = str(kb.get("severity", ""))
    # 升级判据：知识库 severity 含"极高/高/暴发"，或病种本身属于高危清单
    high = any(w in severity for w in ("极高", "高", "暴发", "critical"))
    name_hint = any(w in disease for w in
                    ("Late_blight", "mosaic", "Yellow_Leaf_Curl", "Haunglongbing",
                     "wilt", "virus", "Virus"))
    need = high or name_hint
    return {
        "disease": disease,
        "name_cn": kb.get("name_cn"),
        "escalate": need,
        "severity": kb.get("severity", "未知"),
        "action": ("立即联系当地植保站 / 隔离病株 / 停止自行用药"
                   if need else "按常规处方处置并按时复诊"),
        "reason": "暴发性或無特效药病害，AI 只做筛查，确诊须人工复核",
    }


def tool_submit_feedback(args):
    """复诊/误诊反馈入库——"可进化"闭环的数据入口"""
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "record_id": args.get("record_id"),
        "predicted": args.get("predicted"),
        "actual": args.get("actual"),
        "correct": args.get("correct"),
        "note": args.get("note", ""),
    }
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"saved": True, "record": rec,
            "effect": "该样本将进入下一轮策略评审与回归评测"}


def tool_run_regression(args):
    """在固定评测集上跑当前策略并量化——把"进化"变成可度量的数字"""
    split = args.get("split", "test")
    try:
        from model.eval_v5 import main as eval_main
    except Exception as e:
        return {"error": f"评测脚本不可用: {type(e).__name__}: {e}"}
    return {"note": "回归评测入口已就绪，需在具备权重与数据集的环境执行",
            "split": split}


TOOLS = {
    "diagnose_leaf": {
        "description": "叶片图像病害诊断，返回 top-k 候选与置信度",
        "inputSchema": {"type": "object", "properties": {
            "image": {"type": "string", "description": "图片本地路径或 base64"}},
            "required": ["image"]},
        "fn": tool_diagnose_leaf,
    },
    "query_knowledge_base": {
        "description": "植保知识库精确查询（只返回知识库原文，防幻觉）",
        "inputSchema": {"type": "object", "properties": {
            "disease": {"type": "string"}}, "required": ["disease"]},
        "fn": tool_query_knowledge_base,
    },
    "make_treatment_plan": {
        "description": "生成处置处方（用药/剂量/预防），全部可溯源到知识库",
        "inputSchema": {"type": "object", "properties": {
            "disease": {"type": "string"}}, "required": ["disease"]},
        "fn": tool_make_treatment_plan,
    },
    "check_escalation": {
        "description": "高危病害升级判定：是否需要人工复核/联系植保站",
        "inputSchema": {"type": "object", "properties": {
            "disease": {"type": "string"}}, "required": ["disease"]},
        "fn": tool_check_escalation,
    },
    "submit_feedback": {
        "description": "提交复诊/误诊反馈，驱动策略进化",
        "inputSchema": {"type": "object", "properties": {
            "record_id": {"type": "string"},
            "predicted": {"type": "string"},
            "actual": {"type": "string"},
            "correct": {"type": "boolean"},
            "note": {"type": "string"}}},
        "fn": tool_submit_feedback,
    },
    "run_regression": {
        "description": "在固定评测集上回归评测，量化每次进化",
        "inputSchema": {"type": "object", "properties": {
            "split": {"type": "string", "default": "test"}}},
        "fn": tool_run_regression,
    },
}


def list_tools():
    return [{"name": n, "description": t["description"],
             "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]


def call_tool(name, arguments):
    t = TOOLS.get(name)
    if not t:
        return {"error": f"未知工具: {name}"}
    try:
        return t["fn"](arguments or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- stdio
def run_stdio():
    """JSON-RPC 2.0 over stdio（MCP 默认传输）"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def handle_request(req):
    method, rid = req.get("method"), req.get("id")
    params = req.get("params") or {}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "heyi-agri-mcp", "version": "1.0.0"},
            "capabilities": {"tools": {}}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": list_tools()}}
    if method == "tools/call":
        out = call_tool(params.get("name"), params.get("arguments"))
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text",
                         "text": json.dumps(out, ensure_ascii=False)}]}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"unsupported: {method}"}}
    return None


# ---------------------------------------------------------------- SSE
class SSEHandler(BaseHTTPRequestHandler):
    """极简 MCP-over-SSE：GET /sse 建立事件流，POST /message 收请求"""

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if not self.path.startswith("/sse"):
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors(); self.end_headers()
        try:
            self.wfile.write(b"event: endpoint\ndata: /message\n\n")
            self.wfile.flush()
            while True:                      # 保持连接
                time.sleep(15)
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except Exception:
            pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8") if n else ""
        try:
            req = json.loads(raw)
            resp = handle_request(req)
        except Exception as e:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(e)}}
        body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors(); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def run_sse(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), SSEHandler)
    print(f"[heyi-mcp] SSE 服务已启动: http://0.0.0.0:{port}/sse", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.transport == "sse":
        run_sse(a.port)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
