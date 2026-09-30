# -*- coding: utf-8 -*-
"""
禾医·后端服务 (FastAPI)
========================
接口:
  GET  /                  -> 前端演示页
  GET  /api/health        -> 引擎状态(当前使用的推理引擎/LLM配置)
  POST /api/diagnose      -> 上传叶片照片 -> 识别 -> 智能体报告 (multipart: file)
  POST /api/chat          -> 农事问答智能体          (json: {question})
  GET  /api/records       -> 历史诊断记录(SQLite)
启动:
  uvicorn backend.main:app --reload --port 8000
"""
import io, json, os, sqlite3, sys, time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import infer as model_infer          # noqa: E402
from agent import agent as heyi_agent           # noqa: E402

app = FastAPI(title="禾医 HeYi API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB = os.path.join(ROOT, "data", "heyi.sqlite3")

def db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, class_name TEXT, name_cn TEXT, confidence REAL,
        engine TEXT, report TEXT)""")
    return conn

# 静态资源（手机端 PWA 的 manifest / service worker / 图标）
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))

@app.get("/mobile")
def mobile():
    """手机端页面（PWA，可"添加到主屏幕"当 App 用）"""
    return FileResponse(os.path.join(ROOT, "frontend", "mobile.html"))

@app.get("/api/health")
def health():
    eng = model_infer.get_engine()
    engine_name = eng if eng == "heuristic" else eng.__class__.__name__
    return {"status": "ok", "engine": engine_name,
            "llm": heyi_agent.llm.enabled, "classes": len(model_infer.LABELS)}

@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "无法解析图片，请上传JPG/PNG叶片照片")
    result = model_infer.predict(img)
    top = result["predictions"][0]
    explained = heyi_agent.explain_diagnosis(top["class_name"], top["confidence"],
                                             result.get("features"))
    with db() as conn:
        conn.execute("INSERT INTO records(ts,class_name,name_cn,confidence,engine,report) VALUES(?,?,?,?,?,?)",
                     (time.strftime("%Y-%m-%d %H:%M:%S"), top["class_name"],
                      explained["plan"]["diagnosis"]["name_cn"], top["confidence"],
                      result["engine"], explained["report"]))
    return JSONResponse({
        "engine": result["engine"], "latency_ms": int((time.time() - t0) * 1000),
        "top": {**top, "name_cn": explained["plan"]["diagnosis"]["name_cn"]},
        "predictions": result["predictions"],
        "features": result.get("features"),
        "plan": explained["plan"], "report": explained["report"],
        "llm_used": explained["llm_used"],
    })

@app.post("/api/chat")
async def chat(payload: dict):
    q = (payload or {}).get("question", "").strip()
    if not q:
        raise HTTPException(400, "question 不能为空")
    return heyi_agent.chat(q)

@app.post("/api/feedback")
async def feedback(payload: dict):
    """复诊/误诊反馈入库 —— "可进化"闭环的数据入口。

    与 MCP 工具 submit_feedback 写入**同一份** data/feedback.jsonl，
    保证手机端反馈与智能体反馈进入同一条进化通道：
    反馈 → 策略评审 → 回归评测（model/eval_evolve.py）→ 版本化。
    """
    p = payload or {}
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "record_id": p.get("record_id"),
        "predicted": p.get("predicted"),
        "actual": p.get("actual"),
        "correct": p.get("correct"),
        "note": p.get("note", ""),
        "source": "mobile",
    }
    path = os.path.join(ROOT, "data", "feedback.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"saved": True, "record": rec,
            "effect": "该样本将进入下一轮策略评审与回归评测"}

@app.get("/api/records")
def records(limit: int = 20):
    with db() as conn:
        rows = conn.execute("SELECT id,ts,class_name,name_cn,confidence,engine FROM records "
                            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r[0], "ts": r[1], "class_name": r[2], "name_cn": r[3],
             "confidence": r[4], "engine": r[5]} for r in rows]
