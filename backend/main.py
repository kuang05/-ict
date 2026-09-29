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

@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))

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

@app.get("/api/records")
def records(limit: int = 20):
    with db() as conn:
        rows = conn.execute("SELECT id,ts,class_name,name_cn,confidence,engine FROM records "
                            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r[0], "ts": r[1], "class_name": r[2], "name_cn": r[3],
             "confidence": r[4], "engine": r[5]} for r in rows]
