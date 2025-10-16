from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from src.infer import CodeGenService

app = FastAPI(title="Code Generation API")
svc = None

class GenReq(BaseModel):
    prompt: str
    temperature: float = 0.2
    top_p: float = 0.95

@app.on_event("startup")
def _load():
    global svc
    try:
        svc = CodeGenService()
    except Exception:
        svc = None

@app.get("/health")
def health():
    return {"status":"ok","model_loaded": svc is not None}

@app.post("/generate")
def generate(req: GenReq):
    if svc is None:
        return {"error":"Model not loaded. Train and save to models/ first."}
    out = svc.generate(req.prompt, temperature=req.temperature, top_p=req.top_p)
    return {"completion": out}
