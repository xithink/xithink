from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from xithink.dialogue import DialogueService
from xithink.engine import XThinkEngine
from xithink.modules.life_simulation import profile_from_name


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_DB = Path(os.environ.get("XITHINK_DB", "data/xithink.db"))
DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="XThink Cognitive Engine", version="0.7.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": "0.7.0"}


@app.get("/api/snapshot")
def snapshot() -> dict:
    engine = XThinkEngine(DEFAULT_DB)
    try:
        return engine.snapshot()
    finally:
        engine.close()


@app.post("/api/simulate")
async def simulate(payload: dict) -> dict:
    total = max(10, min(5000, int(payload.get("total_experiences", 1000))))
    max_age = max(10, min(100, int(payload.get("max_age", 40))))
    seed = int(payload.get("seed", 20261001))
    profile = profile_from_name(str(payload.get("profile", "balanced")))
    for key in ("skepticism", "exploration", "active_test_aggressiveness"):
        if key in payload:
            setattr(profile, key, max(0.0, min(1.0, float(payload[key]))))
    checkpoints = list(range(0, max_age + 1, 10))
    if checkpoints[-1] != max_age:
        checkpoints.append(max_age)

    def run() -> dict:
        engine = XThinkEngine(":memory:")
        try:
            report = engine.simulate_life(
                total_experiences=total, max_age=max_age, age_checkpoints=checkpoints,
                seed=seed, profile=profile, learning_batch=25,
            )
            return report.to_dict()
        finally:
            engine.close()

    return await asyncio.to_thread(run)


@app.websocket("/ws/chat")
async def chat_socket(ws: WebSocket) -> None:
    await ws.accept()
    engine = XThinkEngine(DEFAULT_DB)
    service = DialogueService(engine)
    try:
        await ws.send_json({"type": "ready", "message": "XThink 已连接"})
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"text": raw}

            text = str(payload.get("text", "")).strip()
            cycles = int(payload.get("cycles", 3))
            cycles = max(1, min(cycles, 8))
            environment = str(payload.get("environment", "")).strip() or None
            profile_name = str(payload.get("profile", "balanced")).strip() or "balanced"
            engine.apply_cognitive_profile(profile_from_name(profile_name))
            if not text:
                await ws.send_json({"type": "error", "message": "输入不能为空"})
                continue

            await ws.send_json({"type": "thinking", "message": "正在形成结构化思考…"})
            result = await asyncio.to_thread(service.respond, text, cycles, environment)
            await ws.send_json({"type": "semantic", "data": result["semantic"]})
            await ws.send_json({"type": "symbolic_answer", "data": result["symbolic_answer"]})
            await ws.send_json({"type": "cognition", "data": result["cognition"]})
            for thought in result["thoughts"]:
                await ws.send_json({"type": "thought", "data": thought})
            await ws.send_json({"type": "reply", "text": result["reply"]})
    except WebSocketDisconnect:
        pass
    finally:
        engine.close()


def main() -> None:
    import uvicorn

    uvicorn.run("xithink.web.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
