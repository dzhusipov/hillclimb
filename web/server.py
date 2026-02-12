"""FastAPI web dashboard for HCR2 emulator monitoring."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from web.emulator import (
    get_all_status,
    restart_emulator,
    start_emulator,
    stop_emulator,
    start_game,
    stop_game,
    touch_down,
    touch_up,
)
from web.streamer import StreamManager


class TouchEvent(BaseModel):
    action: str  # "down" or "up"
    x: float = 0.0  # normalized 0..1
    y: float = 0.0  # normalized 0..1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
LOGS_DIR = Path("/app/logs") if Path("/app/logs").exists() else BASE_DIR.parent / "logs"
stream_manager = StreamManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: discover emulators and start streams
    statuses = get_all_status()
    running_ids = [s.id for s in statuses if s.docker_status == "running"]
    if running_ids:
        stream_manager.start_streams(running_ids)
        logger.info("Started streams for emulators: %s", running_ids)
    else:
        logger.warning("No running emulators found")
    yield
    # Shutdown
    stream_manager.stop_all()


app = FastAPI(title="HCR2 Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    statuses = get_all_status()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "emulators": [s.to_dict() for s in statuses],
    })


@app.get("/stream/{emu_id}")
async def video_stream(emu_id: int):
    """MJPEG stream (legacy, limited by browser connection pool)."""
    stream_manager.get_or_create(emu_id)

    async def generate():
        while True:
            frame = stream_manager.get_frame(emu_id)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot/{emu_id}")
async def snapshot(emu_id: int):
    """Single JPEG snapshot (used by polling-based display for >6 emus)."""
    from fastapi.responses import Response
    stream_manager.get_or_create(emu_id)
    frame = stream_manager.get_frame(emu_id)
    if frame:
        return Response(content=frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})
    return Response(status_code=204)


@app.websocket("/ws/stream/{emu_id}")
async def ws_stream(websocket: WebSocket, emu_id: int):
    """WebSocket: push JPEG frames at ~30 FPS."""
    await websocket.accept()
    stream_manager.get_or_create(emu_id)
    prev_frame = None
    try:
        while True:
            frame = stream_manager.get_frame(emu_id)
            if frame is not None and frame is not prev_frame:
                await websocket.send_bytes(frame)
                prev_frame = frame
            await asyncio.sleep(1 / 30)
    except (WebSocketDisconnect, Exception):
        pass


@app.get("/api/status")
async def api_status():
    statuses = get_all_status()
    return [s.to_dict() for s in statuses]


@app.post("/api/emulator/{emu_id}/restart")
async def api_restart(emu_id: int):
    ok, msg = restart_emulator(emu_id)
    if ok:
        # Restart the stream
        stream_manager.get_or_create(emu_id)
    return JSONResponse(
        content={"ok": ok, "message": msg},
        status_code=200 if ok else 500,
    )


@app.post("/api/emulator/{emu_id}/start")
async def api_start(emu_id: int):
    ok, msg = start_emulator(emu_id)
    if ok:
        stream_manager.get_or_create(emu_id)
    return JSONResponse(
        content={"ok": ok, "message": msg},
        status_code=200 if ok else 500,
    )


@app.post("/api/emulator/{emu_id}/stop")
async def api_stop(emu_id: int):
    ok, msg = stop_emulator(emu_id)
    return JSONResponse(
        content={"ok": ok, "message": msg},
        status_code=200 if ok else 500,
    )


@app.post("/api/emulator/{emu_id}/start-game")
async def api_start_game(emu_id: int):
    ok, msg = start_game(emu_id)
    return JSONResponse(
        content={"ok": ok, "message": msg},
        status_code=200 if ok else 500,
    )


@app.post("/api/emulator/{emu_id}/stop-game")
async def api_stop_game(emu_id: int):
    ok, msg = stop_game(emu_id)
    return JSONResponse(
        content={"ok": ok, "message": msg},
        status_code=200 if ok else 500,
    )


@app.post("/api/emulator/{emu_id}/touch")
async def api_touch(emu_id: int, event: TouchEvent):
    if event.action == "down":
        rotated, (cap_w, cap_h) = stream_manager.get_stream_info(emu_id)
        ok, msg = touch_down(emu_id, event.x, event.y, rotated, cap_w, cap_h)
    elif event.action == "up":
        ok, msg = touch_up(emu_id)
    else:
        return JSONResponse(content={"ok": False, "message": "Unknown action"}, status_code=400)
    return JSONResponse(content={"ok": ok, "message": msg})


# ---------------------------------------------------------------------------
# Training metrics API
# ---------------------------------------------------------------------------

def _read_jsonl_tail(filepath: Path, limit: int, env_idx: Optional[int] = None,
                     event_type: Optional[str] = None) -> list[dict]:
    """Read last `limit` lines from a JSONL file with optional filtering."""
    if not filepath.exists():
        return []
    records: list[dict] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if env_idx is not None and rec.get("env_idx") != env_idx:
                continue
            if event_type is not None and rec.get("event") != event_type:
                continue
            records.append(rec)
    return records[-limit:]


@app.get("/api/training/status")
async def api_training_status():
    """Current training status (atomic JSON, updated every 10 episodes)."""
    status_file = LOGS_DIR / "training_status.json"
    if not status_file.exists():
        return {"training_active": False}
    try:
        with open(status_file) as f:
            data = json.load(f)
        data["stale"] = (time.time() - data.get("last_update", 0)) > 120
        return data
    except (json.JSONDecodeError, KeyError):
        return {"training_active": False, "error": "corrupted"}


@app.get("/api/training/episodes")
async def api_training_episodes(
    limit: int = Query(default=200, ge=1, le=2000),
    env_idx: Optional[int] = Query(default=None),
):
    """Last N training episodes (JSONL)."""
    episodes = _read_jsonl_tail(LOGS_DIR / "train_episodes.jsonl", limit, env_idx=env_idx)
    return {"episodes": episodes, "total": len(episodes)}


@app.get("/api/training/events")
async def api_training_events(
    limit: int = Query(default=50, ge=1, le=500),
    env_idx: Optional[int] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
):
    """Last N navigation events (JSONL)."""
    events = _read_jsonl_tail(LOGS_DIR / "nav_events.jsonl", limit,
                              env_idx=env_idx, event_type=event_type)
    return {"events": events}
