"""FastAPI web dashboard for HCR2 emulator monitoring."""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from web.emulator import (
    get_all_status,
    restart_emulator,
    start_emulator,
    stop_emulator,
    start_game,
    stop_game,
)
from web.streamer import StreamManager

# 1x1 чёрный JPEG — placeholder когда кадр ещё не получен
import cv2
import numpy as np
_, _buf = cv2.imencode(".jpg", np.zeros((1, 1, 3), dtype=np.uint8))
_PLACEHOLDER_JPEG = _buf.tobytes()


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


@app.get("/snapshot/{emu_id}")
async def snapshot(emu_id: int):
    """Single JPEG snapshot (used by polling-based display)."""
    frame = stream_manager.get_frame(emu_id)
    if frame:
        return Response(content=frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})
    # Нет кадра — 1x1 чёрный JPEG чтобы фронт не мигал
    return Response(content=_PLACEHOLDER_JPEG, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
async def api_status():
    statuses = get_all_status()
    return [s.to_dict() for s in statuses]


@app.post("/api/emulator/{emu_id}/restart")
async def api_restart(emu_id: int):
    ok, msg = restart_emulator(emu_id)
    if ok:
        # Restart the stream
        stream_manager.ensure_emulator(emu_id)
    return JSONResponse(
        content={"ok": ok, "message": msg},
        status_code=200 if ok else 500,
    )


@app.post("/api/emulator/{emu_id}/start")
async def api_start(emu_id: int):
    ok, msg = start_emulator(emu_id)
    if ok:
        stream_manager.ensure_emulator(emu_id)
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


# ---------------------------------------------------------------------------
# Training metrics API
# ---------------------------------------------------------------------------

def _read_jsonl_tail(filepath: Path, limit: int, env_idx: Optional[int] = None,
                     event_type: Optional[str] = None) -> list[dict]:
    """Read last `limit` lines from a JSONL file with optional filtering.

    Uses seek-from-end to avoid reading the entire file on every request.
    """
    if not filepath.exists():
        return []
    has_filter = env_idx is not None or event_type is not None
    # Without filters: read only tail of the file (seek from end)
    # With filters: must scan whole file (filter reduces result set)
    read_limit = limit * 20 if has_filter else limit * 2  # read extra for safety
    try:
        with open(filepath, "rb") as fb:
            fb.seek(0, 2)
            fsize = fb.tell()
            # Estimate ~200 bytes per line, read enough bytes
            chunk = min(fsize, read_limit * 200)
            fb.seek(max(0, fsize - chunk))
            if fb.tell() > 0:
                fb.readline()  # skip partial first line
            raw_lines = fb.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records: list[dict] = []
    for line in raw_lines:
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
