"""FastAPI web dashboard for HCR2 emulator monitoring."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
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
    # Ensure stream is running
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
