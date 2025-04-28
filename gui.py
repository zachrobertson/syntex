import os
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # TODO: Connect to RAG backend
            # For now, just echo the message
            await websocket.send_json({"text": f"Echo: {data['text']}"})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8080)
