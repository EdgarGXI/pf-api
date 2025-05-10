from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from ultralytics import YOLO
import shutil
import os
import uuid
import cv2
import base64

model = None  # modelo global

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("🟡 Iniciando aplicación y cargando modelo...")
    try:
        model = YOLO("best.pt")
        print("🟢 Modelo cargado correctamente.")
    except Exception as e:
        print("🔴 Error al cargar el modelo:", e)
    yield  # Aquí se arranca la app
    print("🔴 Finalizando aplicación...")

app = FastAPI(lifespan=lifespan)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    temp_input_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_input_path)

    response = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            clase = int(box.cls[0])
            confianza = float(box.conf[0])
            coords = list(map(float, box.xyxy[0]))
            response.append({
                "class": clase,
                "confidence": confianza,
                "bbox": coords
            })

    annotated_image = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated_image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    os.remove(temp_input_path)

    return JSONResponse(content={
        "results": response,
        "annotated_image_base64": img_base64
    })

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "API is running"}
