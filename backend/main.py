from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import List, Optional
import os
import shutil
import json

from backend.geospatial.metadata import extract_metadata
from backend.api.schemas.domain import AnalysisInput, AnalysisQuery, ExecutionTrace
from backend.agents.controller import AgentController
from backend.services.storage_service import StorageService
from backend.services.report_service import ReportService

app = FastAPI(title="SatQuery AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/satquery_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

controller = AgentController()
storage = StorageService()

@app.get("/")
def read_root():
    return {"status": "SatQuery AI API is running"}

@app.post("/api/analyze", response_model=ExecutionTrace)
async def analyze_query(
    background_tasks: BackgroundTasks,
    query: str = Form(...),
    input_type: str = Form(...), # "single", "temporal_pair", "optical_sar_pair"
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    image_paths = []
    metadata_list = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)

        try:
            meta = extract_metadata(file_path)
            metadata_list.append(meta)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Failed to parse {file.filename}: {e}")

    analysis_input = AnalysisInput(input_type=input_type, images=metadata_list)
    analysis_query = AnalysisQuery(text=query)

    trace = controller.process(analysis_query, analysis_input, image_paths)

    # Save to SQLite in background
    if trace.trace_id:
        background_tasks.add_task(
            storage.save_trace,
            trace.trace_id,
            trace.task.task_type,
            query,
            trace.final_result or "",
            trace.final_confidence or 0.0,
            trace.model_dump(mode='json')
        )

    # Save trace to disk temporarily for report generation
    with open(f"/tmp/{trace.trace_id}.json", "w") as f:
        f.write(trace.model_dump_json())

    # Generate thumbnails for frontend
    import base64
    import io
    from PIL import Image
    import numpy as np
    import rasterio

    thumbnails = []
    for path in image_paths:
        try:
            if path.lower().endswith(".tif") or path.lower().endswith(".tiff"):
                with rasterio.open(path) as src:
                    arr = src.read()
                    if arr.shape[0] == 13: # Sentinel-2 optical
                        arr = arr[[3, 2, 1]] # RGB
                        arr = np.clip(arr, 0, 3000) / 3000.0 * 255
                    elif arr.shape[0] == 2: # SAR
                        arr = arr[0:1] # Just VV
                        arr = np.clip(arr, -25, 0) / -25.0 * 255
                        arr = np.repeat(arr, 3, axis=0)
                    elif arr.shape[0] == 1:
                        arr = np.repeat(arr, 3, axis=0)
                    else:
                        arr = arr[:3]
                    arr = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
                    img = Image.fromarray(arr)
            else:
                img = Image.open(path).convert("RGB")

            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            thumbnails.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Thumbnail error:", e)
            thumbnails.append(None)

    if trace.final_evidence is None:
        trace.final_evidence = {}
    trace.final_evidence["input_thumbnails_b64"] = thumbnails

    return trace


@app.get("/api/report/{trace_id}", response_class=PlainTextResponse)
def get_report(trace_id: str):
    try:
        with open(f"/tmp/{trace_id}.json", "r") as f:
            data = json.load(f)
            trace = ExecutionTrace(**data)
            return ReportService.generate_report(trace)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
