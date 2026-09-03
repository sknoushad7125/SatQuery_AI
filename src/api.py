from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.agent.schemas import SatQueryResponse, ImageInput, Modality
from src.agent.controller import SatQueryController

app = FastAPI(title="SatQuery AI Backend")
controller = SatQueryController()

class QueryRequest(BaseModel):
    images: List[ImageInput]
    query: Optional[str] = None

@app.post("/api/query", response_model=SatQueryResponse)
def api_query(request: QueryRequest):
    try:
        response = controller.process_query(request.images, request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tools")
def get_tools():
    return list(controller.registry.get_all_tools().keys())

@app.get("/api/health")
def health():
    return {"status": "ok"}
