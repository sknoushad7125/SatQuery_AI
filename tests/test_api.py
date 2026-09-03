import pytest
from fastapi.testclient import TestClient
from backend.main import app
import os
import io
from PIL import Image

client = TestClient(app)

def create_dummy_image(name, ext="jpg"):
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format=ext.upper() if ext.upper() != "JPG" else "JPEG")
    buf.seek(0)
    return (name, buf, f"image/{ext}")

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "SatQuery AI API is running"}

def test_analyze_single_image_vqa():
    # We will test the API by simulating a file upload
    file_tuple = create_dummy_image("test1.jpg")
    
    response = client.post(
        "/api/analyze",
        data={
            "query": "Is there a water body?",
            "input_type": "single"
        },
        files={"files": file_tuple}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "trace_id" in data
    assert data["task"]["task_type"] == "single_vqa" # Since query has no 'highlight', 'where', 'describe'
    
    if data["final_result"] is None:
        print("ERROR DUMP:", data)
    assert data["final_result"] is not None

def test_analyze_captioning():
    file_tuple = create_dummy_image("test2.jpg")
    
    response = client.post(
        "/api/analyze",
        data={
            "query": "Describe the image.",
            "input_type": "single"
        },
        files={"files": file_tuple}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["task"]["task_type"] == "captioning"

