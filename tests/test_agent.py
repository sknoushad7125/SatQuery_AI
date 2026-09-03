import pytest
from src.agent.schemas import ImageInput, Modality
from src.agent.controller import SatQueryController

@pytest.fixture
def controller():
    return SatQueryController()

def test_validation_missing_file(controller):
    with pytest.raises(ValueError, match="Missing file"):
        controller.process_query([ImageInput(filepath="doesnotexist.png")], "test")

def test_single_image_captioning(controller, tmp_path):
    img = tmp_path / "test1.png"
    from PIL import Image
    Image.new("RGB", (256, 256)).save(img)
    
    res = controller.process_query([ImageInput(filepath=str(img))], "describe the scene")
    assert res.workflow == "captioning"
    assert "MockCaptioningModel-DEMO" in res.execution_trace.models

def test_bitemporal_change(controller, tmp_path):
    img1 = tmp_path / "test1.png"
    img2 = tmp_path / "test2.png"
    from PIL import Image
    Image.new("RGB", (256, 256)).save(img1)
    Image.new("RGB", (256, 256)).save(img2)
    
    res = controller.process_query([ImageInput(filepath=str(img1)), ImageInput(filepath=str(img2))], "what changed?")
    assert res.workflow in ["change_analysis", "change_vqa"]
    assert "change_detector" in res.execution_trace.selected_tools
    
def test_optical_sar(controller, tmp_path):
    img1 = tmp_path / "test1.png"
    img2 = tmp_path / "test2.png"
    from PIL import Image
    Image.new("RGB", (256, 256)).save(img1)
    Image.new("RGB", (256, 256)).save(img2)
    
    res = controller.process_query([
        ImageInput(filepath=str(img1), modality=Modality.OPTICAL), 
        ImageInput(filepath=str(img2), modality=Modality.SAR)
    ], "analyze fusion")
    assert res.workflow == "optical_sar_analysis"
