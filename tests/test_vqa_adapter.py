import pytest
import os
from src.agent.schemas import ToolRequest, ImageInput
from src.tools.single_image_vqa import SingleImageVQATool

def test_vqa_tool_initialization():
    tool = SingleImageVQATool()
    assert tool.name == "single_image_vqa"
    assert tool.model_name == "Custom-RSVQA-ResNet18-GRU"

def test_vqa_tool_execution(tmp_path):
    tool = SingleImageVQATool()
    
    # Missing image
    req1 = ToolRequest(tool_name="single_image_vqa", images=[ImageInput(filepath="doesnotexist.tif")], query="Is this urban?")
    res1 = tool.execute(req1)
    assert res1.success == False
    assert "No such file or directory" in res1.error
    
    # Valid image execution
    from PIL import Image
    img_path = tmp_path / "test.tif"
    Image.new("RGB", (256, 256)).save(img_path)
    
    req2 = ToolRequest(tool_name="single_image_vqa", images=[ImageInput(filepath=str(img_path))], query="Is this urban?")
    res2 = tool.execute(req2)
    assert res2.success == True
    assert "answer" in res2.data
    assert "confidence" in res2.data
    assert isinstance(res2.data["confidence"], float)

