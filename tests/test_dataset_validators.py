import os
import pytest
from scripts.dataset_tools.validate_levir import validate_levir
from scripts.dataset_tools.validate_sen12ms import validate_sen12ms
from scripts.dataset_tools.validate_rsvqa import validate_rsvqa
from scripts.dataset_tools.validate_bigearthnet import validate_bigearthnet

def test_levir_absent():
    res = validate_levir("nonexistent_dir")
    assert res["status"] == "ABSENT"

def test_sen12ms_absent():
    res = validate_sen12ms("nonexistent_dir")
    assert res["status"] == "ABSENT"
    
def test_rsvqa_absent():
    res = validate_rsvqa("nonexistent_dir")
    assert res["status"] == "ABSENT"

def test_bigearthnet_absent():
    res = validate_bigearthnet("nonexistent_dir")
    assert res["status"] == "ABSENT"
