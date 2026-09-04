import sys; sys.path.append(".")

from training_data.dataloaders import (
    create_vqa_dataloader,
    create_caption_dataloader,
    create_grounding_dataloader,
    create_change_vqa_dataloader
)
import traceback

def test_vqa():
    print("VQA")
    try:
        dl = create_vqa_dataloader("training_data/subsets/vqa_val.jsonl", batch_size=2)
        batch = next(iter(dl))
        print(f"records: {len(dl.dataset)}")
        print(f"batch images: {batch['images'].shape}")
        print(f"batch questions: {batch['questions']}")
        print(f"batch answers: {batch['answers']}")
        print("PASS\n")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

def test_caption():
    print("CAPTIONING")
    try:
        dl = create_caption_dataloader("training_data/subsets/caption_val.jsonl", batch_size=2)
        batch = next(iter(dl))
        print(f"records: {len(dl.dataset)}")
        print(f"batch images: {batch['images'].shape}")
        print(f"batch captions: {batch['captions']}")
        print("PASS\n")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

def test_grounding():
    print("GROUNDING")
    try:
        dl = create_grounding_dataloader("training_data/subsets/grounding_val.jsonl", batch_size=2)
        batch = next(iter(dl))
        print(f"records: {len(dl.dataset)}")
        print(f"batch images: {batch['images'].shape}")
        print(f"batch queries: {batch['queries']}")
        print(f"batch boxes: {batch['bboxes']}")
        print("PASS\n")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

def test_change_vqa():
    print("CHANGE VQA")
    try:
        dl = create_change_vqa_dataloader("training_data/subsets/change_vqa_val.jsonl", batch_size=2)
        batch = next(iter(dl))
        print(f"records: {len(dl.dataset)}")
        print(f"before images: {batch['before_images'].shape}")
        print(f"after images: {batch['after_images'].shape}")
        print(f"questions: {batch['questions']}")
        print(f"answers: {batch['answers']}")
        print("PASS\n")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    vqa = test_vqa()
    cap = test_caption()
    grd = test_grounding()
    chg = test_change_vqa()

    if vqa and cap and grd and chg:
        print("ALL DRY RUN TESTS PASSED")
    else:
        print("DRY RUN FAILED")
