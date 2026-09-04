import json
from pathlib import Path

def run():
    vrsbench_vqa_val = Path('datasets/vrsbench/VRSBench_EVAL_vqa.json')
    if vrsbench_vqa_val.exists():
        with open(vrsbench_vqa_val) as f:
            vqa_orig = json.load(f)
        vqa_map = {f"{item['image_id']}_{item['question']}": item["ground_truth"] for item in vqa_orig}

        recs = []
        with open('training_data/subsets/vqa_val.jsonl') as f:
            for line in f:
                r = json.loads(line)
                if r['dataset'] == 'vrsbench' and r['answer'] is None:
                    # try to fix
                    img_id = r['image']['member'].split('/')[-1] if isinstance(r['image'], dict) else r['image']
                    k = f"{img_id}_{r['question']}"
                    r['answer'] = vqa_map.get(k)
                recs.append(r)

        with open('training_data/subsets/vqa_val.jsonl', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    run()
