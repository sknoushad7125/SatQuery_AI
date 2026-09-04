import json
from pathlib import Path

def run():
    out_dir = Path("training_data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for subset in Path("training_data/subsets").glob("*.jsonl"):
        with open(subset) as f:
            count = sum(1 for _ in f)
            stats[subset.name] = count

    with open(out_dir / "phase2_dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(out_dir / "phase2_dataset_statistics.md", "w") as f:
        f.write("# Phase 2 Dataset Statistics\n\n")
        f.write("| Subset File | Record Count |\n")
        f.write("|-------------|--------------|\n")
        for k, v in stats.items():
            f.write(f"| {k} | {v:,} |\n")

        f.write("\n## Notes\n")
        f.write("BigEarthNet grounding was excluded from the initial grounding subset because the axis semantics remain unresolved. The original BigEarthNet records were preserved unchanged for future resolution.\n")

if __name__ == "__main__":
    run()
