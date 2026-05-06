
"""
Evaluation client. Fetches daily batches from the server and submits predictions.

This shows candidates the expected workflow:
  1. Fetch the day's queries from the server
  2. Run your model on each query
  3. Submit predictions back to the server
  4. Get your score

Usage:
    uv run python evaluate.py                        # scan ./data/days for day files
    uv run python evaluate.py --data-dir ./my_data   # use a custom data directory
"""
import argparse
import json
import re
from pathlib import Path
import requests

SERVER_URL = "http://localhost:5117"


def discover_days(data_dir: Path) -> list[int]:
    """Scan data_dir for day_XX.jsonl files and return sorted day numbers."""
    days = []
    for p in data_dir.iterdir():
        m = re.match(r"day_(\d+)\.jsonl$", p.name)
        if m:
            days.append(int(m.group(1)))
    return sorted(days)


def fetch_queries(day: int) -> list[dict]:
    r = requests.get(f"{SERVER_URL}/day/{day}")
    r.raise_for_status()
    return r.json()["queries"]


def submit_predictions(day: int, predictions: list[dict]) -> dict:
    r = requests.post(
        f"{SERVER_URL}/day/{day}/submit",
        json={"predictions": predictions},
    )
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="Evaluate your model against daily batches")
    parser.add_argument("--data-dir", type=Path, default=Path("data/days"),
                        help="Directory containing day_XX.jsonl files (default: data/days)")
    args = parser.parse_args()

    days = discover_days(args.data_dir)
    if not days:
        print(f"No day_*.jsonl files found in {args.data_dir}")
        return

    # ── Load the baseline model ──
    # Replace this section with your own model.
    from sentence_transformers import SentenceTransformer
    from baseline import load_actions, build_action_index, predict, MODEL_NAME

    model = SentenceTransformer(MODEL_NAME)
    actions = load_actions()
    action_embs, action_ids = build_action_index(model, actions)

    total_correct = 0
    total_queries = 0
    results = {}

    for day in days:
        try:
            queries = fetch_queries(day)
        except requests.HTTPError:
            continue

        predictions = []
        for q in queries:
            predicted_action = predict(model, action_embs, action_ids, q["query"])
            predictions.append({"id": q["id"], "action_id": predicted_action})

        result = submit_predictions(day, predictions)
        results[day] = result
        total_correct += result["correct"]
        total_queries += result["total"]

        print(f"Day {day:2d}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        for cat, stats in result["per_category"].items():
            print(f"       {cat}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    if len(results) > 1:
        overall = total_correct / total_queries if total_queries else 0
        print(f"\nOverall: {overall:.1%} ({total_correct}/{total_queries}) across {len(results)} days")


if __name__ == "__main__":
    main()
