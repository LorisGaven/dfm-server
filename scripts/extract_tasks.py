"""Extract unique tasks from val.jsonl and write tasks.json."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Extract unique tasks from JSONL")
    parser.add_argument("--input", required=True, help="Path to val.jsonl")
    parser.add_argument("--output", default="tasks.json", help="Output tasks.json path")
    args = parser.parse_args()

    tasks = set()
    with open(args.input) as f:
        for line in f:
            record = json.loads(line)
            tasks.update(record["tasks"])

    tasks_list = sorted(tasks)
    with open(args.output, "w") as f:
        json.dump(tasks_list, f)

    print(f"Extracted {len(tasks_list)} unique tasks → {args.output}")


if __name__ == "__main__":
    main()
