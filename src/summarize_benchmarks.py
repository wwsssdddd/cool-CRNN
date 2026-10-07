"""Собирает summary.json разных benchmark-наборов в CSV и Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="DATASET=/path/to/summary.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for specification in args.inputs:
        if "=" not in specification:
            raise ValueError(f"Ожидается DATASET=PATH, получено {specification!r}")
        dataset, raw_path = specification.split("=", maxsplit=1)
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        for mode, values in payload["modes"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "word_accuracy_percent": values["word_accuracy"] * 100,
                    "cer": values["cer"],
                    "mean_time_ms": values["mean_time_ms"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = ["| dataset | mode | accuracy, % | CER | mean ms |", "|---|---|---:|---:|---:|"]
    markdown.extend(
        f"| {row['dataset']} | {row['mode']} | {row['word_accuracy_percent']:.2f} | "
        f"{row['cer']:.4f} | {row['mean_time_ms']:.2f} |"
        for row in rows
    )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Сохранено: {args.output} и {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
