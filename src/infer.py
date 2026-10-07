"""Командная строка для распознавания одного изображения."""

from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_model
from .config import DEVICE
from .dataset import read_image
from .lexicon import LexiconDecoder, greedy_decode_log_probs, load_lexicon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRNN inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--lexicon", help="TXT, одно слово на строку")
    parser.add_argument(
        "--delta",
        type=int,
        help="BK-tree радиус; без него выполняется точный полный перебор словаря",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.checkpoint, args.device)
    image = read_image(args.image)
    width = torch.tensor([image.shape[-1]], dtype=torch.long)
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(args.device), width)
        valid_steps = int(model.output_lengths(width)[0])
        log_probs = logits[:valid_steps, 0].float().log_softmax(1).cpu()

    result: dict[str, object] = {
        "image": args.image,
        "greedy": greedy_decode_log_probs(log_probs),
        "time_steps": valid_steps,
    }
    if args.lexicon:
        decoded = LexiconDecoder(load_lexicon(args.lexicon)).decode(
            log_probs,
            delta=args.delta,
        )
        result["lexicon"] = decoded.text
        result["candidate_count"] = decoded.candidate_count
        result["lexicon_time_ms"] = decoded.search_time_ms
        result["delta"] = args.delta
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
