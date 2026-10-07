from __future__ import annotations

import math

import torch
from torch.nn import functional as functional

from src.bktree import BKTree
from src.config import BLANK_IDX, NUM_CLASSES
from src.ctc_probability import ctc_log_probability
from src.dataset import CHAR_2_LABEL
from src.lexicon import LexiconDecoder
from src.metrics import levenshtein_distance


def test_ctc_forward_probability_matches_pytorch() -> None:
    log_probs = torch.randn(9, NUM_CLASSES, dtype=torch.float64).log_softmax(1)
    labels = [CHAR_2_LABEL[char] for char in "book"]
    ours = ctc_log_probability(log_probs, labels)
    torch_loss = functional.ctc_loss(
        log_probs[:, None, :],
        torch.tensor(labels),
        torch.tensor([len(log_probs)]),
        torch.tensor([len(labels)]),
        blank=BLANK_IDX,
        reduction="sum",
        zero_infinity=False,
    )
    assert math.isclose(ours, -float(torch_loss), rel_tol=1e-10, abs_tol=1e-10)


def test_bktree_matches_brute_force_radius_search() -> None:
    words = ["book", "books", "back", "cake", "boo", "cape"]
    tree = BKTree(words)
    expected = sorted(
        (levenshtein_distance("book", word), word)
        for word in words
        if levenshtein_distance("book", word) <= 1
    )
    assert tree.query("book", 1) == expected


def test_lexicon_decoder_uses_full_ctc_probability() -> None:
    path = ["c", None, "a", None, "t"]
    logits = torch.full((len(path), NUM_CLASSES), -12.0)
    for time_index, character in enumerate(path):
        logits[time_index, BLANK_IDX if character is None else CHAR_2_LABEL[character]] = 12.0
    log_probs = logits.log_softmax(1)
    result = LexiconDecoder(["car", "cat", "dog"]).decode(log_probs, delta=None)
    assert result.text == "cat"
    assert result.greedy_text == "cat"
    assert result.candidate_count == 3


def test_empty_bktree_neighborhood_falls_back_to_greedy() -> None:
    path = ["c", None, "a", None, "t"]
    logits = torch.full((len(path), NUM_CLASSES), -12.0)
    for time_index, character in enumerate(path):
        logits[time_index, BLANK_IDX if character is None else CHAR_2_LABEL[character]] = 12.0
    result = LexiconDecoder(["dog"]).decode(logits.log_softmax(1), delta=0)
    assert result.text == "cat"
    assert result.candidate_count == 0
