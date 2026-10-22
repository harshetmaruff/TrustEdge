"""Local natural language inference scoring.

A cross-encoder is run locally over ``(premise, hypothesis)`` pairs. The
premise is a retrieved chunk of evidence, the hypothesis is a declarative
restatement of the question. The model output is a three-way distribution
over entailment, contradiction and neutral.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .logging_utils import get_logger

logger = get_logger(__name__)

LabelScores = Dict[str, float]


class NLIScorer:
    """Score ``(premise, hypothesis)`` pairs with a local cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        """Load the cross-encoder.

        Args:
            model_name: The HuggingFace model id of the cross-encoder.
        """
        self.model_name = model_name
        logger.info("loading NLI model %s", model_name)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        # Trust the checkpoint's own label mapping rather than assuming an order.
        self.labels: List[str] = [
            self._model.config.id2label[index].lower() for index in range(self._model.config.num_labels)
        ]
        logger.info("NLI model %s ready with labels %s", model_name, self.labels)

    def score(self, premise: str, hypothesis: str) -> LabelScores:
        """Score a single pair.

        Args:
            premise: The evidence text.
            hypothesis: The declarative claim to test.

        Returns:
            A mapping from label name to probability.
        """
        return self.score_pairs([(premise, hypothesis)])[0]

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[LabelScores]:
        """Score several pairs in one batch.

        Args:
            pairs: The ``(premise, hypothesis)`` pairs to score.

        Returns:
            One label-to-probability mapping per pair, in the same order.
        """
        if not pairs:
            return []
        import torch

        encoded = self._tokenizer(
            [premise for premise, _ in pairs],
            [hypothesis for _, hypothesis in pairs],
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        results: List[LabelScores] = []
        for row in probabilities:
            results.append({label: float(row[index]) for index, label in enumerate(self.labels)})
        return results
