from sentence_transformers import CrossEncoder
import torch

from src.metrics.Metric import Metric


class Faithfulness(Metric):
    def __init__(self):
        """
        Initialize the Faithfulness metric with a pre-trained CrossEncoder model.

        """
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-small', device=self.device)

    def compute(self, answer:str, context:list[str]) -> tuple[bool, float]:
        """
        Compute faithfulness by checking if any context sentence entails the answer.

        Args:
            answer: The generated answer to evaluate
            context: List of context sentences

        Returns:
            A tuple of (is_faithful, max_confidence) where:
            - is_faithful: True if any context sentence entails the answer
            - max_confidence: The highest entailment confidence score
        """
        # Create pairs of (context_sentence, answer) for each sentence
        pairs = [(sent, answer) for sent in context]

        # Get scores (Contradiction, Entailment, Neutral)
        scores = self.model.predict(pairs)

        # Label mapping for this specific model
        label_mapping = ['contradiction', 'entailment', 'neutral']

        # Check each sentence for entailment
        max_entailment_confidence = 0.0
        is_faithful = False

        for score_array in scores:
            # Get the predicted label for this sentence
            result_idx = score_array.argmax()
            label = label_mapping[result_idx]

            # Calculate confidence using softmax
            confidence = torch.nn.functional.softmax(torch.tensor(score_array), dim=0)[result_idx].item()

            # If this sentence entails the answer, mark as faithful
            if label == 'entailment':
                is_faithful = True
                max_entailment_confidence = max(max_entailment_confidence, confidence)

        return is_faithful, max_entailment_confidence
