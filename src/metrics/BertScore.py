import torch
import warnings
import logging
from bert_score import score

from src.metrics.Metric import Metric

# Silence transformers warnings about unused weights
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint")
logging.getLogger("transformers").setLevel(logging.ERROR)


class BertScore(Metric):
    def __init__(self):
        """
        Initialize the BertScore metric.
        """
        # Use MPS if available, otherwise fall back to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def compute(self, answer:str, ground_truth:str) -> float:
        """
        Compute the BertScore F1 between the answer and ground truth.
        :param answer: The generated answer from an LLM.
        :param ground_truth: The reference ground truth answer.
        :return: F1 score as a float.
        """
        P, R, F1 = score([answer], [ground_truth], lang="en", device=self.device)
        return F1.item()