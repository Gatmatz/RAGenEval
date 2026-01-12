from typing import List, Dict, Any
import random
from src.datasets.UniversalDataset import UniversalDataset
from src.retriever.Retriever import Retriever


class CounterfactualRobustness(Retriever):
    """Retriever that returns positive_wrong context (factually similar but with wrong answer)"""

    def retrieve(self, question: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Retrieve positive_wrong context chunks, randomly selecting k if more than top_k available"""
        positive_wrong_contexts = UniversalDataset.get_positive_wrong_contexts(question)

        # If there are more contexts than top_k, randomly select k
        if len(positive_wrong_contexts) > top_k:
            return random.sample(positive_wrong_contexts, top_k)

        return positive_wrong_contexts
