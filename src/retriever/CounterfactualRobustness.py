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


if __name__ == "__main__":
    dataset = UniversalDataset.from_json("../../data/en_fact.json")
    question = dataset[0]
    retriever = CounterfactualRobustness()
    retrieved_contexts = retriever.retrieve(question, top_k=5)
    print(f"Question: {question.get('query', question.get('question'))}")
    print(f"Correct answer: {question.get('answer')}")
    print(f"Fake answer: {question.get('fakeanswer')}")
    print(f"\nRetrieved {len(retrieved_contexts)} positive_wrong contexts:")
    for i, context in enumerate(retrieved_contexts, 1):
        print(f"{i}. {context}")
