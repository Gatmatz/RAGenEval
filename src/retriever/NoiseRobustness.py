import os
from typing import List, Dict, Any
import random

from datasets import load_dataset

from src.datasets.UniversalDataset import UniversalDataset
from src.retriever.Retriever import Retriever


class NoiseRobustness(Retriever):
    """Retriever that returns a mix of relevant and irrelevant context based on noise ratio"""

    def __init__(self, noise_ratio: float = 0.5):
        """
        Args:
            noise_ratio: Ratio of irrelevant context (0.0 = all relevant, 1.0 = all irrelevant)
        """
        self.noise_ratio = max(0.0, min(1.0, noise_ratio))  # Clamp between 0 and 1

    def retrieve(self, question: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Retrieve mixed context based on noise ratio"""
        self.supporting_facts = UniversalDataset.get_supporting_facts(question)
        self.context = UniversalDataset.get_context(question)

        # Extract supporting facts
        supporting_titles = self.supporting_facts.get('title', [])
        supporting_sent_ids = self.supporting_facts.get('sent_id', [])

        # Get context titles and sentences
        context_titles = self.context.get('title', [])
        context_sentences = self.context.get('sentences', [])

        # Build mapping and collect relevant/irrelevant contexts
        title_to_idx = {title: idx for idx, title in enumerate(context_titles)}
        supporting_set = set(zip(supporting_titles, supporting_sent_ids))

        relevant_contexts = []
        irrelevant_contexts = []

        # Collect relevant contexts (supporting facts)
        for title, sent_id in zip(supporting_titles, supporting_sent_ids):
            if title in title_to_idx:
                idx = title_to_idx[title]
                sentences = context_sentences[idx]
                if sent_id < len(sentences):
                    relevant_contexts.append(sentences[sent_id])

        # Collect irrelevant contexts
        for idx, title in enumerate(context_titles):
            sentences = context_sentences[idx]
            for sent_id, sentence in enumerate(sentences):
                if (title, sent_id) not in supporting_set:
                    irrelevant_contexts.append(sentence)

        # Calculate how many of each type to include
        num_irrelevant = int(top_k * self.noise_ratio)
        num_relevant = top_k - num_irrelevant

        # Sample contexts
        selected_relevant = random.sample(relevant_contexts, min(num_relevant, len(relevant_contexts)))
        selected_irrelevant = random.sample(irrelevant_contexts, min(num_irrelevant, len(irrelevant_contexts)))

        # Combine and shuffle
        mixed_contexts = selected_relevant + selected_irrelevant
        random.shuffle(mixed_contexts)

        return mixed_contexts[:top_k]

if __name__ == "__main__":
    dataset = UniversalDataset.from_huggingface(
        dataset_path="hotpot_qa",
        dataset_name="distractor",
        split="train"
    )
    question = dataset.get_question_by_id("5a7a06935542990198eaf050")
    retriever = NoiseRobustness()
    retrieved_contexts = retriever.retrieve(question, top_k=5)
    for context in retrieved_contexts:
        print(context)