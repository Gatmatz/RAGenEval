from typing import List, Dict, Any

from datasets import load_dataset
import os

from src.retriever.Retriever import Retriever


class PerfectContext(Retriever):
    """
    A frozen retriever that returns context based on supporting facts from HotPot QA.
    """
    def retrieve(self, question, top_k: int = 5) -> List[str]:
        """Retrieve context chunks based on supporting facts"""
        self.supporting_facts = question.get('supporting_facts', {})
        self.context = question.get('context', {})

        contexts = []

        # Extract supporting facts
        supporting_titles = self.supporting_facts.get('title', [])
        supporting_sent_ids = self.supporting_facts.get('sent_id', [])

        # Get context titles and sentences
        context_titles = self.context.get('title', [])
        context_sentences = self.context.get('sentences', [])

        # Build mapping of title to index
        title_to_idx = {title: idx for idx, title in enumerate(context_titles)}

        # Retrieve relevant sentences based on supporting facts
        for title, sent_id in zip(supporting_titles, supporting_sent_ids):
            if title in title_to_idx:
                idx = title_to_idx[title]
                sentences = context_sentences[idx]
                if sent_id < len(sentences):
                    contexts.append(sentences[sent_id])

        return contexts[:top_k]


# Example usage:
if __name__ == "__main__":
    dataset = load_dataset("hotpot_qa", "distractor", token=os.getenv("HF_TOKEN"))
    train_data = dataset['train']
    sample = train_data[0]  # Get a sample question data
    retriever = PerfectContext()
    retrieved_contexts = retriever.retrieve(sample, top_k=5)
    for context in retrieved_contexts:
        print(context)