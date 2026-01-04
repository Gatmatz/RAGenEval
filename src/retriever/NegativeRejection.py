import os
from typing import List, Dict, Any

from datasets import load_dataset


class NegativeRejection:
    """Retriever that returns irrelevant context by excluding supporting facts from HotPot QA"""

    def retrieve(self, question: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Retrieve irrelevant context chunks by excluding supporting facts"""
        self.supporting_facts = question.get('supporting_facts', {})
        self.context = question.get('context', {})

        irrelevant_contexts = []

        # Extract supporting facts
        supporting_titles = self.supporting_facts.get('title', [])
        supporting_sent_ids = self.supporting_facts.get('sent_id', [])

        # Create set of supporting fact identifiers
        supporting_set = set(zip(supporting_titles, supporting_sent_ids))

        # Get context titles and sentences
        context_titles = self.context.get('title', [])
        context_sentences = self.context.get('sentences', [])

        # Retrieve all sentences that are NOT in supporting facts
        for idx, title in enumerate(context_titles):
            sentences = context_sentences[idx]
            for sent_id, sentence in enumerate(sentences):
                if (title, sent_id) not in supporting_set:
                    irrelevant_contexts.append(sentence)

        return irrelevant_contexts[:top_k]

if __name__ == "__main__":
    dataset = load_dataset("hotpot_qa", "distractor", token=os.getenv("HF_TOKEN"))
    train_data = dataset['train']
    sample = train_data[0]
    retriever = NegativeRejection()
    retrieved_contexts = retriever.retrieve(sample, top_k=5)
    for context in retrieved_contexts:
        print(context)
