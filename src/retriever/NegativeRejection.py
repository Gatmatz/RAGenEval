from typing import List, Dict, Any
from src.datasets.UniversalDataset import UniversalDataset
from src.retriever.Retriever import Retriever


class NegativeRejection(Retriever):
    """Retriever that returns irrelevant context by excluding supporting facts from HotPot QA"""

    def retrieve(self, question: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Retrieve irrelevant context chunks by excluding supporting facts"""
        self.supporting_facts = UniversalDataset.get_supporting_facts(question)
        self.context = UniversalDataset.get_context(question)

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
    dataset = UniversalDataset.from_huggingface(
        dataset_path="hotpot_qa",
        dataset_name="distractor",
        split="train"
    )
    question = dataset.get_question_by_id("5a7a06935542990198eaf050")
    retriever = NegativeRejection()
    retrieved_contexts = retriever.retrieve(question, top_k=5)
    for context in retrieved_contexts:
        print(context)
