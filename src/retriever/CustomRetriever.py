import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from src.datasets.UniversalDataset import UniversalDataset

class CustomRetriever:
    """
    Dense FAISS-based retriever for HotpotQA.
    """

    def __init__(self,model_name: str = "all-MiniLM-L6-v2", normalize: bool = True, chunk_size: int = 150, chunk_overlap: int = 40):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.index = None
        self.passages: List[str] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
        return embeddings.astype("float32")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks of chunk_size words.
        """
        return self.text_splitter.split_text(text=text)

    def build_index(self, dataset: UniversalDataset):
        """
        Build FAISS index from HotpotQA dataset.
        """
        passages = []

        for item in dataset.dataset:
            titles = item["context"]["title"]
            sentences_list = item["context"]["sentences"]

            for title, sentences in zip(titles, sentences_list):
                paragraph = " ".join(sentences)
                for chunk in self.chunk_text(paragraph):
                    passages.append(f"{title}: {chunk}")

        self.passages = passages
        embeddings = self._embed(passages)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"[Retriever] Indexed {len(passages)} passages")

    def build_index_singlequestion(self, qa_item):
        """
        Build FAISS index from the context of a single question.
        """
        passages = []

        titles = qa_item["context"]["title"]
        sentences_list = qa_item["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            paragraph = " ".join(sentences)
            for chunk in self.chunk_text(paragraph):
                passages.append(f"{title}: {chunk}")

        self.passages = passages
        embeddings = self._embed(passages)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"[Retriever] Indexed {len(passages)} passages for the question.")


    def retrieve(self, qa_item, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k diverse passages based on title.
        """
        self.build_index_singlequestion(qa_item)
        query_emb = self._embed([qa_item["question"]])
        scores, indices = self.index.search(query_emb, top_k * 3)

        seen_titles = set()
        results = []

        for idx in indices[0]:
            passage = self.passages[idx]
            title = passage.split(":")[0]

            if title not in seen_titles:
                seen_titles.add(title)
                results.append(passage)

            if len(results) == top_k:
                break
        return results