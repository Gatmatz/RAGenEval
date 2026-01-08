from typing import List, Dict
from tqdm import tqdm
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

from src.evaluation.Judge import Judge


class NegativeJudge(Judge):
    """
    Evaluator for Generator to check refusal to answer unanswerable questions.
    Evaluates if the generated answer matches the expected negative response.
    Uses semantic similarity to check if answers are similar to expected refusal.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize evaluator with semantic similarity model

        Args:
            similarity_threshold: Minimum cosine similarity score (0-1) to consider
                                 an answer as a correct refusal. Default is 0.75.
        """
        self.results = []
        self.expected_response = "I cannot find the answer in the provided context."
        self.similarity_threshold = similarity_threshold

        # Initialize embeddings model for semantic similarity
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Pre-compute embedding for expected response
        self.expected_embedding = self.embeddings.embed_query(self.expected_response)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using embeddings

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        embedding1 = np.array(self.embeddings.embed_query(text1))
        embedding2 = np.array(self.embeddings.embed_query(text2))

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        return float(similarity)

    def _is_refusal_answer(self, answer: str) -> tuple[bool, float]:
        """
        Check if answer is a refusal (similar to expected negative response)

        Args:
            answer: Generated answer to check

        Returns:
            Tuple of (is_refusal, similarity_score)
        """
        # Compute semantic similarity
        similarity = self._compute_similarity(answer.strip(), self.expected_response)

        # Check if similarity exceeds threshold
        is_refusal = similarity >= self.similarity_threshold

        return is_refusal, similarity

    def evaluate_dataset(
            self,
            dataset,
            qa_ids: List[str],
            generator,
            retriever,
            output_file: str = "negative_evaluation_results.json"
    ) -> Dict:
        """
        Evaluate if system correctly refuses to answer unanswerable questions
        Uses semantic similarity to determine if answers match expected refusal
        """

        all_results = []
        correct_count = 0
        similarity_scores = []

        for qa_id in tqdm(qa_ids, desc="Generating answers"):
            q = dataset.filter(lambda x: x["id"] == qa_id)[0]
            question = q["question"]

            # Get contexts and generate answer
            contexts = retriever.retrieve(q, top_k=5)
            answer = generator.generate(question, contexts)

            # Check if answer is a refusal using semantic similarity
            is_correct, similarity_score = self._is_refusal_answer(answer)
            correct_count += int(is_correct)
            similarity_scores.append(similarity_score)

            result_dict = {
                "question_id": qa_id,
                "answer": answer,
                "is_correct_refusal": is_correct,
                "similarity_score": similarity_score
            }
            all_results.append(result_dict)

        # Calculate aggregate metrics
        total = len(qa_ids)
        accuracy = correct_count / total if total > 0 else 0
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

        aggregate = {
            "accuracy": accuracy,
            "average_similarity": float(avg_similarity),
            "similarity_threshold": self.similarity_threshold,
            "correct_refusals": correct_count,
            "total_questions": total
        }

        # Save results
        output_data = {
            "aggregate_metrics": aggregate,
            "individual_results": all_results
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_file}")
        print(f"\n{'=' * 50}")
        print("AGGREGATE METRICS")
        print(f"{'=' * 50}")
        print(f"{'Accuracy':.<30} {accuracy:.3f}")
        print(f"{'Average Similarity':.<30} {avg_similarity:.3f}")
        print(f"{'Similarity Threshold':.<30} {self.similarity_threshold:.3f}")
        print(f"{'Correct Refusals':.<30} {correct_count}/{total}")

        return output_data