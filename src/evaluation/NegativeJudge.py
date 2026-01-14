from typing import List, Dict
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np
from time import sleep
from src.evaluation.Judge import Judge
from src.metrics.BertScore import BertScore


class NegativeJudge(Judge):
    """
    Evaluator for Generator to check refusal to answer unanswerable questions.
    Evaluates if the generated answer matches the expected negative response.
    Uses BertScore to check if answers are similar to expected refusal.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize evaluator with BertScore metric

        Args:
            similarity_threshold: Minimum BertScore F1 score (0-1) to consider
                                 an answer as a correct refusal. Default is 0.75.
        """
        self.results = []
        self.expected_response = "I don't know."
        self.similarity_threshold = similarity_threshold

        # Initialize BertScore metric
        self.bert_score = BertScore()


    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute BertScore F1 between two texts

        Args:
            text1: Generated text
            text2: Reference text (ground truth)

        Returns:
            BertScore F1 score between 0 and 1
        """
        return self.bert_score.compute(text1, text2)

    def _is_refusal_answer(self, answer: str) -> tuple[bool, float]:
        """
        Check if answer is a refusal (similar to expected negative response using BertScore)

        Args:
            answer: Generated answer to check

        Returns:
            Tuple of (is_refusal, similarity_score)
        """
        # Compute BertScore similarity
        similarity = self._compute_similarity(answer.strip(), self.expected_response)

        # Check if similarity exceeds threshold
        is_refusal = similarity >= self.similarity_threshold

        return is_refusal, similarity

    def bulk_evaluation(
            self,
            questions: List[str],
            contexts_list: List[List[str]],
            question_ids: List[str],
            generator,
            output_file: str = "negative_evaluation_results.json"
    ) -> Dict:
        """
        Evaluate if system correctly refuses to answer unanswerable questions
        Uses BertScore to determine if answers match expected refusal

        Args:
            questions: List of questions
            contexts_list: List of context lists for each question
            question_ids: List of question IDs
            generator: Generator instance to generate answers
            output_file: Path to save evaluation results

        Returns:
            Dictionary containing aggregate metrics and individual results
        """

        all_results = []
        correct_count = 0
        similarity_scores = []

        for qa_id, question, contexts in tqdm(zip(question_ids, questions, contexts_list),
                                               desc="Generating answers",
                                               total=len(questions)):
            # Generate answer
            if generator.model_name.startswith("gemma"):
                answer = generator.generate(question, contexts)
                sleep(5) # Pause for Gemma rate limits
            else:
                answer = generator.generate(question, contexts)

            # Check if answer is a refusal using BertScore
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
        total = len(question_ids)
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

        return output_data