from typing import List, Dict
from tqdm import tqdm
import json
from time import sleep
from pathlib import Path
import numpy as np

from src.evaluation.Judge import Judge
from src.metrics.BertScore import BertScore


class CounterfactualJudge(Judge):
    """
    Evaluator for Generator to test counterfactual robustness.
    Evaluates if the system can detect factual errors in provided documents.
    Uses BertScore to check if answers correctly identify factual inconsistencies.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        Initialize evaluator with BertScore metric

        Args:
            similarity_threshold: Minimum BertScore F1 score (0-1) to consider
                                 an answer as correctly detecting factual errors. Default is 0.75.
        """
        self.results = []
        self.expected_response = "There are factual errors in the provided context"
        self.similarity_threshold = similarity_threshold

        # Initialize BertScore metric
        self.bert_score = BertScore()

    def _is_error_detection(self, answer: str) -> tuple[bool, float]:
        """
        Check if answer correctly identifies factual errors (similar to expected error detection response)

        Args:
            answer: Generated answer to check

        Returns:
            Tuple of (is_correct_detection, bert_f1_score)
        """
        # Compute BertScore
        bert_f1 = self.bert_score.compute(answer.strip(), self.expected_response)

        # Check if F1 score exceeds threshold
        is_correct_detection = bert_f1 >= self.similarity_threshold

        return is_correct_detection, float(bert_f1)

    def bulk_evaluation(
            self,
            questions: List[str],
            contexts_list: List[List[str]],
            question_ids: List[str],
            generator,
            true_answers: List[str] = None,
            fake_answers: List[str] = None,
            output_file: str = "counterfactual_evaluation_results.json"
    ) -> Dict:
        """
        Evaluate if system correctly detects factual errors in provided documents
        Uses BertScore to determine if answers match expected error detection

        Args:
            questions: List of questions
            contexts_list: List of context lists for each question (containing factual errors)
            question_ids: List of question IDs
            generator: Generator instance to generate answers
            true_answers: List of true answers for similarity comparison (optional)
            fake_answers: List of fake answers for similarity comparison (optional)
            output_file: Path to save evaluation results

        Returns:
            Dictionary containing aggregate metrics and individual results
        """

        all_results = []
        correct_count = 0
        bert_f1_scores = []
        true_answer_bert_scores = []
        fake_answer_bert_scores = []

        # Prepare iterables for true and fake answers
        if true_answers is None:
            true_answers = [None] * len(questions)
        if fake_answers is None:
            fake_answers = [None] * len(questions)

        for qa_id, question, contexts, true_answer, fake_answer in tqdm(
                zip(question_ids, questions, contexts_list, true_answers, fake_answers),
                desc="Generating answers",
                total=len(questions)):
            # Generate answer
            if generator.model_name.startswith("gemma"):
                answer = generator.generate(question, contexts)
                sleep(5)  # Pause for  rate limits
            else:
                answer = generator.generate(question, contexts)

            # Check if answer correctly detects errors using BertScore
            is_correct, bert_f1 = self._is_error_detection(answer)
            correct_count += int(is_correct)
            bert_f1_scores.append(bert_f1)

            # Compute BertScore with true answer if provided
            true_answer_bert_score = None
            if true_answer is not None:
                true_answer_bert_score = self.bert_score.compute(answer, str(true_answer))
                true_answer_bert_scores.append(true_answer_bert_score)

            # Compute BertScore with fake answer if provided
            fake_answer_bert_score = None
            if fake_answer is not None:
                fake_answer_bert_score = self.bert_score.compute(answer, str(fake_answer))
                fake_answer_bert_scores.append(fake_answer_bert_score)

            result_dict = {
                "question_id": qa_id,
                "answer": answer,
                "is_correct_detection": is_correct,
                "bert_f1_score": bert_f1,
                "true_answer": true_answer,
                "true_answer_bert_score": true_answer_bert_score,
                "fake_answer": fake_answer,
                "fake_answer_bert_score": fake_answer_bert_score
            }
            all_results.append(result_dict)

        # Calculate aggregate metrics
        total = len(question_ids)
        accuracy = correct_count / total if total > 0 else 0
        avg_bert_f1 = np.mean(bert_f1_scores) if bert_f1_scores else 0
        avg_true_answer_bert_score = np.mean(true_answer_bert_scores) if true_answer_bert_scores else None
        avg_fake_answer_bert_score = np.mean(fake_answer_bert_scores) if fake_answer_bert_scores else None

        aggregate = {
            "accuracy": accuracy,
            "average_bert_f1": float(avg_bert_f1),
            "similarity_threshold": self.similarity_threshold,
            "correct_detections": correct_count,
            "total_questions": total,
            "average_true_answer_bert_score": float(
                avg_true_answer_bert_score) if avg_true_answer_bert_score is not None else None,
            "average_fake_answer_bert_score": float(
                avg_fake_answer_bert_score) if avg_fake_answer_bert_score is not None else None
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