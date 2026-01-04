from typing import List, Dict
from tqdm import tqdm
import json
from pathlib import Path

from src.evaluation.Judge import Judge


class NegativeJudge(Judge):
    """
    Evaluator for Generator to check refusal to answer unanswerable questions.
    Evaluates if the generated answer matches the expected negative response.
    """

    def __init__(self):
        """Initialize evaluator"""
        self.results = []
        self.expected_response = "I cannot find the answer in the provided context."

    def evaluate_dataset(
            self,
            dataset,
            qa_ids: List[str],
            generator,
            retriever,
            output_file: str = "negative_evaluation_results.json"
    ) -> Dict:
        """Evaluate if system correctly refuses to answer unanswerable questions"""

        all_results = []
        correct_count = 0

        for qa_id in tqdm(qa_ids, desc="Generating answers"):
            q = dataset.filter(lambda x: x["id"] == qa_id)[0]
            question = q["question"]

            # Get contexts and generate answer
            contexts = retriever.retrieve(q, top_k=5)
            answer = generator.generate(question, contexts)

            # Check if answer matches expected negative response
            is_correct = answer.strip() == self.expected_response
            correct_count += int(is_correct)

            result_dict = {
                "question_id": qa_id,
                "answer": answer,
                "is_correct_refusal": is_correct
            }
            all_results.append(result_dict)

        # Calculate aggregate metrics
        total = len(qa_ids)
        accuracy = correct_count / total if total > 0 else 0

        aggregate = {
            "accuracy": accuracy,
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
        print(f"{'Correct Refusals':.<30} {correct_count}/{total}")

        return output_data