from typing import List, Dict

import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.evaluation.Judge import Judge

load_dotenv()

class LLMJudge(Judge):
    """
    Evaluator for Generator using Groq LLM and RAGAS.
    Evaluates generated answers against ground truth using faithfulness and answer relevancy metrics.

    """

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        """
        Initialize evaluator with Groq LLM

        Args:
            model: Groq model name
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")

        self.llm = ChatGroq(model=model, temperature=0.1, groq_api_key=api_key)
        self.results = []

    def evaluate_dataset(
            self,
            dataset,
            qa_ids: List[str],
            generator,
            retriever,
            output_file: str = "evaluation_results.json"
    ) -> Dict:
        """Evaluate entire dataset using RAGAS"""

        # Prepare data for RAGAS
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        question_ids = []

        for qa_id in tqdm(qa_ids, desc="Generating answers"):
            q = dataset.filter(lambda x: x["id"] == qa_id)[0]
            question = q["question"]
            ground_truth = q["answer"]

            # Get contexts and generate answer
            contexts = retriever.retrieve(q, top_k=5)
            answer = generator.generate(question, contexts)

            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(ground_truth)
            question_ids.append(qa_id)

        # Create RAGAS dataset
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(dataset_dict)

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        # Evaluate with RAGAS
        print("\nEvaluating with RAGAS...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.llm,
            embeddings = embeddings
        )

        # Format individual results
        all_results = []
        for idx in range(len(questions)):
            result_dict = {
                "question_id": question_ids[idx],
                "answer": answers[idx],
                "metrics": {
                    "faithfulness": result["faithfulness"][idx],
                    "answer_relevancy": result["answer_relevancy"][idx]
                }
            }
            all_results.append(result_dict)

        # Calculate aggregate metrics
        aggregate = {
            "faithfulness": np.mean(result["faithfulness"]),
            "answer_relevancy": np.mean(result["answer_relevancy"])
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
        for metric, value in aggregate.items():
            print(f"{metric:.<30} {value:.3f}")

        return output_data
