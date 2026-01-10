from typing import List, Dict

import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, AnswerRelevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.evaluation.Judge import Judge

load_dotenv()

class LLMJudge(Judge):
    """
    Evaluator for Generator using OpenRouter LLM and RAGAS.
    Evaluates generated answers against ground truth using faithfulness and answer relevancy metrics.

    """

    def __init__(self, model_name: str = "meta-llama/llama-3.1-405b-instruct:free"):
        """
        Initialize evaluator with OpenRouter LLM

        Args:
            model_name: OpenRouter model name
        """
        self.model_name = model_name
        API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not API_KEY:
            raise ValueError("OPENROUTER_API_KEY must be set as environment variable")

        # Initialize LangChain ChatOpenAI with OpenRouter
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1
        )

        self.results = []

        # Change to 1 sample due to LLM limitations
        self.answer_relevancy_metric = AnswerRelevancy()
        self.answer_relevancy_metric.strictness = 1


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
        # dataset_dict_path = output_file
        # with open(dataset_dict_path, 'w') as f:
        #     json.dump(dataset_dict, f, indent=2)

        dataset = Dataset.from_dict(dataset_dict)

        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        #
        # # Evaluate with RAGAS
        # print("\nEvaluating with RAGAS...")
        # result = evaluate(
        #     dataset,
        #     metrics=[faithfulness, answer_correctness, self.answer_relevancy_metric],
        #     llm=self.llm,
        #     embeddings = embeddings
        # )
        #
        # # Format individual results
        # all_results = []
        # for idx in range(len(questions)):
        #     result_dict = {
        #         "question_id": question_ids[idx],
        #         "metrics": {
        #             "faithfulness": float(result["faithfulness"][idx]),
        #             "answer_relevancy": float(result["answer_relevancy"][idx]),
        #             "answer_correctness": float(result["answer_correctness"][idx]),
        #         }
        #     }
        #     all_results.append(result_dict)
        #
        # # Calculate aggregate metrics
        # aggregate = {
        #     "faithfulness": np.nanmean(result["faithfulness"]),
        #     "answer_relevancy": np.nanmean(result["answer_relevancy"]),
        #     "answer_correctness": np.nanmean(result["answer_correctness"])
        #
        # }
        #
        # # Save results
        # output_data = {
        #     "aggregate_metrics": aggregate,
        #     "individual_results": all_results
        # }
        #
        # Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        # with open(output_file, 'w') as f:
        #     json.dump(output_data, f, indent=2)
        #
        # print(f"\nResults saved to {output_file}")
        # print(f"\n{'=' * 50}")
        # print("AGGREGATE METRICS")
        # print(f"{'=' * 50}")
        # for metric, value in aggregate.items():
        #     print(f"{metric:.<30} {value:.3f}")
        #
        # return output_data
