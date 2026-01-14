from typing import List, Dict
from time import sleep
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from ragas import evaluate, RunConfig
from ragas.metrics import AnswerRelevancy
from datasets import Dataset
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.evaluation.Judge import Judge
from src.metrics.Faithfulness import Faithfulness
from src.metrics.BertScore import BertScore

load_dotenv()

class LLMJudge(Judge):
    """
    Evaluator for Generator using OpenRouter LLM and RAGAS.
    Evaluates generated answers against ground truth using faithfulness and answer relevancy metrics.

    """

    def __init__(self, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Initialize evaluator with OpenRouter LLM

        Args:
            model_name: OpenRouter model name (e.g., "meta-llama/llama-4-scout-17b-16e-instruct")
        """
        self.model_name = model_name
        API_KEY = os.getenv("GROQ_API_KEY")

        if not API_KEY:
            raise ValueError("OPENROUTER_API_KEY must be set as environment variable")

        # Initialize LangChain ChatOpenAI with OpenRouter
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=API_KEY,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.5
        )

        self.run_config = RunConfig(timeout=500,
                                    max_workers=3)
        self.results = []

        # Initialize local metrics
        self.faithfulness_metric = Faithfulness()
        self.bertscore_metric = BertScore()

        # Change to 1 sample due to LLM limitations
        self.answer_relevancy_metric = AnswerRelevancy()
        self.answer_relevancy_metric.strictness = 1


    def bulk_evaluation(
            self,
            questions: List[str],
            contexts_list: List[List[str]],
            ground_truths: List[str],
            question_ids: List[str],
            generator,
            output_file: str = "evaluation_results.json"
    ) -> Dict:
        """Evaluate entire dataset using RAGAS

        Args:
            questions: List of questions
            contexts_list: List of context lists for each question
            ground_truths: List of ground truth answers
            question_ids: List of question IDs
            generator: Generator instance to generate answers
            output_file: Path to save evaluation results

        Returns:
            Dictionary containing aggregate metrics and individual results
        """

        # Generate answers only
        answers = []
        for question, contexts in tqdm(zip(questions, contexts_list), desc="Generating answers", total=len(questions)):
            if generator.model_name.startswith("gemma"):
                answer = generator.generate(question, contexts)
                answers.append(answer)
                sleep(5) # Pause for Gemma rate limits
            else:
                answer = generator.generate(question, contexts)
                answers.append(answer)

        # Compute local metrics (faithfulness and BertScore)
        faithfulness_scores = []
        bertscore_scores = []

        for answer, contexts, ground_truth in tqdm(zip(answers, contexts_list, ground_truths),
                                                     desc="Computing local metrics",
                                                     total=len(answers)):
            # Compute faithfulness (returns tuple of (is_faithful, confidence))
            _, faithfulness_confidence = self.faithfulness_metric.compute(answer, contexts)
            faithfulness_scores.append(faithfulness_confidence)

            # Compute BertScore for answer correctness
            bertscore_f1 = self.bertscore_metric.compute(answer, ground_truth)
            bertscore_scores.append(bertscore_f1)

        # Create RAGAS dataset for answer relevancy only
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        dataset = Dataset.from_dict(dataset_dict)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Evaluate with RAGAS (answer relevancy only)
        try:
            result = evaluate(
                dataset,
                metrics=[self.answer_relevancy_metric],
                llm=self.llm,
                embeddings=embeddings,
                run_config=self.run_config
            )
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            raise

        # Format individual results
        all_results = []
        for idx in range(len(questions)):
            result_dict = {
                "question_id": question_ids[idx],
                "question": questions[idx],
                "generated_answer": answers[idx],
                "ground_truth": ground_truths[idx],
                "metrics": {
                    "faithfulness": float(faithfulness_scores[idx]),
                    "answer_relevancy": float(result["answer_relevancy"][idx]),
                    "answer_correctness": float(bertscore_scores[idx]),
                }
            }
            all_results.append(result_dict)

        # Calculate aggregate metrics
        aggregate = {
            "faithfulness": np.nanmean(faithfulness_scores),
            "answer_relevancy": np.nanmean(result["answer_relevancy"]),
            "answer_correctness": np.nanmean(bertscore_scores)
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
