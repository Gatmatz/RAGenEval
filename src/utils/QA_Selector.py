from src.datasets.UniversalDataset import UniversalDataset
from typing import Optional


class QA_Selector:
    """
    Utility class to select questions from datasets (HuggingFace or local JSON).

    Usage:
        # For HuggingFace datasets (default behavior)
        selector = QA_Selector(num_questions=100)

        # For local JSON files
        selector = QA_Selector.from_json(num_questions=100, json_path="data/en_fact.json")

        # For custom HuggingFace datasets
        selector = QA_Selector.from_huggingface(
            num_questions=100,
            dataset_path="squad",
            dataset_name=None,
            split="train"
        )
    """
    def __init__(self, num_questions: int, dataset: Optional[UniversalDataset] = None, seed: int = 42):
        """
        Initialize QA_Selector.

        Args:
            num_questions: Number of questions to select
            dataset: UniversalDataset instance (if None, defaults to HotpotQA)
            seed: Random seed for shuffling (default: 42)
        """
        self.num_questions = num_questions
        self.seed = seed

        # If no dataset provided, use default HotpotQA for backward compatibility
        if dataset is None:
            self.dataset = UniversalDataset.from_huggingface(
                dataset_path="hotpot_qa",
                dataset_name="distractor",
                split="train"
            )
        else:
            self.dataset = dataset

        self._dataset_list = None  # Cache for the dataset

    @classmethod
    def from_huggingface(cls, num_questions: int,
                        dataset_path: str = "hotpot_qa",
                        dataset_name: Optional[str] = "distractor",
                        split: str = "train",
                        seed: int = 42) -> 'QA_Selector':
        """
        Create QA_Selector from a HuggingFace dataset.

        Args:
            num_questions: Number of questions to select
            dataset_path: Path to the HuggingFace dataset (default: "hotpot_qa")
            dataset_name: Name/config of the dataset (default: "distractor")
            split: Dataset split to use (default: "train")
            seed: Random seed for shuffling (default: 42)
        """
        dataset = UniversalDataset.from_huggingface(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split=split
        )
        return cls(num_questions, dataset, seed)

    @classmethod
    def from_json(cls, num_questions: int,
                  json_path: str,
                  id_field: str = "id",
                  question_field: str = "question",
                  answer_field: str = "answer",
                  seed: int = 42) -> 'QA_Selector':
        """
        Create QA_Selector from a local JSON file.

        Args:
            num_questions: Number of questions to select
            json_path: Path to the JSON file
            id_field: Name of the ID field (default: "id")
            question_field: Name of the question field (default: "question")
            answer_field: Name of the answer field (default: "answer")
            seed: Random seed for shuffling (default: 42)
        """
        dataset = UniversalDataset.from_json(
            json_path=json_path,
            id_field=id_field,
            question_field=question_field,
            answer_field=answer_field
        )
        return cls(num_questions, dataset, seed)

    def get_question_ids(self):
        """
        Returns a list of question IDs from the dataset.
        """
        import random

        if self.dataset.source_type == 'huggingface':
            # Use manual shuffling for consistency across all dataset types
            random.seed(self.seed)

            # Create a list of indices and shuffle them
            indices = list(range(len(self.dataset.dataset)))
            random.shuffle(indices)

            # Select the first num_questions
            selected_indices = indices[:min(self.num_questions, len(indices))]

            # Get questions by index
            return [self.dataset.dataset[i]['id'] for i in selected_indices]
        else:
            # For JSON datasets, shuffle manually
            random.seed(self.seed)

            # Get ID field name
            id_field = self.dataset._field_mappings.get('id', 'id')

            # Create a copy of indices and shuffle
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)

            # Select the first num_questions
            selected_indices = indices[:min(self.num_questions, len(indices))]

            return [self.dataset[i][id_field] for i in selected_indices]

    def get_dataset(self):
        """
        Returns the underlying dataset.
        For HuggingFace datasets, returns the HF Dataset object.
        For JSON datasets, returns the list of dictionaries.
        """
        return self.dataset.dataset

    def generate_evaluation_lists(self, retriever, top_k: int = 5):
        """
        Generate lists of questions, contexts, ground_truths, fake_answers, and question_ids for evaluation.

        Args:
            retriever: Retriever instance to get contexts
            top_k: Number of context chunks to retrieve per question (default: 5)

        Returns:
            Tuple of (questions, contexts_list, ground_truths, fake_answers, question_ids) if fake answers exist,
            otherwise (questions, contexts_list, ground_truths, question_ids)
        """
        from tqdm import tqdm

        question_ids = self.get_question_ids()
        questions = []
        contexts_list = []
        ground_truths = []
        fake_answers = []

        for qa_id in tqdm(question_ids, desc="Retrieving contexts"):
            q = self.dataset.get_question_by_id(id=qa_id)
            if self.dataset.source_type == 'huggingface':
                question = q["question"]
                ground_truth = q["answer"]
            else:
                question = q["query"]
                ground_truth = q["answer"]

            # Get fake answer if available
            fake_answer = self.dataset.get_fake_answer(q)

            # Get contexts using the retriever
            contexts = retriever.retrieve(q, top_k=top_k)

            questions.append(question)
            contexts_list.append(contexts)
            ground_truths.append(ground_truth)

            # Only append fake answer if it's not the default "None" string
            if fake_answer != "None":
                fake_answers.append(fake_answer)

        # Return with fake_answers only if we actually have some
        if len(fake_answers) > 0:
            return questions, contexts_list, ground_truths, fake_answers, question_ids
        else:
            return questions, contexts_list, ground_truths, question_ids

