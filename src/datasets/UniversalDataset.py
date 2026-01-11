from dotenv import load_dotenv
import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset
from typing import Union, Optional, Dict, Any, List

load_dotenv()


class UniversalDataset:
    """
    A flexible dataset loader that works with both HuggingFace datasets and local JSON files.

    Usage:
        # For HuggingFace datasets
        dataset = UniversalDataset.from_huggingface("hotpot_qa", "distractor", split="train")

        # For local JSON files
        dataset = UniversalDataset.from_json("path/to/data.json")

        # For local JSON files with custom field mappings
        dataset = UniversalDataset.from_json("path/to/data.json",
                                            id_field="query_id",
                                            question_field="query",
                                            answer_field="answer")
    """

    def __init__(self, dataset: Union[HFDataset, List[Dict]], source_type: str):
        """
        Initialize the dataset.

        Args:
            dataset: Either a HuggingFace Dataset or a list of dictionaries
            source_type: Either 'huggingface' or 'json'
        """
        self.dataset = dataset
        self.source_type = source_type
        self._index = None  # Cache for ID-based lookups

    @classmethod
    def from_huggingface(cls, dataset_path: str, dataset_name: Optional[str] = None,
                        split: Optional[str] = None, token: Optional[str] = None) -> 'UniversalDataset':
        """
        Load a dataset from HuggingFace.

        Args:
            dataset_path: Path to the HuggingFace dataset
            dataset_name: Name/config of the dataset (optional)
            split: Dataset split to use (e.g., 'train', 'validation')
            token: HuggingFace token (optional, defaults to HF_TOKEN env var)
        """
        if token is None:
            token = os.getenv("HF_TOKEN")

        dataset = load_dataset(dataset_path, dataset_name, token=token)

        if split:
            dataset = dataset[split]

        return cls(dataset, 'huggingface')

    @classmethod
    def from_json(cls, json_path: Union[str, Path],
                  id_field: str = "id",
                  question_field: str = "question",
                  answer_field: str = "answer") -> 'UniversalDataset':
        """
        Load a dataset from a local JSON file.

        Args:
            json_path: Path to the JSON file
            id_field: Name of the ID field in the JSON (default: "id")
            question_field: Name of the question field (default: "question")
            answer_field: Name of the answer field (default: "answer")
        """
        json_path = Path(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            # Try to load as JSON array or JSONL (one JSON object per line)
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                # JSONL format
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        # Store field mappings for later use
        instance = cls(data, 'json')
        instance._field_mappings = {
            'id': id_field,
            'question': question_field,
            'answer': answer_field
        }
        return instance

    def set_split(self, split_name: str):
        """
        Set the dataset split (only works for HuggingFace datasets).

        Args:
            split_name: Name of the split (e.g., 'train', 'validation', 'test')
        """
        if self.source_type == 'huggingface':
            self.dataset = self.dataset[split_name]
        else:
            print(f"Warning: set_split() is only applicable to HuggingFace datasets")

    def get_question_by_id(self, id: Any) -> Dict:
        """
        Get a question by its ID.

        Args:
            id: The ID of the question to retrieve

        Returns:
            Dictionary containing the question data
        """
        if self.source_type == 'huggingface':
            return self.dataset.filter(lambda x: x["id"] == id)[0]
        else:
            # For JSON, build an index on first call for efficiency
            if self._index is None:
                id_field = self._field_mappings.get('id', 'id')
                self._index = {item[id_field]: item for item in self.dataset}

            return self._index.get(id)

    def filter(self, condition):
        """
        Filter the dataset based on a condition.

        Args:
            condition: A function that takes an item and returns True/False

        Returns:
            Filtered dataset (behavior depends on source type)
        """
        if self.source_type == 'huggingface':
            return self.dataset.filter(condition)
        else:
            return [item for item in self.dataset if condition(item)]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get an item by index."""
        return self.dataset[idx]

    def __iter__(self):
        """Make the dataset iterable."""
        return iter(self.dataset)

    @staticmethod
    def get_supporting_facts(question: Dict) -> Any:
        """
        Get supporting facts from a question.

        Args:
            question: Question dictionary

        Returns:
            Supporting facts if available, empty dict otherwise
        """
        return question.get('supporting_facts', {})

    @staticmethod
    def get_context(question: Dict) -> Any:
        """
        Get context from a question.

        Args:
            question: Question dictionary

        Returns:
            Context if available, empty dict otherwise
        """
        return question.get('context', {})

    @staticmethod
    def get_positive_contexts(question: Dict) -> List[str]:
        """
        Get positive contexts from a question (for datasets with explicit positive/negative contexts).

        Args:
            question: Question dictionary

        Returns:
            List of positive context strings
        """
        return question.get('positive', [])

    @staticmethod
    def get_negative_contexts(question: Dict) -> List[str]:
        """
        Get negative contexts from a question (for datasets with explicit positive/negative contexts).

        Args:
            question: Question dictionary

        Returns:
            List of negative context strings
        """
        return question.get('negative', [])

    @staticmethod
    def get_positive_wrong_contexts(question: Dict) -> List[str]:
        """
        Get positive wrong contexts from a question (for datasets with explicit positive wrong contexts).

        Args:
            question: Question dictionary

        Returns:
            List of positive wrong context strings
        """
        return question.get('positive_wrong', [])

    @staticmethod
    def get_fake_answer(question: Dict) -> str:
        """
        Get positive wrong contexts from a question (for datasets with explicit fake answer contexts).

        Args:
            question: Question dictionary

        Returns:
            Fake answer string
        """
        return question.get('fakeanswer', "None")