from typing import List, Dict
import random


class MockGenerator:
    """Mock generator for testing evaluation pipeline"""

    def __init__(self, model_name: str = "mock-7b"):
        self.model_name = model_name

    def generate(self, question, contexts: List[str]) -> str:
        """Generate answer based on question and contexts"""
        return "I cannot find the answer in the provided context."
