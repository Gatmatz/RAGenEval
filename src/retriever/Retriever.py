from abc import ABC, abstractmethod

class Retriever:
    """
    Abstract base class for Retriever implementations.
    User can extend this class to create custom retriever scenarios.
    """
    @abstractmethod
    def retrieve(self, question, top_k):
        # Placeholder for retrieval logic
        pass