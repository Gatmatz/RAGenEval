from abc import ABC, abstractmethod


class Judge(ABC):
    """
    Abstract base class for evaluating Generator part of RAG systems.
    """
    @abstractmethod
    def bulk_evaluation(self, **kwargs):
        pass