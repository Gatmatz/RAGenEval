from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def compute(self, **kwargs) -> float:
        """
        Compute the metric between the answer and the ground truth.
        :param answer: the generated answer from an LLM
        :param ground_truth: the expected correct answer
        :return: metric score
        """
        pass